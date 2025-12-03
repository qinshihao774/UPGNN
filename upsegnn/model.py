## In[Import]
import networkx as nx
import torch
import torch.nn as nn
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Subset
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as tDataLoader
from dig.sslgraph.method.contrastive.objectives import JSE_loss, NCE_loss
from upsegnn.metrics import compute_fidelity_plus, compute_fidelity_minus


def set_seed(seed):
    random.seed(seed)  # Python随机数种子
    np.random.seed(seed)  # NumPy随机数种子
    torch.manual_seed(seed)  # PyTorch CPU随机数种子
    torch.cuda.manual_seed(seed)  # PyTorch GPU随机数种子
    torch.cuda.manual_seed_all(seed)  # 多GPU时设置所有GPU的种子
    torch.backends.cudnn.deterministic = True  # 确保CuDNN的确定性
    torch.backends.cudnn.benchmark = False  # 禁用CuDNN的自动优化


class MLP(nn.Module):
    def __init__(self, num_layer, input_dim, hidden_dim, device):
        super(MLP, self).__init__()
        self.num_layer = num_layer
        self.device = device

        self.edge_mlp = nn.ModuleList()  # 边掩码 MLP
        if num_layer > 1:
            self.edge_mlp.append(nn.Linear(2 * input_dim, hidden_dim))  # 边特征维度 = 源+目标节点的嵌入维度
            for n in range(num_layer - 1):
                self.edge_mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.edge_mlp.append(nn.Linear(hidden_dim, 1))  # [edge_nums] 列表[0.5,0.75,0.31...]
        else:
            self.edge_mlp.append(nn.Linear(2 * input_dim, 1))

        # ========== 关键初始化 ==========

    #     self._init_weights()
    #
    # def _init_weights(self):
    #     for layer in self.edge_mlp:
    #         if isinstance(layer, nn.Linear):
    #             # Xavier 初始化权重
    #             nn.init.xavier_uniform_(layer.weight)
    #             # 最后一层偏置初始化为负值 → logits 初始偏负 → mask 偏稀疏
    #             if layer.bias is not None:
    #                 if layer == self.edge_mlp[-1]:  # 最后一层
    #                     nn.init.constant_(layer.bias, -3.0)  # 强烈建议 -2 ~ -5
    #                 else:
    #                     nn.init.constant_(layer.bias, 0.0)
    #             # 为什么负偏置？初始 logits 负 → sigmoid <0.5 → mask 稀疏，便于训练从“全保留”向“选择性保留”学习。

    def forward(self, data, node_embed):
        # node_embed = F.normalize(node_embed, p=2, dim=-1)
        # 边掩码  是否考虑双向边？ 需要根据图的边性质 须考虑有向、无向边的信息传递
        edge_index = data.edge_index
        f1, f2 = node_embed[edge_index[0]], node_embed[edge_index[1]]  # 根据边索引提取源节点和目标节点的表示向量
        edge_embed = torch.cat([f1, f2], dim=-1)  # [num_edges, 2 * emb_dim] 边表示=源节点表示+目标节点表示
        # edge_embed = edge_cat
        # 然后通过一个mlp层最终得到边掩码
        for layer in self.edge_mlp[:-1]:
            edge_embed = F.relu(layer(edge_embed))
        edge_mask = torch.sigmoid(self.edge_mlp[-1](edge_embed)).squeeze(-1)  # [num_edges] 且 值为 (0, 1) 区间。
        # logits = self.edge_mlp[-1](edge_embed).squeeze(-1)  # [num_edges] 且 值为 (0, 1) 区间。

        # 理想：logits mean: 0.0, std: 1.2, max: 5.0
        # print(f"logits mean: {logits.mean():.4f}, std: {logits.std():.4f}, max: {logits.max():.4f}")

        # edge_mask = torch.sigmoid(logits)

        return edge_mask


class Pretrain_Explainer(torch.nn.Module):  # 预训练解释器模型
    def __init__(self, model, num_layer, hidden_dim: int, device, explain_graph: bool = True, coff_size: float = 0.01,
                 coff_ent: float = 5e-4, loss_type='NCE', t0: float = 5.0, t1: float = 1.0):
        super(Pretrain_Explainer, self).__init__()
        self.device = device
        self.explain_graph = explain_graph  # bool
        self.model = model  # .to(device)  # encoder
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim

        if self.model is None:
            self.exp_input_dim = 32  # 默认32
        else:
            self.exp_input_dim = self.model.gnn.out_dim

        self.explainer = MLP(self.num_layer, self.exp_input_dim, self.hidden_dim, self.device)
        self.explainer.to(device)

        # objective parameters for PGExplainer
        self.coff_size = coff_size  # coff_size 太小 → 稀疏惩罚不足  但是如果太大时 → 稀疏 那么生成的解释子图边 呈现出来会很稀疏
        self.coff_ent = coff_ent  # 越大越趋近于 2值化 0 1分布
        self.coff_max = 0.015
        self.t0 = t0
        self.t1 = t1
        self.loss_type = loss_type

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Concrete 分布采样"""
        if training:
            debug_var = 0.0
            bias = 0.0
            random_noise = bias + torch.rand_like(log_alpha) * (1.0 - debug_var)
            gate_inputs = torch.log(random_noise + 1e-10) - torch.log(1.0 - random_noise + 1e-10)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)
        return gate_inputs

    def __loss__(self, data, pred_y, tua, device):
        # /**
        #  * 计算解释器的损失函数。
        #  * @param data 单图数据
        #  * @param pred_y 预测标签
        #  * @param tmp 临时变量
        #  * @param device cup或者cuda设备
        #  * @return 损失函数
        #  */
        # with torch.no_grad():
        embed, node_embed = self.model.gnn(data, isbatch=False)  # 模型参数已经固定不再改变，并得到节点表示向量

        # 经过归一化sigmoid 处理节点掩码列表[num_nodes,]、边掩码列表[num_edges,]
        edge_mask = self.explainer(data, node_embed)  # 每一轮模型都要对这些节点和边进行掩码处理

        edge_mask = torch.clamp(edge_mask, min=0.01, max=0.99)

        # 正常应该是： Edge mask mean: 0.3~0.7, min: 0.01, max: 0.99

        # 每个训练周期动态对每一个图生成负样本进行训练 比 提前生成一堆负样本供整个周期训练 的效果好
        neg_subgraphs = self.generate_negative_subgraphs(data)

        neg_zs = []
        for neg_data in neg_subgraphs:
            neg_data = neg_data.to(device)
            neg_z, _ = self.model.gnn(neg_data, isbatch=False)
            neg_zs.append(neg_z)
        neg_embeds = torch.stack(neg_zs, dim=0)  # [num_neg, emb_dim]

        # 确保edge_mask只有横向一维 默认压缩存在1的维度
        edge_mask = edge_mask.squeeze()

        # F+ 子图：移除高重要性边 非重要结构子图
        # masked_data_plus = fn_softedgemask(data, edge_mask, isFidelitPlus=True)

        # F- 子图：保留高重要性边 重要结构子图
        masked_data_minus = fn_softedgemask(data, edge_mask, isFidelitPlus=False)

        # if masked_data_plus.num_nodes == 0 or masked_data_plus.edge_index.size(1) == 0 or \
        #         masked_data_minus.num_nodes == 0 or masked_data_minus.edge_index.size(1) == 0:
        #     print("警告 masked_data_plus / masked_data_minus error !")
        #     return torch.tensor(0.0, device=device, requires_grad=True)

        if masked_data_minus.num_nodes == 0 or masked_data_minus.edge_index.size(1) == 0:
            print("警告 masked_data_minus error !")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # with torch.no_grad():
        #     masked_pred_plus = self.model(masked_data_plus)
        # F+ 用 1-CE
        # loss_fidelity_plus = F.cross_entropy(masked_pred_plus, pred_y)
        # pos_embed, _ = self.model.gnn(masked_data_minus, isbatch=False)

        with torch.no_grad():
            masked_pred_minus = self.model(masked_data_minus)
        # F- 用 CE
        loss_fidelity_minus = F.cross_entropy(masked_pred_minus, pred_y)
        pos_embed, _ = self.model.gnn(masked_data_minus, isbatch=False)
        loss_infonce = self.info_nce_loss(embed, pos_embed, neg_embeds, tua=tua)

        # 稀疏性正则化 损失越小，越容易学习到稳定的子图
        # coff_size = self.coff_size + (self.coff_max - self.coff_size) * tua
        # loss_reg = coff_size * torch.mean(edge_mask)  # 参数越大 mask掩码越小 经尽可能保留关键边
        loss_reg = self.coff_size * torch.mean(edge_mask)  # 参数越大 mask掩码越小
        # edge_mask = edge_mask * 0.99 + 0.005
        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        loss_ent = self.coff_ent * torch.mean(mask_ent)  # 参数越大 mask掩码越小 让掩码趋近于0 或 1

        # 总损失
        #   loss_fidelity_plus = masked_prob_plus[0, y.item()]：值在 [0, 1]，目标是接近 0（F+ 接近 1）。
        #   loss_infonce：InfoNCE 损失非负，目标是接近 0（正样本与原始图相似，负样本远离）。
        #   loss_reg 和 loss_ent：非负，鼓励 edge_mask 稀疏化。
        #       影响损失关键                  最大可能互信息                 子图规不规范的关键
        # loss = 0.4 * loss_infonce + 0.3 * loss_fidelity_minus + 0.2 * (loss_reg + loss_ent)
        loss = 0.8 * loss_infonce + 0.2 * (loss_reg + loss_ent)

        # 返回拆分
        return {
            'total': loss,
            'fid_loss': loss_fidelity_minus,
            'infonce': loss_infonce,
            'loss_reg': loss_reg,
            'ent': loss_ent
        }

        # return loss

    def __retune_loss__(self, data):
        # /**
        #  * 计算解释器的损失函数。
        #  * @param data 单图数据
        #  * @param true_y 真实标签
        #  * @param device cup或者cuda设备
        #  * @return 损失函数
        #  */
        # with torch.no_grad():
        # criterion = torch.nn.CrossEntropyLoss()
        embed, node_embed = self.model.gnn(data, isbatch=False)  # 模型参数已经固定不再改变，并得到节点表示向量

        # 经过归一化sigmoid 处理节点掩码列表[num_nodes,]、边掩码列表[num_edges,]
        edge_mask = self.explainer(data, node_embed)  # 每一轮模型都要对这些节点和边进行掩码处理

        edge_mask = torch.clamp(edge_mask, min=0.01, max=0.99)

        # print(f"Edge mask mean: {edge_mask.mean():.4f}, min: {edge_mask.min():.4f}, max: {edge_mask.max():.4f}")

        # 确保edge_mask只有横向一维 默认压缩存在1的维度
        edge_mask = edge_mask.squeeze()

        # F+ 子图：移除高重要性边 非重要结构子图
        # masked_data_plus = fn_softedgemask(data, edge_mask, isFidelitPlus=True)
        # F- 子图：保留高重要性边 重要结构子图
        masked_data_minus = fn_softedgemask(data, edge_mask, isFidelitPlus=False)  # 边权重最小0.01 最大0.99
        # masked_data_minus = topk_edge_mask(data, edge_mask, isFidelitPlus=False) # 边权重最小0.0 最大1.0

        with torch.no_grad():
            masked_pred_minus = self.model(masked_data_minus)

        loss_entropy = F.cross_entropy(masked_pred_minus, data.y.long())
        # loss_entropy = criterion(masked_pred_minus, data.y.long())

        loss_reg = 5e-3 * torch.mean(edge_mask)  # 参数越大 mask掩码越小
        # 熵项（逐元素）
        entropy = -edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        # 完整损失：熵 + L1
        loss_ent = entropy.sum() + edge_mask.sum()

        reg_ent = loss_reg + 0.02 * loss_ent
        loss = loss_entropy + 0.1 * reg_ent
        # loss = loss_ent

        return {
            'total': loss,
            'ent': reg_ent
        }

        # return loss

    def __batched_JSE__(self, cond_embed, cond_pruned_embed, batch_size):  # 批量计算JSE损失
        loss = 0
        for i, (z1, z2) in enumerate(tDataLoader(
                TensorDataset(cond_embed, cond_pruned_embed), batch_size)):
            if len(z1) <= 1:
                i -= 1
                break
            loss += JSE_loss([z1, z2])
        return loss / (i + 1.0)

    # def triplet_loss_safe(self, embed_orig, pos_embeds, neg_embeds, tmp=1.0):
    #     """
    #     完全兼容：
    #     loss = pe.triplet_loss_safe(embed, pos_embed.unsqueeze(0), neg_embeds, tmp)
    #     """
    #     device = embed_orig.device
    #     # 不管你传进来什么鬼形状，我都要把它变成 [D]、[1,D]、[N,D]
    #     embed_orig = embed_orig.view(-1)
    #     pos_embeds = pos_embeds.view(-1, embed_orig.size(-1))
    #     neg_embeds = neg_embeds.view(-1, embed_orig.size(-1))
    #
    #     #     # 归一化嵌入
    #     anchor = embed_orig / (torch.norm(embed_orig, dim=-1, keepdim=True) + 1e-6)
    #     pos = pos_embeds / (torch.norm(pos_embeds, dim=-1, keepdim=True) + 1e-6)
    #     neg = neg_embeds / (torch.norm(neg_embeds, dim=-1, keepdim=True) + 1e-6)
    #
    #     # 2. 距离 = 1 - cos_sim
    #     pos_dist = 1 - (anchor * pos).sum(dim=-1)  # [1]
    #     # neg_dist = 1 - torch.matmul(anchor.unsqueeze(0), neg.T)  # [1, N]
    #     neg_dist = 1 - (anchor.unsqueeze(0) * neg).sum(dim=-1)
    #     # 3. Triplet：正样本要近，负样本要远
    #     # loss = F.relu(neg_dist - pos_dist)  # [1, N]
    #
    #     # pos_dist = pos_dist.mean()
    #     neg_dist = neg_dist.mean()
    #
    #     loss = -torch.log(pos_dist / (neg_dist + pos_dist + 1e-6))
    #     # loss = loss.mean()  # 标量
    #
    #     # 4. 终极防炸
    #     # print("loss: ", loss)
    #     return loss if torch.isfinite(loss) else torch.tensor(0.0, device=device, requires_grad=True)

    def info_nce_loss(self, embed_orig, pos_embed, neg_embeds, tua=1.0):
        """
        完美复现公式，兼容多正样本，永不 NaN！
        """
        device = embed_orig.device

        # Step 1: 也就是说 不管传进来是怎么样的一种形状，我都要把它变成 [D]、[1,D]、[N,D]
        embed_orig = embed_orig.view(-1)  # [D]
        pos_embed = pos_embed.view(-1, embed_orig.size(-1))  # [P, D]
        neg_embeds = neg_embeds.view(-1, embed_orig.size(-1))  # [N, D]

        # Step 2: L2 归一化 → 余弦相似度
        anchor = F.normalize(embed_orig, dim=-1)  # [D]
        pos = F.normalize(pos_embed, dim=-1)  # [P, D]
        neg = F.normalize(neg_embeds, dim=-1)  # [N, D]

        # Step 3: 相似度 + 温度缩放（公式核心！）
        pos_sim = torch.sum(anchor * pos, dim=-1) / tua  # [P]
        neg_sim = torch.matmul(anchor, neg.T) / tua  # [N]

        # Step 4: 拼接所有样本
        all_sim = torch.cat([pos_sim, neg_sim], dim=0)  # [P+N] 数组形式

        exp_logits = torch.exp(all_sim)

        # 5. 公式原装：-log(正样本总概率)
        pos_prob = exp_logits[:pos.size(0)].sum()
        total = exp_logits.sum()
        loss = -torch.log(pos_prob / (total + 1e-8))  # 因为pos_prob概率一定小于total 所以log（小于1）为负数

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=embed_orig.device, requires_grad=True)

        return loss

    # def generate_negative_subgraphs(self, data, num_negatives=3, edge_drop_ratio=0.7, feature_noise=0.1):
    # 
    #     """
    #     生成负样本子图，通过边删除和特征扰动增加区分度。
    #     要求：1、且负样本也不能完全丢失太多边，不然也会丢失太多信息，导致负样本没有任何参考性，导致模型学坏
    #          2、节点扰动也不能太大，与边丢弃同理
    #     Args:
    #         data: PyG Data 对象
    #         num_negatives: 负样本数量
    #         edge_drop_ratio: 边删除比例
    #         feature_noise: 特征扰动标准差
    #     Returns:
    #         neg_subgraphs: 负样本子图列表
    #     """
    #     neg_subgraphs = []
    #     num_edges = data.edge_index.size(1)
    #     device = data.x.device
    # 
    #     for _ in range(num_negatives):
    #         # 随机删除边（更强扰动）
    #         keep_ratio = 1 - edge_drop_ratio
    #         # 执行伯努利随机采样
    #         neg_mask = torch.bernoulli(torch.ones(num_edges, device=device) * keep_ratio).bool()
    #         neg_edge_index = data.edge_index[:, neg_mask]
    # 
    #         # 获取保留边的节点
    #         active_nodes = torch.unique(neg_edge_index)  # 获取仍与边相连的节点索
    # 
    #         # 特征扰动 没有移除节点
    #         # neg_x = data.x.float() + torch.randn_like(data.x.float()) * feature_noise
    #         # neg_x = torch.clamp(neg_x, min=-1.0, max=1.0)  # 防止数值溢出
    # 
    #         # 特征扰动，仅保留活跃节点的特征
    #         neg_x = data.x.float()[active_nodes] + torch.randn_like(data.x.float()[active_nodes]) * feature_noise
    #         neg_x = torch.clamp(neg_x, min=-1.0, max=1.0)  # 防止数值溢出
    # 
    #         # # 重新映射边索引以匹配新的节点索引
    #         node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(active_nodes)}
    #         neg_edge_index = torch.tensor(
    #             [[node_mapping[idx.item()] for idx in neg_edge_index[0]],
    #              [node_mapping[idx.item()] for idx in neg_edge_index[1]]],
    #             device=device, dtype=torch.long
    #         )
    # 
    #         neg_data = Data(
    #             x=neg_x,
    #             edge_index=neg_edge_index,
    #             num_nodes=neg_x.size(0)
    #         ).to(device)
    # 
    #         neg_subgraphs.append(neg_data)
    # 
    #     return neg_subgraphs

    def generate_negative_subgraphs(self, data, num_negatives=3, edge_drop_ratio=0.7, feature_noise=0.1):

        """
        生成负样本子图，通过边删除和特征扰动增加区分度。
        要求：1、且负样本也不能完全丢失太多边，不然也会丢失太多信息，导致负样本没有任何参考性，导致模型学坏
             2、节点扰动也不能太大，与边丢弃同理
        Args:
            data: PyG Data 对象
            num_negatives: 负样本数量
            edge_drop_ratio: 边删除比例
            feature_noise: 特征扰动标准差
        Returns:
            neg_subgraphs: 负样本子图列表
        """
        neg_subgraphs = []
        device = data.x.device
        edges = data.edge_index.t()  # [E, 2]
        E = edges.size(0)

        # ====== 1. 构建真边 mask（向量化，一次搞定）======
        keep_gt = torch.ones(E, dtype=torch.bool, device=device)  # 默认全保留
        if hasattr(data, 'exp_gt') and data.exp_gt is not None and data.exp_gt.numel() > 0:
            gt = data.exp_gt if data.exp_gt.dim() == 2 else data.exp_gt.view(-1, 2)
            gt_pairs = {tuple(sorted(p.tolist())) for p in gt}
            edge_pairs = [tuple(sorted(e.tolist())) for e in edges]
            is_gt = torch.tensor([p in gt_pairs for p in edge_pairs], device=device)
            keep_gt = ~is_gt  # 真边 100% 不保留！

        for _ in range(num_negatives):
            # # 随机删除边（更强扰动）
            # keep_ratio = 1 - edge_drop_ratio
            # # 执行伯努利随机采样
            # neg_mask = torch.bernoulli(torch.ones(num_edges, device=device) * keep_ratio).bool()
            # neg_edge_index = data.edge_index[:, neg_mask]

            # 先强制剔除gt边
            mask = keep_gt.clone()

            # 再对剩余边随机删
            if mask.any():  # 还有非gt边
                remain_ratio = 1.0 - edge_drop_ratio
                extra_keep = torch.bernoulli(torch.full_like(mask, remain_ratio, dtype=torch.float)).bool()
                mask = mask & extra_keep

            # # 防空图：至少保留 5 条非gt边
            if mask.sum() < 5:
                non_gt_idx = torch.where(keep_gt)[0]
                if non_gt_idx.numel() >= 5:
                    extra = non_gt_idx[torch.randperm(non_gt_idx.numel())[:3]]
                    temp = torch.zeros(E, dtype=torch.bool, device=device)
                    temp[extra] = True
                    mask = temp

            neg_edge_index = edges[mask].t()  # [2, E']

            # 获取保留边的节点
            active_nodes = torch.unique(neg_edge_index)  # 获取仍与边相连的节点索

            # 特征扰动，仅保留活跃节点的特征
            neg_x = data.x.float()[active_nodes] + torch.randn_like(data.x.float()[active_nodes]) * feature_noise
            # neg_x = torch.clamp(neg_x, min=-1.0, max=1.0)  # 防止数值溢出

            # # 重新映射边索引以匹配新的节点索引
            node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(active_nodes)}
            neg_edge_index = torch.tensor(
                [[node_mapping[idx.item()] for idx in neg_edge_index[0]],
                 [node_mapping[idx.item()] for idx in neg_edge_index[1]]],
                device=device, dtype=torch.long
            )

            neg_data = Data(
                x=neg_x,
                edge_index=neg_edge_index,
                num_nodes=neg_x.size(0)
            ).to(device)

            neg_subgraphs.append(neg_data)

        return neg_subgraphs

    def generate_explanation(self, dataset, device):
        self.model.eval()  # classfier
        self.explainer.eval()

        sub_dataset = Subset(dataset, range(15))  # 前 100 个
        for data in sub_dataset:
            # for data in valid_dataset:
            data = data.to(device)
            print("data.y", data.y)
            with torch.no_grad():
                embed, node_embed = self.model.gnn(data, isbatch=False)
                edge_mask = self.explainer(data, node_embed)  # 节点掩码

            self.visualize_subgraph(data, edge_mask, 12)

    def visualize_subgraph(self, data, edge_mask, k=10):
        """
        可视化GNN解释子图，突出显示前k条边，节点按特征类型用不同颜色区分。

        参数：
            data (Data): PyG的Data对象，包含edge_index、x
            edge_mask (torch.Tensor): 边重要性掩码，形状[num_edges]
            k (int): 选择前k条边，默认为15
        """
        # 确保掩码为一维
        # node_mask = node_mask.squeeze()
        edge_mask = edge_mask.squeeze()

        # 获取前k条边（值最大的边）
        k = min(k, edge_mask.shape[0])  # 避免k超过边数
        _, selected_edge_indices = torch.topk(edge_mask, k=k, largest=True)
        masked_edge_index = data.edge_index[:, selected_edge_indices]

        # 创建NetworkX图
        G = nx.Graph()
        num_nodes = data.x.shape[0] if hasattr(data, 'x') and data.x is not None else 0

        G.add_nodes_from(range(num_nodes))
        edge_list = data.edge_index.t().cpu().numpy().tolist()
        G.add_edges_from(edge_list)

        # 创建子图（只包含前k条边）
        G_sub = nx.Graph()
        sub_edge_list = masked_edge_index.t().cpu().numpy().tolist()
        G_sub.add_edges_from(sub_edge_list)

        # 节点颜色：基于data.x（节点特征）通过KMeans聚类区分类型
        node_colors = ['lightblue' for _ in range(num_nodes)]  # 默认颜色
        if hasattr(data, 'x') and data.x is not None and data.x.shape[0] > 0:
            node_features = data.x.cpu().numpy()
            # 检查特征是否有效（无NaN、无Inf、节点数足够）
            if np.any(np.isnan(node_features)) or np.any(np.isinf(node_features)):
                print("Warning: node_features contains NaN or Inf. Using default colors.")
            elif num_nodes < 2:
                print("Warning: Too few nodes ({}) for clustering. Using default colors.".format(num_nodes))
            else:
                # 动态设置簇数（不超过节点数）
                n_clusters = min(3, num_nodes)
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                    kmeans.fit(node_features)  # 确保拟合
                    node_labels = kmeans.labels_  # 获取聚类标签
                    unique_labels = set(node_labels)
                    color_map = plt.cm.get_cmap('Set1', max(len(unique_labels), 1))  # 避免0簇
                    node_colors = [color_map(node_labels[i]) for i in range(num_nodes)]
                    print(f"KMeans successful: {n_clusters} clusters, labels: {node_labels}")
                except Exception as e:
                    print(f"KMeans failed: {e}. Using default colors.")
        else:
            print("No valid node features (data.x). Using default colors.")

        # 绘制图
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)  # 固定种子以确保布局可重现

        # 绘制原图（淡色）
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.3)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2)
        nx.draw_networkx_labels(G, pos, font_size=8)

        # 绘制子图（只高亮边）
        nx.draw_networkx_edges(G_sub, pos, edge_color='red', width=2.0)

        plt.title("GNN Explanation: Original Graph (gray) and Important Edges (red)")
        plt.show()

    def fn_softedgemask(self, data, edge_mask, isFidelitPlus=True):
        # 如果 data.edge_attr 为 None，创建默认边特征（全 1）
        # if data.edge_attr is None:
        #     num_edges = data.edge_index.size(1)
        #     edge_attr = torch.ones((num_edges, 1), device=edge_mask.device, dtype=edge_mask.dtype)
        # else:
        #     edge_attr = data.edge_attr

        # masked_x = data.x * node_mask.unsqueeze(1)  # 广播至节点维度[num,dim]

        num_edges = data.edge_index.size(1)
        edge_weight = torch.ones((num_edges, 1), device=edge_mask.device, dtype=edge_mask.dtype)
        # # 应用边掩码
        # masked_edge_attr = (edge_mask if not isFidelitPlus else 1 - edge_mask) * edge_attr

        # 应用边掩码 isFidelitPlus 1 则执行else
        # edge_mask: [num_edges], edge_attr: [num_edges, num_features]
        masked_edge_weight = edge_mask.view(-1, 1) * edge_weight if not isFidelitPlus else (1 - edge_mask.view(-1,
                                                                                                               1)) * edge_weight

        # 创建新的 Data 对象
        masked_data = data.clone()
        masked_data.edge_weight = masked_edge_weight

        return masked_data

    def __edge_mask_to_node__(self, data, edge_mask, top_k):  # 利用边掩码做节点掩码
        threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0] - 1)])
        hard_mask = (edge_mask > threshold).cpu()
        edge_idx_list = torch.where(hard_mask == 1)[0]

        selected_nodes = []
        edge_index = data.edge_index.cpu().numpy()
        for edge_idx in edge_idx_list:
            selected_nodes += [edge_index[0][edge_idx], edge_index[1][edge_idx]]
        selected_nodes = list(set(selected_nodes))
        maskout_nodes = [node for node in range(data.x.shape[0]) if node not in selected_nodes]

        node_mask = torch.zeros(data.num_nodes).type(torch.float32).to(self.device)
        node_mask[maskout_nodes] = 1.0
        return node_mask

    # def forward(self, data: Data, mlp_explainer: nn.Module, **kwargs):
    #     """ explain the GNN behavior for graph and calculate the metric values.
    #     The interface for the :class:`dig.evaluation.XCollector`.
    #
    #     Args:
    #         x (:obj:`torch.Tensor`): Node feature matrix with shape
    #           :obj:`[num_nodes, dim_node_feature]`
    #         edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
    #           with shape :obj:`[2, num_edges]`
    #         kwargs(:obj:`Dict`):
    #           The additional parameters
    #             - top_k (:obj:`int`): The number of edges in the final explanation results
    #             - y (:obj:`torch.Tensor`): The ground-truth labels
    #
    #     :rtype: (:obj:`None`, List[torch.Tensor], List[Dict])
    #     """
    #     top_k = kwargs.get('top_k') if kwargs.get('top_k') is not None else 10
    #     node_idx = kwargs.get('node_idx')
    #     # cond_vec = kwargs.get('cond_vec')
    #
    #     self.model.eval()
    #     mlp_explainer = mlp_explainer.to(self.device).eval()
    #     data = data.to(self.device)
    #
    #     self.__clear_masks__()
    #     if node_idx is not None:
    #         _, node_embed = self.model(data, isbatch=True)
    #         embed = node_embed[node_idx:node_idx + 1]
    #     elif self.explain_graph:
    #         embed, node_embed = self.embed_model(data, emb=True)
    #     else:
    #         assert node_idx is not None, "please input the node_idx"
    #
    #     probs = mlp_explainer(embed, mode='pred')
    #     # node_mask, edge_mask = mlp_explainer(embed, mode='explain')  # if cond_vec is None else cond_vec
    #     probs = probs.squeeze()
    #
    #     if self.explain_graph:  # 图
    #         subgraph = None
    #         target_class = torch.argmax(probs) if data.y is None else max(data.y.long(),
    #                                                                       0)  # sometimes labels are +1/-1
    #         # _, _, edge_mask, log = self.explain(data, embed=node_embed, training=False)
    #         node_mask, edge_mask = mlp_explainer(embed, mode='explain')  # if cond_vec is None else cond_vec
    #         # node_mask = self.__edge_mask_to_node__(data, edge_mask, top_k)  # 设置子图节点掩码
    #         masked_data = mask_fn_nodemask(data, node_mask)  # 利用掩码构建子图
    #         masked_embed = self.embed_model(masked_data)  # 构建子图表示
    #         masked_prob = mlp_explainer(masked_embed, mode='pred')  # 预测的结果概率
    #         masked_prob = masked_prob[:, target_class]  # 从子图的预测概率分布 masked_prob 中，提取出与目标类别 target_class 对应的概率值。
    #         sparsity_score = sum(node_mask) / data.num_nodes  # 稀疏度
    #     else:  # 节点
    #         target_class = torch.argmax(probs) if data.y is None else max(data.y[node_idx].long(),
    #                                                                       0)  # sometimes labels are +1/-1
    #         subgraph, subset = self.get_subgraph(node_idx=node_idx, data=data)
    #         new_node_idx = torch.where(subset == node_idx)[0]
    #         _, edge_mask, log = self.explain(subgraph, node_embed[subset], training=False, node_idx=new_node_idx)
    #         node_mask = self.__edge_mask_to_node__(subgraph, edge_mask, top_k)
    #         masked_embed = self.model(mask_fn_nodemask(subgraph, node_mask))
    #         masked_prob = mlp_explainer(masked_embed, mode='pred')[new_node_idx, target_class.long()]
    #         sparsity_score = sum(node_mask) / subgraph.num_nodes
    #
    #     # return variables
    #     pred_mask = edge_mask.cpu()
    #
    #     related_preds = [{
    #         'maskout': masked_prob.item(),  # 子图预测概率
    #         'origin': probs[target_class].item(),  # 原始类别的概率
    #         'sparsity': sparsity_score}]  # 稀疏度
    #     return subgraph, pred_mask, related_preds


def generate_negative_subgraph(data, num_negatives=2, edge_drop_ratio=0.6, feature_noise=0.1) -> {Data}:
    """
    生成负样本子图，通过边删除和特征扰动增加区分度。
    要求：1、且负样本也不能完全丢失太多边，不然也会丢失太多信息，导致负样本没有任何参考性，导致模型学坏
         2、节点扰动也不能太大，与边丢弃同理
    Args:
        data: PyG Data 对象
        num_negatives: 负样本数量
        edge_drop_ratio: 边删除比例
        feature_noise: 特征扰动标准差
    Returns:
        neg_subgraphs: 负样本子图列表
    """
    neg_subgraphs = []
    num_edges = data.edge_index.size(1)
    device = data.x.device
    edge_weight = torch.ones(num_edges, device=data.x.device)

    for _ in range(num_negatives):
        # 随机删除边（更强扰动）
        keep_ratio = 1 - edge_drop_ratio
        # 执行伯努利随机采样
        neg_mask = torch.bernoulli(torch.ones(num_edges, device=device) * keep_ratio).bool()
        neg_edge_index = data.edge_index[:, neg_mask]

        # 获取保留边的节点
        active_nodes = torch.unique(neg_edge_index)  # 获取仍与边相连的节点索引

        # 特征扰动 没有移除节点
        # neg_x = data.x.float() + torch.randn_like(data.x.float()) * feature_noise
        # neg_x = torch.clamp(neg_x, min=-1.0, max=1.0)  # 防止数值溢出

        # 特征扰动，仅保留活跃节点的特征
        neg_x = data.x.float()[active_nodes] + torch.randn_like(data.x.float()[active_nodes]) * feature_noise
        neg_x = torch.clamp(neg_x, min=-1.0, max=1.0)  # 防止数值溢出

        # # 重新映射边索引以匹配新的节点索引
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(active_nodes)}
        neg_edge_index = torch.tensor(
            [[node_mapping[idx.item()] for idx in neg_edge_index[0]],
             [node_mapping[idx.item()] for idx in neg_edge_index[1]]],
            device=device, dtype=torch.long
        )

        neg_data = Data(
            x=neg_x,
            edge_index=neg_edge_index,
            batch=data.batch
        ).to(device)

        neg_subgraphs.append(neg_data)

    return neg_subgraphs


# def generate_negative_subgraph(data, num_negatives=3, edge_drop_ratio=0.3, feature_noise=0.1):
#     """
#     生成负样本子图，通过边删除和特征扰动增加区分度。
#     Args:
#         data: PyG Data 对象
#         num_negatives: 负样本数量
#         edge_drop_ratio: 边删除比例
#         feature_noise: 特征扰动标准差
#     Returns:
#         neg_subgraphs: 负样本子图列表
#     """
#     neg_subgraphs = []
#     num_edges = data.edge_index.size(1)
#     device = data.x.device
#
#     for _ in range(num_negatives):
#         # 随机删除边（更强扰动）
#         keep_ratio = 1 - edge_drop_ratio
#         neg_mask = torch.bernoulli(torch.ones(num_edges, device=device) * keep_ratio).bool()
#         neg_edge_index = data.edge_index[:, neg_mask]
#
#         # 特征扰动
#         neg_x = data.x.float() + torch.randn_like(data.x.float()) * feature_noise
#         neg_x = torch.clamp(neg_x, min=-1.0, max=1.0)  # 防止数值溢出
#
#         neg_data = Data(
#             x=neg_x,
#             edge_index=neg_edge_index,
#             batch=data.batch
#         ).to(device)
#
#         neg_subgraphs.append(neg_data)
#
#     return neg_subgraphs

def explainer_loss(classifier, explainer, data, y, tmp, device):
    # /**
    #  * 计算解释器的损失函数。
    #  *
    #  * @param classifier 模型
    #  * @param explainer 解释器
    #  * @param data 单图数据
    #  * @param y 预测标签
    #  * @param device cup或者cuda设备
    #  * @return 损失函数
    #  */

    # classifier.eval()
    gcn = classifier.gnn.to(device)

    with torch.no_grad():
        embed, node_embed = gcn(data, isbatch=False)

    # 经过归一化sigmoid 处理节点掩码列表[num_nodes,]、边掩码列表[num_edges,]
    node_mask, edge_mask = explainer(data, node_embed)

    # 是每个训练周期都生成负样本 还是提前生成负样本在供整个训练周期的使用 !!??
    neg_subgraphs = generate_negative_subgraph(data)

    neg_zs = []
    for neg_data in neg_subgraphs:
        neg_data = neg_data.to(device)
        neg_z, _ = gcn(neg_data, isbatch=False)
        neg_zs.append(neg_z)
    neg_embeds = torch.stack(neg_zs, dim=0)  # [num_neg, emb_dim]

    # 确保edge_mask只有横向一维 默认压缩存在1的维度
    edge_mask = edge_mask.squeeze()

    # F+ 子图：移除高重要性边
    masked_data_plus = fn_softedgemask(data, edge_mask, isFidelitPlus=True)

    # F- 子图：保留高重要性边
    masked_data_minus = fn_softedgemask(data, edge_mask, isFidelitPlus=False)
    #
    if masked_data_plus.num_nodes == 0 or masked_data_plus.edge_index.size(1) == 0 or \
            masked_data_minus.num_nodes == 0 or masked_data_minus.edge_index.size(1) == 0:
        print("警告 masked_data_plus / masked_data_minus error !")
        return torch.tensor(0.0, device=device, requires_grad=True)

    with torch.no_grad():
        masked_pred_plus = classifier(masked_data_plus)
        # masked_prob_plus = F.softmax(masked_pred_plus, dim=1)
        # loss_fidelity_plus = masked_prob_plus[0, y.item()]
        loss_fidelity_plus = F.cross_entropy(masked_pred_plus, y)

        # F- 损失：子图保留关键边后预测应一致
        # masked_pred_minus = classifier(masked_data_minus)
        # masked_prob_minus = F.softmax(masked_pred_minus, dim=1)
        # loss_fidelity_minus = 1 - masked_prob_minus[0, y.item()]
        # loss_fidelity_minus = -F.cross_entropy(masked_pred_minus, y)  # 负值鼓励预测一致

    embed_sub, _ = gcn(masked_data_minus, isbatch=False)

    loss_infonce = info_nce_loss(embed, embed_sub.unsqueeze(0), neg_embeds, tmp=tmp)

    if loss_infonce is None:
        # print("警告: 损失为 NaN 或 Inf，返回 0")
        return None

    # 稀疏性正则化         coff_size: float = 0.01, coff_ent: float = 5e-4
    loss_reg = 0.01 * torch.mean(edge_mask)
    edge_mask = edge_mask * 0.99 + 0.005
    mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
    loss_ent = 5e-4 * torch.mean(mask_ent)

    # loss_reg = edge_mask.abs().mean()
    # edge_mask = edge_mask * 0.99 + 0.005  # 避免 log(0)
    # loss_ent = -torch.mean(edge_mask * torch.log(edge_mask) + (1 - edge_mask) * torch.log(1 - edge_mask))

    # 总损失
    #   loss_fidelity_plus = masked_prob_plus[0, y.item()]：值在 [0, 1]，目标是接近 0（F+ 接近 1）。
    #   loss_infonce：InfoNCE 损失非负，目标是接近 0（正样本与原始图相似，负样本远离）。
    #   loss_reg 和 loss_ent：非负，鼓励 edge_mask 稀疏化。

    # loss = loss_fidelity_plus + 0.2 * loss_infonce + 0.05 * (loss_reg + 0.01 * loss_ent)
    loss = loss_fidelity_plus + 0.5 * loss_infonce + (loss_reg + loss_ent)

    return loss


def mutiple_embed_explainer_loss(classifiers, explainer, data, y, tmp, device):
    # 融合多个分类器的gnn表示
    embeds = []
    node_embeds = []
    with torch.no_grad():
        for classifier in classifiers:
            gcn = classifier.gnn.to(device)
            embed_single, node_embed_single = gcn(data, isbatch=False)
            embeds.append(embed_single)
            node_embeds.append(node_embed_single)

    # 平均融合（维度不变）
    embed = torch.mean(torch.stack(embeds), dim=0)
    node_embed = torch.mean(torch.stack(node_embeds), dim=0)

    # 其余代码不变：生成掩码
    node_mask, edge_mask = explainer(data, node_embed)
    edge_mask = edge_mask.squeeze()

    # 动态生成负样本（不变）
    neg_subgraphs = generate_negative_subgraph(data)
    neg_zs = []
    for neg_data in neg_subgraphs:
        neg_data = neg_data.to(device)
        # 对于负样本，也融合多个gnn的embed
        neg_embeds = []
        for classifier in classifiers:
            gcn = classifier.gnn.to(device)
            neg_z_single, _ = gcn(neg_data, isbatch=False)
            neg_embeds.append(neg_z_single)
        neg_z = torch.mean(torch.stack(neg_embeds), dim=0)
        neg_zs.append(neg_z)
    neg_embeds = torch.stack(neg_zs, dim=0)  # [num_neg, emb_dim]

    # F+ 子图（不变）
    masked_data_plus = fn_softedgemask(data, edge_mask, isFidelitPlus=True)

    # F- 子图（不变）
    masked_data_minus = fn_softedgemask(data, edge_mask, isFidelitPlus=False)

    if masked_data_plus.num_nodes == 0 or masked_data_plus.edge_index.size(1) == 0 or \
            masked_data_minus.num_nodes == 0 or masked_data_minus.edge_index.size(1) == 0:
        print("警告 masked_data_plus / masked_data_minus error !")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 融合分类器预测（不变，但用平均pred）
    masked_preds_plus = [classifier(masked_data_plus) for classifier in classifiers]
    masked_pred_plus = torch.mean(torch.stack(masked_preds_plus), dim=0)
    loss_fidelity_plus = F.cross_entropy(masked_pred_plus, y)

    # 对于F-，融合embed_sub
    embed_subs = []
    for classifier in classifiers:
        gcn = classifier.gnn.to(device)
        embed_sub_single, _ = gcn(masked_data_minus, isbatch=False)
        embed_subs.append(embed_sub_single)
    embed_sub = torch.mean(torch.stack(embed_subs), dim=0)

    # InfoNCE损失（不变）
    loss_infonce = info_nce_loss(embed, embed_sub.unsqueeze(0), neg_embeds, tmp=tmp)

    if loss_infonce is None:
        return None

    # 稀疏性正则化（不变）
    loss_reg = 0.01 * torch.mean(edge_mask)
    edge_mask = edge_mask * 0.99 + 0.005
    mask_ent = -edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
    loss_ent = 5e-4 * torch.mean(mask_ent)

    # 总损失（不变）
    loss = loss_fidelity_plus + 0.6 * loss_infonce + (loss_reg + loss_ent)
    return loss


def train(pe, train_dataset, val_dataset, logger, save_path, device, epochs=10):
    print("开始训练...数据长度为:", len(train_dataset))
    # model.train()
    # classifier.to(device)
    pe.to(device)
    optimizer = torch.optim.Adam(pe.explainer.parameters(), lr=0.0005, weight_decay=1e-5)  # lr=0.01,0.001  原 0.0001

    # 记录最佳 F+
    best_fid_plus = -1.0
    # best_epoch = -1

    for epoch in range(epochs):
        # ==================== 训练阶段 ====================
        tua = float(pe.t0 * np.power(pe.t1 / pe.t0, epoch / epochs))
        pe.explainer.train()
        pe.model.eval()
        total_loss = 0.0

        log = {'total': 0.0, 'fid_loss': 0.0, 'infonce': 0.0, 'loss_reg': 0.0, 'ent': 0.0}
        for data in train_dataset:
            data = data.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                pred = pe.model(data)
                pred_y = torch.argmax(pred, dim=-1)  # 0 1

            # loss = pe.__loss__(data, pred_y, tmp, device)
            loss_dict = pe.__loss__(data, pred_y, tua, device)

            # loss.backward()
            loss_dict['total'].backward()
            optimizer.step()
            # total_loss += loss_dict['total'].item()
            # 累加
            for k in log:
                log[k] += loss_dict[k].item()

            # n_samples += 1

        # print(f'Epoch: {epoch + 1:03d}, Train Loss: {total_loss:.4f}')
        # ==================== 验证阶段 ====================
        # with torch.no_grad():
        avg_fid_plus = compute_fidelity_plus(pe.model, pe, val_dataset, device)
        avg_fid_minus = compute_fidelity_minus(pe.model, pe, val_dataset, device)
        #
        # # print(f'Epoch: {epoch + 1:03d} | '
        # #       f'Total Train Loss: {total_loss:.4f} | '
        # #       f'Val F+: {avg_fid_plus:.4f} | '
        # #       f'Val F-: {avg_fid_minus:.4f}')
        #

        # 打印日志
        log_line = (f"epoch: {epoch + 1:3d} | total loss: {log['total']:8.4f} | "
                    f"fid_loss: {log['fid_loss']:6.4f} | infonce: {log['infonce']:7.4f} | "
                    f"loss_reg: {log['loss_reg']:6.4f} | ent: {log['ent']:6.4f} | "
                    f"avg f+: {avg_fid_plus:.4f} | avg f-: {avg_fid_minus:.4f} ")
        logger.info(log_line)

        # 判断是否保存最佳模型
        # if avg_fid_plus > best_fid_plus:
        #     best_fid_plus = avg_fid_plus
        #     best_epoch = epoch + 1
        #     torch.save(pe.explainer.state_dict(), save_path)
        #     print(f"New best model saved! F+ = {avg_fid_plus:.4f},best_epoch at {best_epoch}")

    # torch.save(pe.explainer.state_dict(), save_path)

    # print(f"\nTraining finished. Best Val F+: {best_fid_plus:.4f}, Best Val F-: {best_fid_minus:.4f}")


def retune(pe, train_dataset, device, epochs=3):
    pe.to(device)
    # # 验证解释器参数的requires_grad状态
    # print("解释器参数的requires_grad状态：")
    # for name, param in pe.explainer.named_parameters():
    #     print(f"{name}: {param.requires_grad}")  # 应均为True

    optimizer = torch.optim.Adam(pe.explainer.parameters(), lr=5e-5, weight_decay=1e-5)  # lr=0.01,0.001  原 0.0001

    for epoch in range(epochs):
        # ==================== 微调阶段 ====================
        pe.explainer.train()
        pe.model.eval()
        log = {'total': 0.0, 'ent': 0.0}
        for data in train_dataset:
            data = data.to(device)
            optimizer.zero_grad()
            # true_y = data.y
            # true_y = torch.where(data.y == -1, torch.tensor(0.0), data.y).squeeze()

            loss_dict = pe.__retune_loss__(data)

            loss_dict['total'].backward()
            optimizer.step()
            for k in log:
                log[k] += loss_dict[k].item()

        print(f"Epoch:{epoch + 1:3d} | total loss: {log['total']:8.4f} | ent: {log['ent']:6.4f} ")


def generate_explanation(classifier, explainer, dataset, device):
    classifier.eval()
    classifier.to(device)
    explainer.eval()
    explainer.to(device)

    for data in dataset[0:20]:
        # for data in valid_dataset:
        data = data.to(device)
        with torch.no_grad():
            embed, node_embed = classifier.gnn(data, isbatch=False)
            edge_mask = explainer(data, node_embed)  # 节点掩码

        visualize_subgraph(data, edge_mask)


def visualize_subgraph(data, edge_mask, k=8):
    """
    可视化GNN解释子图，突出显示前k条边，节点按特征类型用不同颜色区分。

    参数：
        data (Data): PyG的Data对象，包含edge_index、x
        edge_mask (torch.Tensor): 边重要性掩码，形状[num_edges]
        k (int): 选择前k条边，默认为15
    """
    # 确保掩码为一维
    # node_mask = node_mask.squeeze()
    set_seed(42)  # 确保 KMeans 和 NetworkX 一致
    edge_mask = edge_mask.squeeze()

    # 获取前k条边（值最大的边）
    k = min(k, edge_mask.shape[0])  # 避免k超过边数
    _, selected_edge_indices = torch.topk(edge_mask, k=k, largest=True)
    masked_edge_index = data.edge_index[:, selected_edge_indices]

    # 创建NetworkX图
    G = nx.Graph()
    num_nodes = data.x.shape[0] if hasattr(data, 'x') and data.x is not None else 0

    G.add_nodes_from(range(num_nodes))
    edge_list = data.edge_index.t().cpu().numpy().tolist()
    G.add_edges_from(edge_list)

    # 创建子图（只包含前k条边）
    G_sub = nx.Graph()
    sub_edge_list = masked_edge_index.t().cpu().numpy().tolist()
    G_sub.add_edges_from(sub_edge_list)

    # 节点颜色：基于data.x（节点特征）通过KMeans聚类区分类型
    node_colors = ['lightblue' for _ in range(num_nodes)]  # 默认颜色
    if hasattr(data, 'x') and data.x is not None and data.x.shape[0] > 0:
        node_features = data.x.cpu().numpy()
        # 检查特征是否有效（无NaN、无Inf、节点数足够）
        if np.any(np.isnan(node_features)) or np.any(np.isinf(node_features)):
            print("Warning: node_features contains NaN or Inf. Using default colors.")
        elif num_nodes < 2:
            print("Warning: Too few nodes ({}) for clustering. Using default colors.".format(num_nodes))
        else:
            # 动态设置簇数（不超过节点数）
            n_clusters = min(3, num_nodes)
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                kmeans.fit(node_features)  # 确保拟合
                node_labels = kmeans.labels_  # 获取聚类标签
                unique_labels = set(node_labels)
                color_map = plt.cm.get_cmap('Set1', max(len(unique_labels), 1))  # 避免0簇
                node_colors = [color_map(node_labels[i]) for i in range(num_nodes)]
                print(f"KMeans successful: {n_clusters} clusters, labels: {node_labels}")
            except Exception as e:
                print(f"KMeans failed: {e}. Using default colors.")
    else:
        print("No valid node features (data.x). Using default colors.")

    # 绘制图
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # 固定种子以确保布局可重现

    # 绘制原图（淡色）
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.3)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # 绘制子图（只高亮边）
    nx.draw_networkx_edges(G_sub, pos, edge_color='red', width=2.0)

    plt.title("GNN Explanation: Original Graph (gray) and Important Edges (red)")
    plt.show()


def info_nce_loss(embed_orig, pos_embeds, neg_embeds, tmp=1.0):
    """
    计算 InfoNCE 对比损失，显式利用正负样本。

    参数：
        embed_orig (torch.Tensor): 原图嵌入，形状 (D,)。
        pos_embeds (torch.Tensor): 正样本嵌入，形状 (P, D)。
        neg_embeds (torch.Tensor): 负样本嵌入，形状 (N, D)。
        tau (float): 温度参数。

    返回：
        torch.Tensor: InfoNCE 损失。
    """
    # 确保形状正确
    embed_orig = embed_orig.view(-1)
    pos_embeds = pos_embeds.view(-1, embed_orig.size(-1))
    neg_embeds = neg_embeds.view(-1, embed_orig.size(-1))

    # 归一化嵌入
    embed_orig_norm = embed_orig / (torch.norm(embed_orig, dim=-1, keepdim=True) + 1e-6)
    pos_embeds_norm = pos_embeds / (torch.norm(pos_embeds, dim=-1, keepdim=True) + 1e-6)
    neg_embeds_norm = neg_embeds / (torch.norm(neg_embeds, dim=-1, keepdim=True) + 1e-6)

    # 检查 NaN 或 Inf
    if torch.isnan(embed_orig_norm).any() or torch.isinf(embed_orig_norm).any():
        print("警告: embed_orig_norm 包含 NaN 或 Inf")
        return torch.tensor(0.0, device=embed_orig.device, requires_grad=True)

    # 计算相似性
    all_embeds = torch.cat([pos_embeds_norm, neg_embeds_norm], dim=0)  # (P+N, D)
    sim_matrix = torch.matmul(embed_orig_norm.unsqueeze(0), all_embeds.T).squeeze() / tmp  # (P+N,)

    # 正样本相似度
    pos_sim = sim_matrix[:pos_embeds.size(0)]
    exp_sim = torch.exp(sim_matrix)
    pos_sum = exp_sim[:pos_embeds.size(0)].sum()
    total_sum = exp_sim.sum()

    # InfoNCE 损失
    loss = -torch.log(pos_sum / (total_sum + 1e-6))

    if torch.isnan(loss) or torch.isinf(loss):
        # print("警告: 损失为 NaN 或 Inf，返回 0")
        return None

    return loss


def fn_softedgemask(data, edge_mask, isFidelitPlus=True):
    # 如果 data.edge_attr 为 None，创建默认边特征（全 1）
    # if data.edge_attr is None:
    #     num_edges = data.edge_index.size(1)
    #     edge_attr = torch.ones((num_edges, 1), device=edge_mask.device, dtype=edge_mask.dtype)
    # else:
    #     edge_attr = data.edge_attr

    # masked_x = data.x * node_mask.unsqueeze(1)  # 广播至节点维度[num,dim]

    num_edges = data.edge_index.size(1)
    edge_weight = torch.ones((num_edges, 1), device=edge_mask.device, dtype=edge_mask.dtype)
    # # 应用边掩码
    # masked_edge_attr = (edge_mask if not isFidelitPlus else 1 - edge_mask) * edge_attr

    # 应用边掩码 isFidelitPlus = True 执行else 为非重要结构
    # edge_mask: [num_edges], edge_attr: [num_edges, num_features]
    masked_edge_weight = edge_mask.view(-1, 1) * edge_weight if not isFidelitPlus else (1 - edge_mask.view(-1,
                                                                                                           1)) * edge_weight

    # 创建新的 Data 对象
    masked_data = data.clone()
    masked_data.edge_weight = masked_edge_weight

    # new_data = Data(
    #     x=data.x,
    #     edge_index=data.edge_index,
    #     edge_attr=(edge_mask if not isFidelitPlus else 1 - edge_mask) * data.edge_attr,
    #     batch=data.batch
    # )
    return masked_data


def topk_edge_mask(data, edge_mask, rate=0.15, isFidelitPlus=False) -> Data:
    num_edges = data.edge_index.size(1)

    # 1. 计算 hard_mask（0/1）
    k = max(1, int(rate * num_edges))
    threshold = edge_mask.topk(k).values.min()
    hard_mask = (edge_mask >= threshold).float()  # [E] → 0.0 or 1.0

    # 2. 关键：子图用 hard_mask，补图用 1 - hard_mask
    final_mask = hard_mask if not isFidelitPlus else (1 - hard_mask)

    # 3. 克隆 + 应用
    masked_data = data.clone()
    masked_data.edge_weight = final_mask  # [E]，自动广播
    return masked_data


def mask_fn_nodemask(data: Data, node_mask: torch.Tensor, isFidelitPlus=False) -> Data:
    """
    Subgraph building by selecting nodes from the original graph using node_mask.
    All operations are performed on GPU.

    Args:
        data: torch_geometric.data.Data object (on cuda:0)
        node_mask: torch.Tensor of shape [num_nodes], importance scores in [0, 1] (on cuda:0)
        isFidelitPlus: bool, if True, remove high-score nodes (Fidelity+); else, keep high-score nodes (Fidelity-)

    Returns:
        subgraph: Data object representing the subgraph (on cuda:0)
    """

    g = data.clone()
    # 确保输入在同一设备
    assert node_mask.device == data.x.device, "node_mask and data must be on the same device"

    # 动态设置移除/保留节点比例
    max_remove_ratio = 0.3
    min_keep_nodes = max(2, int(g.num_nodes * 0.5))
    k = min(max(1, int(g.num_nodes * max_remove_ratio)), g.num_nodes - min_keep_nodes)

    # 计算阈值（Top-K 高分节点的最低得分）
    if k > 0:
        threshold = torch.topk(node_mask, k, largest=True).values.min()
    else:
        threshold = node_mask.max() if isFidelitPlus else node_mask.min()

    # 生成布尔掩码
    node_mask_bool = node_mask < threshold if isFidelitPlus else node_mask >= threshold

    # # 选择子图的节点
    # node_idx = torch.where(node_mask_bool)[0]  # 选中的节点索引
    # if node_idx.size(0) == 0:
    #     # 如果子图为空，保留一个随机节点以避免空子图
    #     node_idx = torch.tensor([torch.randint(0, g.num_nodes, (1,))], device=g.x.device)
    #     print("Warning: 子图为空，保留一个随机节点")

    node_idx = torch.where(node_mask_bool)[0]  # 选中的节点索引
    if node_idx.size(0) == 0:
        # 如果子图为空，取 node_mask 最小的 5 个节点
        _, node_idx = torch.topk(node_mask, 5, largest=False)
        print("Warning: 子图为空，取 node_mask 最小的 5 个节点")

    # 提取子图的节点特征
    ret_x = g.x[node_idx]

    # 计算 edge_mask：边的两端节点都必须选中
    row, col = g.edge_index
    edge_mask = node_mask_bool[row] & node_mask_bool[col]

    # 过滤边
    ret_edge_index = g.edge_index[:, edge_mask]
    ret_edge_attr = None if g.edge_attr is None else g.edge_attr[edge_mask]
    # 提取子图的边标签
    ret_edge_label = getattr(g, 'edge_label', None)  # 安全获取 ret_edge_label
    # 老方法 不安全
    # ret_edge_label = None if g.edge_label is None else g.edge_label[edge_mask]

    # 重新编号边索引
    node_map = torch.full((g.num_nodes,), -1, dtype=torch.long, device=g.x.device)
    node_map[node_idx] = torch.arange(node_idx.size(0), device=g.x.device)
    ret_edge_index = node_map[ret_edge_index]

    # 处理 batch
    ret_batch = g.batch[node_idx] if hasattr(g, 'batch') and g.batch is not None else None

    # 复制原始图的标签（y）
    # ret_y = g.y.clone() if hasattr(g, 'y') and g.y is not None else None

    # 创建子图
    subgraph = Data(
        x=ret_x,
        edge_index=ret_edge_index,
        edge_attr=ret_edge_attr,
        edge_label=ret_edge_label,
        batch=ret_batch,
        # y=ret_y,
        # num_nodes=node_idx.size(0)
    )
    return subgraph


def mask_fn_edgemask(data: Data, hard_edge_mask: torch.Tensor, isFidelitPlus: bool) -> Data:
    """
    Subgraph building by selecting edges from the original graph using hard_edge_mask.
    All operations are performed on GPU.

    Args:
        data: torch_geometric.data.Data object (on cuda:0)
        hard_edge_mask: torch.Tensor of shape [num_edges], binary values (0 or 1) (on cuda:0)
        isFidelitPlus: bool, if True, remove edges with mask=1 (Fidelity+); else, keep edges with mask=1 (Fidelity-)

    Returns:
        subgraph: Data object representing the subgraph (on cuda:0)
    """
    g = data.clone()
    # print(hard_edge_mask)
    # 确保输入在同一设备
    assert hard_edge_mask.device == data.x.device, "hard_edge_mask and data must be on the same device"
    # 方法1：调整 hard_edge_mask 大小以匹配边数
    if hard_edge_mask.size(0) != g.num_edges:
        # 调整到正确的大小
        hard_edge_mask = hard_edge_mask[:g.num_edges] if hard_edge_mask.size(0) > g.num_edges else F.pad(
            hard_edge_mask, (0, g.num_edges - hard_edge_mask.size(0)))
    assert hard_edge_mask.size(0) == g.num_edges, "hard_edge_mask size must match number of edges"

    # 生成布尔边掩码
    if isFidelitPlus:
        edge_mask_bool = hard_edge_mask == 0  # F+: 保留 mask=0 的边，移除 mask=1 的边 （0为true）
    else:
        edge_mask_bool = hard_edge_mask == 1  # F-: 保留 mask=1 的边，移除 mask=0 的边 （1为true）

    # 选择子图的边
    ret_edge_index = g.edge_index[:, edge_mask_bool]  # 过滤边

    node_idx = torch.unique(ret_edge_index)  # 选中的边涉及的节点
    if ret_edge_index.size(1) == 0 or node_idx.size(0) == 0:
        # 如果没有边被选中，返回空子图
        # print("Warning: 无效空子图")
        return Data(
            x=torch.empty((0, g.x.size(1)), dtype=g.x.dtype, device=g.x.device),
            edge_index=torch.empty((2, 0), dtype=torch.long, device=g.x.device),
            edge_attr=None,
            edge_label=None,
            batch=None
        )

    # 提取子图的节点特征
    ret_x = g.x[node_idx]

    # 提取子图的边属性和边标签
    ret_edge_attr = None if g.edge_attr is None else g.edge_attr[edge_mask_bool]
    ret_edge_label = None if not hasattr(g, 'edge_label') else g.edge_label[edge_mask_bool]

    # 重新编号边索引
    node_map = torch.full((g.num_nodes,), -1, dtype=torch.long, device=g.x.device)
    node_map[node_idx] = torch.arange(node_idx.size(0), device=g.x.device)
    ret_edge_index = node_map[ret_edge_index]

    # 处理 batch（如果存在）
    ret_batch = g.batch[node_idx] if hasattr(g, 'batch') and g.batch is not None else None

    # 创建子图
    subgraph = Data(
        x=ret_x,
        edge_index=ret_edge_index,
        edge_attr=ret_edge_attr,
        edge_label=ret_edge_label,
        batch=ret_batch,
    )
    return subgraph

#
# def compute_fidelity_plus(classifier, exp_model, dataset, device: torch.device) -> float:
#     """
#     计算 Fidelity+ 分数：移除关键子结构后预测变化的平均 L2 范数（仅当原始预测正确时）。
# 
#     Args:
#         classifier: GNN 分类器模型
#         dataset: PyG Data 对象列表
#         explainer: 解释器函数，返回 (node_mask, edge_mask)
#         device: 计算设备
# 
#     Returns:
#         avg_fidelity_plus: 平均 Fidelity+ 分数
#     """
#     classifier.eval()
#     exp_model.eval()
#     classifier.to(device)
#     exp_model.to(device)
#     gcn = classifier.gnn
# 
#     total_fidelity = 0.0
#     num_graphs = 0  # 有效图数量（原始预测正确且子图非空）
# 
#     for graph in dataset:
#         graph = graph.to(device)
#         if graph.num_nodes == 0 or graph.edge_index.size(1) == 0:
#             continue
# 
#         with torch.no_grad():
#             # 1. 原始图预测：σ(f(G_i))
#             pred_orig = classifier(graph)
#             pred_orig_prob = F.softmax(pred_orig, dim=-1)
#             pred_orig_label = pred_orig.argmax(dim=-1)
#             # print("原始预测结果：", pred_orig_label)
# 
#             # 检查原始预测是否正确：I(ŷ_i == y_i)
#             if pred_orig_label != graph.y:
#                 num_graphs += 1
#                 continue  # 如果预测错误，贡献为 0
# 
#             # 2. 生成关键子结构掩码（假设 explainer 返回 edge_mask 用于 Fidelity+）
#             # _, node_embed = classifier.gnn(graph, emb=True)
#             _, node_embed = gcn(graph, isbatch=False)
#             node_mask, edge_mask = exp_model.explainer(data=graph, embed=node_embed, mode='explain')
# 
#             # k = 7
#             # top_k_indices = torch.topk(edge_mask, k=k, dim=0).indices  # 获取前k个最高值的索引
#             # hard_edge_mask = torch.zeros_like(edge_mask, dtype=torch.bool)  # 初始化布尔掩码
#             # hard_edge_mask[top_k_indices] = True  # 设置前k个边为True
#             # hard_edge_mask = hard_edge_mask.detach()  # 分离梯度
# 
#             masked_data = fn_softedgemask(graph, node_mask, edge_mask, isFidelitPlus=True)  # 非关键结构
# 
#             if masked_data.num_nodes == 0 or masked_data.edge_index.size(1) == 0:
#                 continue  # 跳过空子图
# 
#             # 4. 子图预测：σ(f(G_i \ S_i))
#             pred_masked = classifier(masked_data)
#             pred_masked_prob = F.softmax(pred_masked, dim=-1)
# 
#             # 5. 计算 L2 范数：||σ(f(G_i)) - σ(f(G_i \ S_i))||_2
#             fidelity_score = torch.norm(pred_orig_prob - pred_masked_prob, p=2).item()
#             total_fidelity += fidelity_score
#             num_graphs += 1
# 
#     avg_fidelity_plus = total_fidelity / max(num_graphs, 1)
#     # print(f"Average Fidelity+: {avg_fidelity_plus:.4f} (over {num_graphs} valid graphs)")
#     print(f"Average Fidelity+: {avg_fidelity_plus:.4f}")
#     return avg_fidelity_plus
# 
# 
# def compute_fidelity_minus(classifier, exp_model, dataset, device: torch.device) -> float:
#     """
#     计算 Fidelity- 分数：保留关键子结构后预测相似性的平均 L2 范数（低值表示好）。
# 
#     Args:
#         classifier: GNN 分类器模型
#         dataset: PyG Data 对象列表
#         explainer: 解释器函数，返回 (node_mask, edge_mask)
#         device: 计算设备
# 
#     Returns:
#         avg_fidelity_minus: 平均 Fidelity- 分数
#     """
#     classifier.eval()
#     exp_model.eval()
#     classifier.to(device)
#     exp_model.to(device)
#     gcn = classifier.gnn
# 
#     total_fidelity = 0.0
#     num_graphs = 0
# 
#     for graph in dataset:
#         graph = graph.to(device)
#         if graph.num_nodes == 0 or graph.edge_index.size(1) == 0:
#             continue
# 
#         with torch.no_grad():
#             # 1. 原始图预测：σ(f(G_i))
#             pred_orig = classifier(graph)
#             pred_orig_prob = F.softmax(pred_orig, dim=-1)
#             pred_orig_label = pred_orig.argmax(dim=-1)
# 
#             if pred_orig_label != graph.y:
#                 num_graphs += 1
#                 continue
# 
#             # 2. 生成关键子结构掩码
#             # _, node_embed = classifier.gnn(graph, emb=True)
#             _, node_embed = gcn(graph, isbatch=False)
#             node_mask, edge_mask = exp_model.explainer(data=graph, embed=node_embed, mode='explain')
# 
#             # 3. 构造 G_i \ S_i（移除高重要性边）
#             # threshold = edge_mask.mean()
#             # hard_edge_mask = (edge_mask > threshold).detach()
# 
#             # k = 7
#             # top_k_indices = torch.topk(edge_mask, k=k, dim=0).indices  # 获取前k个最高值的索引
#             # hard_edge_mask = torch.zeros_like(edge_mask, dtype=torch.bool)  # 初始化布尔掩码
#             # hard_edge_mask[top_k_indices] = True  # 设置前k个边为True
#             # hard_edge_mask = hard_edge_mask.detach()  # 分离梯度
# 
#             masked_data = fn_softedgemask(graph, node_mask, edge_mask, isFidelitPlus=False)  # 关键结构
# 
#             if masked_data.num_nodes == 0 or masked_data.edge_index.size(1) == 0:
#                 continue
# 
#             # 4. 子图预测：σ(f(S_i))
#             pred_masked = classifier(masked_data)
#             pred_masked_prob = F.softmax(pred_masked, dim=-1)
# 
#             # 5. 计算 L2 范数：||σ(f(G_i)) - σ(f(S_i))||_2（低值表示 S_i 足以代表 G_i）
#             fidelity_score = torch.norm(pred_orig_prob - pred_masked_prob, p=2).item()
#             total_fidelity += fidelity_score
#             num_graphs += 1
# 
#     avg_fidelity_minus = total_fidelity / max(num_graphs, 1)
#     print(f"Average Fidelity-: {avg_fidelity_minus:.4f} ")
#     return avg_fidelity_minus
