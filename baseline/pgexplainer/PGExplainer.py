import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import *
from munch import Munch
from torch import Tensor
from torch.nn import ReLU
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.nn import Linear
from torch_geometric.utils import k_hop_subgraph
# from upgnn.model import fn_softedgemask
from base import InstanceExplainAlgorithm
from utils import set_masks, clear_masks, select_func
from torch.nn.parameter import UninitializedParameter


# from dig.xgraph.evaluation import XCollector
# from dig.xgraph.evaluation.metrics import fidelity_plus, fidelity_minus

class PGExplainer(InstanceExplainAlgorithm):
    r"""
    An implementation of PGExplainer in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>'.
    """
    coeffs = {
        'reg_size': 0.1,
        'reg_ent': 1e-3,
        'temp0': 5.0,
        'temp1': 1.0,
        'EPS': 1e-15,
        'edge_reduction': 'sum',
        'bias': 0.001,
        'knn_loss': 1.
    }

    def __init__(self, model: nn.Module, device, epochs: int = 25, gnn_task: str = 'graph', **kwargs):
        super().__init__()
        self.model = model  # Store the GNN model to explain
        self.device = device
        self.gnn_task = gnn_task
        self.coeffs.update(kwargs)
        self.epochs = epochs

        self.explainer = nn.Sequential(
            Linear(-1, 128),
            ReLU(),
            # 在 PGExplainer 的实现中，Linear(128, 1) 的输出维度 1 表示每条边被分配了一个 logits，对应于边的重要性评分。
            Linear(128, 1)
        ).to(device)

        self.temp_schedule = lambda e: (self.coeffs['temp0']
                                        * pow((self.coeffs['temp1'] / self.coeffs['temp0']), e / epochs))

    def __loss__(
            self,
            masked_preds: List[Tensor],
            original_labels: List[Tensor],
            masks: List[Tensor],
            edge_batch: Tensor,
            apply_sigmoid: bool
    ) -> Munch:
        loss_dict = Munch()
        for masked_pred, original_label, mask in zip(masked_preds, original_labels, masks):
            mask_ = mask.sigmoid() if apply_sigmoid else mask
            # Ensure original_label is 1D tensor
            if original_label.dim() > 1:
                original_label = original_label.squeeze(-1).long()
            elif original_label.dim() == 0:
                original_label = original_label.unsqueeze(0).long()
            ce_loss = F.cross_entropy(masked_pred, original_label, reduction='sum')

            size_loss = (scatter(mask_, edge_batch, dim=-1, reduce=self.coeffs['edge_reduction']).sum()
                         * self.coeffs['reg_size'])
            mask_ = mask_ * 0.99 + 0.005
            ent = - mask_ * torch.log(mask_) - (1 - mask_) * torch.log(1 - mask_)

            ent_loss = scatter(ent, edge_batch, dim=-1, reduce='mean').sum() * self.coeffs['reg_ent']

            loss_dict_ = Munch(
                cross_entropy=ce_loss,
                size_loss=size_loss,
                ent_loss=ent_loss
            )
            for u, v in loss_dict_.items():
                loss_dict.__setitem__(u, loss_dict.get(u, 0) + v)

        if self._explain_backward_hook:
            for hook in self._explain_backward_hook.values():
                hook(loss_dict, masks, masked_preds)

        return loss_dict

    def _create_inputs(
            self,
            embeddings: Tensor,
            edge_index: Tensor,
            index: Optional[Tensor] = None,
            edge_batch: Optional[Tensor] = None,
            hard_mask: Optional[Tensor] = None
    ):
        src, trg = edge_index
        src_embeds, trg_embeds = embeddings[src], embeddings[trg]

        if hard_mask is not None:
            src_embeds = src_embeds * hard_mask[..., None]
            trg_embeds = trg_embeds * hard_mask[..., None]

        if self.gnn_task == 'node':
            num_edges_per_graph = edge_batch.bincount()
            node_embed = embeddings[index].repeat_interleave(num_edges_per_graph, dim=0)
            inputs = torch.cat([src_embeds, trg_embeds, node_embed], dim=-1)
        else:
            inputs = torch.cat([src_embeds, trg_embeds], dim=-1)

        return inputs

    def _concrete_sample(self, logits, temperature=0.5, training=True):
        if training:
            bias = self.coeffs['bias']
            eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
            gate_inputs = torch.log(eps + self.coeffs['EPS']) - torch.log(1 - eps + self.coeffs['EPS'])
            gate_inputs = (gate_inputs.to(self.device) + logits) / temperature
            edge_mask = gate_inputs
        else:
            edge_mask = logits

        return edge_mask

    @property
    def pretrain(self):
        return True

    def load_parameters(self, path: str, dataset: str, ood: bool = False):
        # file_path = osp.join(path, f"pge+{'ood' if ood else 'dataset'}/{dataset}.pkl")
        file_path = osp.join(path, f"{dataset}/{dataset}.pkl")
        if not osp.exists(file_path):
            raise FileNotFoundError(f"Pretrained parameters not found at {file_path}")
        # PyTorch在加载模型权重时，由于weights_only=True参数限制，无法反序列化包含UninitializedParameter类型的对象，这是PyTorch 1.13+版本的安全机制。
        torch.serialization.add_safe_globals([UninitializedParameter])
        self.load_state_dict(torch.load(file_path, map_location=self.device, weights_only=True))

    def train_loop(
            self,
            data: Data,
            model_to_explain: nn.Module,
            epoch: int,
            use_edge_weight: bool = False,
            apply_sigmoid: bool = False,
            **kwargs
    ) -> Munch:
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        if (embeds := data.get('ori_embeddings')) is None:  # 为None 进入
            with torch.no_grad():
                _, embeds = model_to_explain.gnn(data, isbatch=False)
                # embeds = [model_to_explain.gnn(x, edge_index, emb=True)[1]]  # Use gnn method
        if (batch := data.get('batch')) is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
        edge_batch = batch[edge_index[0]]

        # epoch在这里仅做线性缩放
        temperature = self.temp_schedule(epoch)

        def get_explainer_pred(embed: Tensor, hard_mask: Tensor) -> Tensor:
            expl_inputs = self._create_inputs(
                embed.to(self.device),
                edge_index,
                data.get('corn_node_id'),
                edge_batch,
                hard_mask).unsqueeze(dim=0)
            logits = self.explainer(expl_inputs)[0]
            mask = self._concrete_sample(logits, temperature).squeeze()
            mask = F.sigmoid(mask)
            # edge_mask = torch.clamp(mask, min=0.01, max=0.99)

            return mask

        def get_gnn_pred(mask: Tensor) -> Tensor:
            set_masks(model_to_explain, mask, edge_index)
            pred = model_to_explain(data)
            clear_masks(model_to_explain)
            return pred

        if not isinstance(embeds, List):
            embeds = [embeds]

        hard_masks = data.get('hard_masks', [None] * len(embeds))
        # 已经被sigmoid了
        masks = [get_explainer_pred(embed, hard_mask) for embed, hard_mask in zip(embeds, hard_masks)]
        masked_pred = [get_gnn_pred(mask) for mask in masks]

        if self.gnn_task == 'node':
            corn_node_id = data.get('corn_node_id')
            assert len(masked_pred) == len(corn_node_id)
            masked_pred = [pred[node_id] for pred, node_id in zip(masked_pred, corn_node_id)]

        target_label = data.y if hasattr(data, 'y') else data.get('target_label')
        if not isinstance(target_label, List):
            target_label = [target_label]
        loss_dict = self.__loss__(masked_pred, target_label, masks, edge_batch, apply_sigmoid=False)  # 训练阶段

        return loss_dict

    # def topk_edge_mask_subgraph(self, data, edge_mask, rate=0.7, isFidelitPlus=False) -> Data:
    #     num_edges = data.edge_index.size(1)
    #
    #     # 1. 计算 hard_mask（0/1）
    #     k = max(1, int(rate * num_edges))
    #     threshold = edge_mask.topk(k).values.min()  # 取前topk比例的阈值
    #     hard_mask = (edge_mask >= threshold).float()  # [E] → 0.0 or 1.0
    #
    #     # 2. 关键：子图用 hard_mask 关键，补图用 1 - hard_mask 非关键
    #     final_mask = hard_mask if not isFidelitPlus else (1 - hard_mask)
    #
    #     # 3. 克隆 + 应用
    #     masked_data = data.clone()
    #     masked_data.edge_weight = final_mask  # [E]，自动广播
    #     return masked_data

    def hard_subgraph_mask(self, data, edge_mask, rate=0.5, keep_important=True):
        """
        标准硬删除方式（2025年顶会通用）
        keep_important=True  → 保留重要边（用于 Fidelity⁻）
        keep_important=False → 保留非重要边（用于 Fidelity⁺）
        """
        num_edges = data.edge_index.size(1)
        k = max(1, int(rate * num_edges))
        threshold = edge_mask.topk(k).values.min()
        hard_edge_mask = (edge_mask >= threshold)  # [E] bool

        if not keep_important:
            hard_edge_mask = ~hard_edge_mask  # Fidelity⁺ 用补集

        # 1. 选出要保留的边
        selected_edges = hard_edge_mask
        edge_index_new = data.edge_index[:, selected_edges]

        # 2. 找出这些边连接的节点（自动去重）
        src = edge_index_new[0]
        dst = edge_index_new[1]
        nodes_to_keep = torch.unique(torch.cat([src, dst]))

        if nodes_to_keep.numel() == 0:
            # 返回一个空图（防止 crash）
            empty_data = data.clone()
            empty_data.edge_index = torch.empty((2, 0), dtype=torch.long, device=data.x.device)
            empty_data.x = torch.empty((0, data.x.size(1)), device=data.x.device)
            return empty_data

        # 3. 构建新图：只保留这些节点和边
        new_data = data.clone()

        # 重映射节点索引（可选，但推荐，防止索引错乱）
        node_map = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        node_map[nodes_to_keep] = torch.arange(nodes_to_keep.size(0), device=data.x.device)

        new_data.x = data.x[nodes_to_keep]
        new_data.edge_index = node_map[edge_index_new]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            new_data.edge_attr = data.edge_attr[selected_edges]
        if hasattr(data, 'y'):
            new_data.y = data.y
        if hasattr(data, 'batch') and data.batch is not None:
            new_data.batch = data.batch[nodes_to_keep]
        else:
            new_data.batch = torch.zeros(nodes_to_keep.size(0), dtype=torch.long, device=data.x.device)

        new_data.num_nodes = nodes_to_keep.size(0)
        return new_data

    @property
    def name(self):
        return 'pgexplainer'

    # def forward(
    #         self,
    #         data: Data,
    #         **kwargs
    # ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    #     r"""
    #     Generating explanation for given graph.
    #     Args:
    #         data: a PYG Data which storage information of graph to be explained.
    #     Returns:
    #         Union[Tensor, Tensor]: edge_mask, feat_mask
    #     """
    #     assert (embeds := data.get('ori_embeddings')) is not None
    #
    #     edge_index = data.edge_index
    #     if (corn_node_id := data.get('corn_node_id')) is not None and kwargs.get("sample_subgraph"):
    #         edge_index = k_hop_subgraph(
    #             corn_node_id,
    #             kwargs.get('num_hops', 3),
    #             data.edge_index,
    #             relabel_nodes=False
    #         )[1]
    #
    #     if (batch := data.get('batch')) is None:
    #         batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)  # 修复：添加dtype和device
    #     edge_batch = batch[edge_index[0]]
    #
    #     if not isinstance(embeds, List): embeds = [embeds]
    #     hard_masks = data.get('hard_masks', [None] * len(embeds))
    #     masks = [self.get_explainer_pred(embed, edge_index, corn_node_id, edge_batch, hard_mask)
    #              for embed, hard_mask in zip(embeds, hard_masks)][0]
    #
    #     # edge_mask = masks[0]
    #     # edge_mask = F.sigmoid(masks)
    #
    #     return masks
    def forward(self, data: Data, **kwargs) -> Tensor:
        with torch.no_grad():  # 添加此行，整个推理无梯度
            assert (embeds := data.get('ori_embeddings')) is not None
        edge_index = data.edge_index
        if (corn_node_id := data.get('corn_node_id')) is not None and kwargs.get("sample_subgraph"):
            edge_index = k_hop_subgraph(corn_node_id, kwargs.get('num_hops', 3),
                                        data.edge_index, relabel_nodes=False)[1]

        batch = data.get('batch')
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
        edge_batch = batch[edge_index[0]]

        embed = embeds[0] if isinstance(embeds, list) else embeds
        hard_mask = None
        if isinstance(data.get('hard_masks'), list):
            hard_mask = data.hard_masks[0] if data.hard_masks else None

        inputs = self._create_inputs(embed, edge_index,
                                     data.get('corn_node_id'), edge_batch, hard_mask).unsqueeze(0)
        logits = self.explainer(inputs)[0].squeeze(-1)
        mask = self._concrete_sample(logits, training=False)

        return torch.sigmoid(mask)  # 直接返回 [0,1] 的 edge_mask

    def get_explainer_pred(self, embed, edge_index, corn_node_id, edge_batch, hard_mask=None):
        inputs = self._create_inputs(embed, edge_index, corn_node_id, edge_batch, hard_mask).unsqueeze(0)
        logits = self.explainer(inputs)[0].squeeze(-1)
        mask = self._concrete_sample(logits, training=False)
        return torch.sigmoid(mask)


def compute_fidelity_plus(classifier, explainer, dataset, device: torch.device) -> float:
    """
    计算 Fidelity+ 分数：移除关键子结构后预测变化的平均 L2 范数（仅当原始预测正确时）。

    Args:
        classifier: GNN 分类器模型
        dataset: PyG Data 对象列表
        explainer: 解释器函数，返回 (node_mask, edge_mask)
        device: 计算设备

    Returns:
        avg_fidelity_plus: 平均 Fidelity+ 分数
    """
    classifier.eval()
    explainer.eval()

    total_fidelity = 0.0
    num_graphs = 0  # 有效图数量（原始预测正确且子图非空）

    for graph in dataset:
        graph = graph.to(device)
        if graph.num_nodes == 0 or graph.edge_index.size(1) == 0:
            continue

        with torch.no_grad():
            # 1. 原始图预测：σ(f(G_i))
            pred_orig = classifier(graph)
            pred_orig_prob = F.softmax(pred_orig, dim=-1)
            pred_orig_label = pred_orig.argmax(dim=-1)
            # print("原始预测结果：", pred_orig_label)

            # 检查原始预测是否正确：I(ŷ_i == y_i)
            if pred_orig_label != graph.y:
                # num_graphs += 1
                continue  # 如果预测错误，贡献为 0

            _, node_embed = classifier.gnn(graph, isbatch=False)
            graph.ori_embeddings = node_embed

            # Generate explanation
            edge_mask = explainer(graph)
            # edge_mask = torch.clamp(edge_mask, min=0.01, max=0.99)
            # masked_data = fn_softedgemask(graph, edge_mask, isFidelitPlus=True)  # 需实现：用掩码过滤节点/边
            # masked_data = explainer.topk_edge_mask_subgraph(graph, edge_mask, rate=0.5,isFidelitPlus=True)  # 需实现：用掩码过滤节点/边
            masked_data = explainer.hard_subgraph_mask(graph, edge_mask, rate=0.7, keep_important=False)  # 需实现：用掩码过滤节点/边

            if masked_data.num_nodes == 0 or masked_data.edge_index.size(1) == 0:
                continue  # 跳过空子图

            # 4. 子图预测：σ(f(G_i \ S_i))
            pred_masked = classifier(masked_data)
            pred_masked_prob = F.softmax(pred_masked, dim=-1)

            # 5. 计算 L2 范数：||σ(f(G_i)) - σ(f(G_i \ S_i))||_2
            fidelity_score = torch.norm(pred_orig_prob - pred_masked_prob, p=2).item()
            total_fidelity += fidelity_score
            num_graphs += 1

    avg_fidelity_plus = total_fidelity / max(num_graphs, 1)
    # print(f"Average Fidelity+: {avg_fidelity_plus:.4f} (over {num_graphs} valid graphs)")

    return avg_fidelity_plus


def compute_fidelity_minus(classifier, explainer, dataset, device: torch.device) -> float:
    """
    计算 Fidelity- 分数：保留关键子结构后预测相似性的平均 L2 范数（低值表示好）。

    Args:
        classifier: GNN 分类器模型
        dataset: PyG Data 对象列表
        explainer: 解释器函数，返回 (node_mask, edge_mask)
        device: 计算设备

    Returns:
        avg_fidelity_minus: 平均 Fidelity- 分数
    """
    classifier.eval()
    explainer.eval()

    total_fidelity = 0.0
    num_graphs = 0

    for graph in dataset:
        graph = graph.to(device)
        if graph.num_nodes == 0 or graph.edge_index.size(1) == 0:
            continue

        with torch.no_grad():
            # 1. 原始图预测：σ(f(G_i))
            pred_orig = classifier(graph)
            pred_orig_prob = F.softmax(pred_orig, dim=-1)
            pred_orig_label = pred_orig.argmax(dim=-1)

            if pred_orig_label != graph.y:
                # num_graphs += 1
                continue

            _, node_embed = classifier.gnn(graph, isbatch=False)
            graph.ori_embeddings = node_embed

            # Generate explanation
            edge_mask = explainer(graph)
            # edge_mask = torch.clamp(edge_mask, min=0.01, max=0.99)
            # masked_data = fn_softedgemask(graph, edge_mask, isFidelitPlus=False)  # 需实现：用掩码过滤节点/边
            # masked_data = explainer.topk_edge_mask_subgraph(graph, edge_mask, rate=0.5,isFidelitPlus=False)  # 需实现：用掩码过滤节点/边
            masked_data = explainer.hard_subgraph_mask(graph, edge_mask, rate=0.7, keep_important=True)  # 需实现：用掩码过滤节点/边

            if masked_data.num_nodes == 0 or masked_data.edge_index.size(1) == 0:
                continue

            # 4. 子图预测：σ(f(S_i))
            pred_masked = classifier(masked_data)
            pred_masked_prob = F.softmax(pred_masked, dim=-1)

            # 5. 计算 L2 范数：||σ(f(G_i)) - σ(f(S_i))||_2（低值表示 S_i 足以代表 G_i）
            fidelity_score = torch.norm(pred_orig_prob - pred_masked_prob, p=2).item()
            total_fidelity += fidelity_score
            num_graphs += 1

    avg_fidelity_minus = total_fidelity / max(num_graphs, 1)
    # print(f"Average Fidelity-: {avg_fidelity_minus:.4f} ")
    return avg_fidelity_minus


def evaluate_single_graph(classifier, explainer, data, device):
    classifier.eval()
    explainer.eval()
    data = data.to(device)
    with torch.no_grad():
        _, node_embed = classifier.gnn(data, isbatch=False)
        data.ori_embeddings = node_embed
        edge_mask = explainer(data)
        # edge_mask = torch.clamp(edge_mask, min=0.01, max=0.99)
        # masked_data_minus = fn_softedgemask(data, edge_mask, isFidelitPlus=False)
        # masked_data = explainer.topk_edge_mask_subgraph(data, edge_mask, rate=0.5,isFidelitPlus=False)  # 需实现：用掩码过滤节点/边
        masked_data = explainer.hard_subgraph_mask(data, edge_mask, rate=0.7,keep_important=True)  # 需实现：用掩码过滤节点/边  # 需实现：用掩码过滤节点/边

        logists = classifier(masked_data)
        pred_prob = torch.sigmoid(logists).squeeze()  # 转换为概率，形状 [2]
        true_label = data.y.item()
        predicted_label = torch.argmax(pred_prob).item()

        return true_label, predicted_label


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'ogb'
    classifier, train_dataset, valid_dataset, test_dataset = select_func(dataset, device=device)
    # dataset_root = '../../upgnn/data/ba2motif'
    #
    # train_dataset = Mutag(root=dataset_root, split='train')
    # valid_dataset = Mutag(root=dataset_root, split='val')
    # test_dataset = Mutag(root=dataset_root, split='test')
    #
    # train_dataset = map_labels(train_dataset)
    # valid_dataset = map_labels(valid_dataset)
    # test_dataset = map_labels(test_dataset)
    print("data:", train_dataset[0])
    node_in_dim = train_dataset[0].x.shape[1]
    print("node_in_dim:", node_in_dim)
    all_labels = [data.y.item() for data in train_dataset]
    num_classes = len(set(all_labels))
    print(f"Number of classes: {num_classes}")

    explainer = PGExplainer(classifier, device, epochs=10, gnn_task='graph')
    # optimizer = torch.optim.Adam(explainer.explainer.parameters(), lr=0.001)
    # explainer.train()
    # classifier.eval()
    # for epoch in range(explainer.epochs):
    #     total_loss = 0.0
    #     for data in train_dataset:
    #         data = data.to(device)
    #         # if data.y.dim() > 1:
    #         #     data.y = data.y.squeeze(-1).long()
    #         # elif data.y.dim() == 0:
    #         #     data.y = data.y.unsqueeze(0).long()
    #
    #         # with torch.no_grad():
    #         _, node_embed = classifier.gnn(data, isbatch=False)
    #         data.ori_embeddings = node_embed.to(device)
    #
    #         loss_dict = explainer.train_loop(data, classifier, epoch=epoch, use_edge_weight=False)
    #         loss = sum(loss_dict.values())
    #         total_loss += loss.item()
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_dataset):.8f}")
    #
    # os.makedirs(f'./pge_pretrained/{dataset}', exist_ok=True)
    # torch.save(explainer.state_dict(), f'./pge_pretrained/{dataset}/{dataset}.pkl')
    explainer.load_parameters(path='./pge_pretrained', dataset=dataset)
    explainer.to(device)

    print("Testing graph classification explanation...")
    true_labels = []
    predicted_labels = []
    skipped_graphs = 0

    # for data in valid_dataset:
    #     data = data.to(device)
    #     # if data.y.dim() > 1:  # 张量
    #     #     data.y = data.y.squeeze(-1).long()
    #     # elif data.y.dim() == 0:  # 标量
    #     #     data.y = data.y.unsqueeze(0).long()
    #
    #     with torch.no_grad():
    #         _, node_embed = classifier.gnn(data, isbatch=False)
    #     data.ori_embeddings = node_embed
    #
    #     # Generate explanation
    #     edge_mask = explainer(data)
    #     # edge_mask = torch.clamp(edge_mask, min=0.01, max=0.99)
    #     # Generate subgraph
    #     # masked_data = fn_softedgemask(data, edge_mask, isFidelitPlus=False)  # 需实现：用掩码过滤节点/边
    #     # masked_data = explainer.topk_edge_mask_subgraph(data, edge_mask, rate=0.5,isFidelitPlus=False)  # 需实现：用掩码过滤节点/边
    #     masked_data = explainer.hard_subgraph_mask(data, edge_mask, rate=0.7,keep_important=True)  # 需实现：用掩码过滤节点/边  # 需实现：用掩码过滤节点/边
    #
    #     if masked_data is None or masked_data.num_nodes == 0 or masked_data.edge_index.size(1) == 0:
    #         print(f"Warning: Empty subgraph for graph {data}, skipping...")
    #         skipped_graphs += 1
    #         continue
    #
    #     # logits = classifier(masked_data)
    #     # pred_prob = F.softmax(logits, dim=-1)  # 概率和为1
    #     # # pred_prob = torch.sigmoid(logits).squeeze() #  二分类概率各自属于[0,1]区间，但是各个分类概率互不影响
    #
    #     true_label, predicted_label = evaluate_single_graph(classifier, explainer, data, device)
    #     true_labels.append(true_label)
    #     predicted_labels.append(predicted_label)
    #
    # accuracy = accuracy_score(true_labels, predicted_labels)
    # conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))
    # print(f"Accuracy: {accuracy:.4f}")
    # print("Confusion Matrix:")
    # print(conf_matrix)

    with torch.no_grad():
        print("Evaluating fidelity...")
        avg_fidelity_plus = compute_fidelity_plus(classifier, explainer, test_dataset, device)
        print(f"Average Fidelity+: {avg_fidelity_plus:.4f}")
        avg_fidelity_minus = compute_fidelity_minus(classifier, explainer, test_dataset, device)
        print(f"Average Fidelity-: {avg_fidelity_minus:.4f}")

    print("--------------done!-----------------")
