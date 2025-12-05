import random

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch import optim, nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from ogb.graphproppred import Evaluator
from torch_geometric.data import Data
from collections import Counter
# from upgnn.model import mask_fn_nodemask

criterion = nn.BCEWithLogitsLoss(reduction="none")
lr = 0.001
weight_decay = 0
epochs = 30


def compute_infonce_loss(embed, pos_embeds, neg_embeds, tau=0.2):
    """
    Compute InfoNCE loss with positive and negative samples from a sample pool.
    Args:
        embed: torch.Tensor, embedding of the current graph [1, emb_dim]
        pos_embeds: torch.Tensor, embeddings of positive samples [num_pos, emb_dim]
        neg_embeds: torch.Tensor, embeddings of negative samples [num_neg, emb_dim]
        tau: float, temperature parameter
    Returns:
        loss: torch.Tensor, InfoNCE loss
    """
    embed = F.normalize(embed, dim=-1)
    pos_embeds = F.normalize(pos_embeds, dim=-1)
    neg_embeds = F.normalize(neg_embeds, dim=-1)

    pos_sim = torch.exp(torch.matmul(embed, pos_embeds.t()) / tau)  # [1, num_pos]
    neg_sim = torch.exp(torch.matmul(embed, neg_embeds.t()) / tau)  # [1, num_neg]

    pos_sum = pos_sim.sum(dim=1)
    neg_sum = neg_sim.sum(dim=1)
    loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-3)).mean()
    return loss


def augment_graph(data):
    """
    Apply edge dropout to augment the graph and remove isolated nodes.
    Args:
        data: torch.geometric.data.Data, input graph
    Returns:
        augmented_data: torch.geometric.data.Data, augmented graph with isolated nodes removed
    """
    # 复制输入数据，避免修改原始数据
    augmented_data = Data(x=data.x.clone(), edge_index=data.edge_index.clone())

    # 1. 边 dropout
    edge_index = augmented_data.edge_index
    num_edges = edge_index.size(1)  # 强调的是维度 边索引一维是2 二维是边数量
    mask = torch.rand(num_edges, device=edge_index.device) > 0.8  # 保留 80% 的边
    augmented_data.edge_index = edge_index[:, mask]  # 删除无用的边

    # 2. 计算节点度，识别孤立节点  计算增广图中每个节点的度（degree），并通过节点度识别孤立节点（度为0的节点）
    num_nodes = data.x.size(0)
    degree = torch.zeros(num_nodes, dtype=torch.long,
                         device=edge_index.device)  # 创建一个全零张量 degree，形状为 (num_nodes,)，用于存储每个节点的度。
    if augmented_data.edge_index.size(1) > 0:  # 如果还有边
        # edge_index[0] 是边的源节点索引，edge_index[1] 是目标节点索引。
        degree = degree.scatter_add(0, augmented_data.edge_index[0],
                                    torch.ones(augmented_data.edge_index.size(1), dtype=torch.long,
                                               device=edge_index.device))
        degree = degree.scatter_add(0, augmented_data.edge_index[1],
                                    torch.ones(augmented_data.edge_index.size(1), dtype=torch.long,
                                               device=edge_index.device))

    # 3. 保留非孤立节点（度 > 0）
    # 孤立节点无拓扑结构信息，对于整个图的贡献较小，所以并不是必要。

    non_isolated_mask = degree > 0
    if not non_isolated_mask.any():  # 如果所有节点都孤立，返回空图
        return Data(x=torch.empty(0, data.x.size(1), device=data.x.device),
                    edge_index=torch.empty(2, 0, dtype=torch.long, device=edge_index.device),
                    batch=torch.empty(0, dtype=torch.long,
                                      device=edge_index.device) if augmented_data.batch is not None else None)

    # 4. 更新节点特征和 batch （摸去孤立节点后）
    augmented_data.x = augmented_data.x[non_isolated_mask]
    # if augmented_data.batch is not None:
    #     augmented_data.batch = augmented_data.batch[non_isolated_mask]

    # 5. 更新 edge_index，重新映射节点索引
    node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
    node_map[non_isolated_mask] = torch.arange(non_isolated_mask.sum(), device=edge_index.device)
    augmented_data.edge_index = node_map[augmented_data.edge_index]

    # 6. 移除无效边（如果有节点被移除）
    valid_edge_mask = (augmented_data.edge_index[0] >= 0) & (augmented_data.edge_index[1] >= 0)
    augmented_data.edge_index = augmented_data.edge_index[:, valid_edge_mask]

    return augmented_data



def check_class_distribution(loader):
    y_all = []
    for batch in loader:
        y = batch.y.squeeze().cpu().numpy()
        y_all.extend(y)
    print(Counter(y_all))


# MLP分类器
class MLP(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, hidden_dim, graph_class=2):
        super(MLP, self).__init__()
        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        if num_layer > 1:
            self.layers.append(nn.Linear(emb_dim, hidden_dim))
            for n in range(num_layer - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, graph_class))
        else:
            self.layers.append(nn.Linear(emb_dim, graph_class))

    def forward(self, emb):
        out = self.layers[0](emb)
        for layer in self.layers[1:]:
            out = layer(F.relu(out))
        # out 没有经过 softmax 函数处理,所以是每个类别的 logits 值，也就是模型对每个类别的未归一化的得分。
        return out


# GNN预测模型
class EndtoEnd(torch.nn.Module):
    # 它的作用是将嵌入模型（embed_model）和下游模型（mlp_model）封装成一个端到端的模型。
    # “端到端”意味着用户可以直接输入图数据（节点特征、边索引等），模型会自动生成嵌入并进行下游任务（例如分类或解释
    '''
    Class to wrap-up embedding model and downstream models into an end-to-end model.
    Args:
        embed_model, mlp_model: obj:`torch.nn.Module` objects.
        wrapped_input: Boolean. Whether (GNN) embedding model taks input wrapped in obj:`Data` object
        or node attributes and edge indices separately.
    wrapped_input：布尔值，控制输入格式：
    True：embed_model 期望输入是一个 Data 对象（PyTorch Geometric 的数据结构）。
    False：embed_model 期望输入是分离的节点特征（x）、边索引（edge_index）、边特征（edge_attr）和批次索引（batch）。
    '''

    def __init__(self, embed_model, mlp_model, wrapped_input=False):
        super(EndtoEnd, self).__init__()
        self.embed_model = embed_model
        self.mlp_model = mlp_model
        self.wrapped_input = wrapped_input

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        '''
        Forward propagation outputs the final prediction.
        '''
        if self.wrapped_input:
            if batch is None:
                batch = torch.zeros_like(x[:, 0], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            return self.forward_w(data)
        else:
            return self.forward_nw(x, edge_index, edge_attr, batch)

    def forward_w(self, data):  # 处理 Data 对象输入，生成最终预测。
        self.embed_model.eval()
        with torch.no_grad():
            emb = self.embed_model(data)
        out = self.mlp_model(emb)
        return out

    def forward_nw(self, x, edge_index, edge_attr, batch):  # 处理分离的输入（x、edge_index 等），生成最终预测。
        self.embed_model.eval()
        with torch.no_grad():
            emb = self.embed_model(x, edge_index, edge_attr, batch)
        out = self.mlp_model(emb)
        return out

    def get_emb(self, x, edge_index, edge_attr=None, batch=None):  # 只获得嵌入模型的emb 不会经过下游模型生成预测
        '''
        Forward propagation outputs only node embeddings.
        '''
        self.embed_model.eval()
        if self.wrapped_input:
            if batch is None:
                batch = torch.zeros_like(x[:, 0], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            emb = self.embed_model(data)
        else:
            emb = self.embed_model(x, edge_index, edge_attr, batch)
        return emb

    # 训练函数


def train_Encoder(embed_model, device, loader, val_loader, save_to=None):
    embed_model = embed_model.to(device)
    optimizer_encoder = optim.Adam(embed_model.parameters(), lr=lr, weight_decay=1e-5)

    val_losses = []
    best_val_loss = float('inf')
    for _ in range(epochs):
        print("--------Encoder_Train_epoch： " + str(_), end='\n')
        total_loss = 0
        train_batches = 0
        embed_model.train()
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            view1 = augment_graph(batch)
            view2 = augment_graph(batch)
            embed1 = embed_model(view1)
            embed2 = embed_model(view2)
            loss = compute_infonce_loss(embed1, embed2, tau=0.5)

            optimizer_encoder.zero_grad()
            loss.backward()
            optimizer_encoder.step()

            total_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_loss / train_batches
        print(f"Training Loss: {avg_train_loss:.4f}")
        # scheduler.step()

        embed_model.eval()
        total_val_loss = 0
        val_batches = 0
        print("--------GnnEncoder Evaluation--------")
        for step, batch in enumerate(tqdm(val_loader, desc="Validation")):
            batch = batch.to(device)
            with torch.no_grad():
                embeds = embed_model(batch)  # [batch_size, 2]
                # pred = mlp_model(embeds)  # [batch_size, 2]
                # 验证损失
                y = batch.y.squeeze()
                if y.dtype != torch.long:
                    y = y.long()

            total_val_loss += F.cross_entropy(embeds, y).item()
            val_batches += 1

        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # 保存最佳模型（基于验证集损失）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_to:
                torch.save(embed_model.state_dict(), save_to)
                print(f"Best GNN encoder saved to {save_to} with Val Loss: {best_val_loss:.4f}")


def train_MLP(embed_model, mlp_model, device, loader, val_loader, save_to=None):
    embed_model = embed_model.to(device)
    mlp_model = mlp_model.to(device)
    optimizer_mlpmodel = optim.Adam(mlp_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer_mlpmodel, step_size=30, gamma=0.1)
    best_roc = 0
    for _ in range(epochs):
        print("--------MLP_Train_epoch： " + str(_))
        embed_model.eval()
        mlp_model.train()
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            embeds = embed_model(batch)
            # print(embeds.shape)
            pred = mlp_model(embeds)

            y = batch.y.view(pred.shape).to(torch.float64)

            is_valid = y ** 2 > 0
            loss_mat = criterion(pred.double(), (y + 1) / 2)
            loss_mat = torch.where(is_valid, loss_mat,
                                   torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

            optimizer_mlpmodel.zero_grad()

            loss = torch.sum(loss_mat) / torch.sum(is_valid)
            loss.backward()
            optimizer_mlpmodel.step()
        # scheduler.step()

        mlp_model.eval()
        evaluator = Evaluator(name='ogbg-molhiv')  # OGB 评估器（ROC-AUC）
        y_true = []
        y_scores = []

        print("--------MLP Evaluation--------")

        for step, batch in enumerate(tqdm(val_loader, desc="Iteration")):
            batch = batch.to(device)
            with torch.no_grad():
                embeds = embed_model(batch).detach()
                pred = mlp_model(embeds)

            y_true.append(batch.y.view(pred.shape))
            y_scores.append(pred)

        y_true = torch.cat(y_true, dim=0)
        y_scores = torch.cat(y_scores, dim=0)
        eval_dict = evaluator.eval({
            'y_true': y_true,
            'y_pred': y_scores
        })
        roc_score = eval_dict['rocauc']
        print('auc roc_score：', roc_score)

        # 原方法计算人工数据集 roc_score
        # y_true = torch.cat(y_true, dim=0).cpu().numpy()
        # y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
        #
        # roc_list = []
        # for i in range(y_true.shape[1]):
        #     if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
        #         is_valid = y_true[:, i] ** 2 > 0
        #         roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        #
        # roc_score = sum(roc_list) / len(roc_list)

        # 使用 OGB Evaluator 计算 ROC-AUC
        '''
        AUC-ROC 的含义：AUC-ROC 是一个衡量分类模型性能的指标，表示模型在所有可能的分类阈值下区分正负类别的能力。它的取值范围是 [0, 1]：
                AUC-ROC 得分 越高越好，因为它反映了模型分类能力的强弱!!!!!
                        1 表示模型完美区分正负类别（理想情况）。
                        0.5 表示模型的分类能力相当于随机猜测（无区分能力）。
                        0 表示模型完全错误（将正类判为负类，反之亦然）。
        '''

        # 保存最佳模型（基于 ROC-AUC）
        if roc_score > best_roc:
            best_roc = roc_score
            if save_to:
                torch.save(mlp_model.state_dict(), save_to)
                print(f"MLPModel saved to {save_to} ")


def train_Encoder_Synthetic(embed_model, device, train_loader, val_loader, save_to=None):
    embed_model = embed_model.to(device)  # GCNencoder
    optimizer = optim.Adam(list(embed_model.parameters()), lr=1e-5, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    best_f1 = 0
    # class_weights = torch.tensor([5.0, 5.0, 1.0]).to(device)

    check_class_distribution(train_loader)
    check_class_distribution(val_loader)

    for epoch in range(200):
        print(f"--------Encoder_Train_epoch: {epoch}--------")
        total_loss = 0
        train_batches = 0
        embed_model.train()
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            batch = batch.to(device)
            embeds = embed_model(batch)  # [batch_size, 2]，移除 .detach()
            # print(f"embeds: {embeds}")
            '''
            embed 值可以是任意实数，包括正数、负数，且可大可小，但这些值的具体范围和特性取决于以下因素：
            GCN 的架构和参数（如权重矩阵、激活函数）。
            输入特征（节点特征 x 和边特征 edge_attr）。
            训练过程中的优化（如损失函数、优化器）。
            数值稳定性（受限于浮点运算和硬件）
            '''
            # 计算损失
            y = batch.y.squeeze()  # [batch_size]
            if y.dtype != torch.long:
                y = y.long()
            loss = F.cross_entropy(embeds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_loss / train_batches
        print(f"Training Loss: {avg_train_loss:.4f}")
        # scheduler.step()

        embed_model.eval()
        y_true = []
        y_pred = []
        val_loss = 0
        val_batches = 0
        print("--------GnnEncoder Evaluation--------")
        for batch in tqdm(val_loader, desc="Iteration"):
            batch = batch.to(device)
            with torch.no_grad():
                embeds = embed_model(batch)
                pred_proba = torch.softmax(embeds, dim=1)
                y = batch.y.squeeze().long()
                val_loss += F.cross_entropy(embeds, y).item()
                val_batches += 1
                y_true.append(y.cpu().numpy())
                y_pred.append(torch.argmax(pred_proba, dim=1).cpu().numpy())

        y_pred_all = np.concatenate(y_pred)
        print("Predicted class distribution:", Counter(y_pred_all))

        avg_val_loss = val_loss / val_batches
        print(f"Validation Loss: {avg_val_loss:.4f}")

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        print(f"Validation Macro-F1 Score: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            if save_to:
                torch.save(embed_model.state_dict(), save_to)
                print(f"GNNModel saved to {save_to} with Macro-F1: {best_f1:.4f}")
                embed_model.load_state_dict(torch.load(save_to, weights_only=False))


def train_MLP_Synthetic(embed_model, mlp_model, device, loader, val_loader, save_to=None):
    embed_model = embed_model.to(device)
    embed_model.eval()
    mlp_model = mlp_model.to(device)
    optimizer = optim.Adam(mlp_model.parameters(), lr=1e-5, weight_decay=1e-3)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    best_f1 = 0
    patience = 10
    epochs_no_improve = 0
    # class_weights = torch.tensor([5.0, 5.0, 1.0]).to(device)

    for epoch in range(50):
        print(f"--------MLP_Train_epoch: {epoch}--------")
        mlp_model.train()
        total_loss = 0
        train_batches = 0
        for batch in tqdm(loader, desc="Iteration"):
            batch = batch.to(device)
            embeds = embed_model(batch).detach()
            pred = mlp_model(embeds)
            y = batch.y.squeeze().long()
            loss = F.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_batches += 1
        avg_train_loss = total_loss / train_batches
        print(f"Training Loss: {avg_train_loss:.4f}")

        mlp_model.eval()
        y_true = []
        y_pred = []
        val_loss = 0
        val_batches = 0
        print("--------GnnEncoder Evaluation--------")
        for batch in tqdm(val_loader, desc="Iteration"):
            batch = batch.to(device)
            with torch.no_grad():
                embeds = embed_model(batch).detach()
                pred = mlp_model(embeds)
                pred_proba = torch.softmax(pred, dim=1)
                y = batch.y.squeeze().long()
                val_loss += F.cross_entropy(pred, y).item()
                val_batches += 1
                y_true.append(y.cpu().numpy())
                y_pred.append(torch.argmax(pred_proba, dim=1).cpu().numpy())

        avg_val_loss = val_loss / val_batches
        print(f"Validation Loss: {avg_val_loss:.4f}")

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        per_class_f1 = f1_score(y_true, y_pred, average=None)
        cm = confusion_matrix(y_true, y_pred)
        print(f"Validation Macro-F1 Score: {macro_f1:.4f}")
        print(f"Validation Weighted-F1 Score: {weighted_f1:.4f}")
        print(f"Per-class F1 Scores (0, 1, 2): {per_class_f1}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Predicted class distribution: {Counter(y_pred)}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            epochs_no_improve = 0
            if save_to:
                torch.save(mlp_model.state_dict(), save_to)
                print(f"MLPModel saved to {save_to} with Macro-F1: {best_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

        # scheduler.step(macro_f1)

    return best_f1
