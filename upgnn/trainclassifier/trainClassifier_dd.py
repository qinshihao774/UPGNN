import os
import random
import numpy as np
import torch

from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

from torch_geometric.utils import add_self_loops, softmax, add_remaining_self_loops
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from upgnn.dataset.dd import DD
from upgnn.downstream_model import MLP
from sklearn.metrics import f1_score, roc_auc_score

from sklearn.metrics import accuracy_score, confusion_matrix

patience = 8

num_atom_type = 120  # mutag 8
# num_atom_type = 120  # including the extra mask tokens （原子类型的总数）。118原子类型的总数+额外掩码原子类型。
num_chirality_tag = 3  # 手性标签（chirality tags）的总数。 手性标签的总数（例如，3 种：无手性、R 构型、S 构型）。
num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens 4
num_bond_direction = 3


def set_seed(seed):
    random.seed(seed)  # Python随机数种子
    np.random.seed(seed)  # NumPy随机数种子
    torch.manual_seed(seed)  # PyTorch CPU随机数种子
    torch.cuda.manual_seed(seed)  # PyTorch GPU随机数种子
    torch.cuda.manual_seed_all(seed)  # 多GPU时设置所有GPU的种子
    torch.backends.cudnn.deterministic = True  # 确保CuDNN的确定性
    torch.backends.cudnn.benchmark = False  # 禁用CuDNN的自动优化


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GCNConv(MessagePassing):  # 通过线性变换和消息传递更新节点特征表示，同时利用边嵌入融入边信息。
    def __init__(self, emb_dim, hidden_dim=32, out_dim=32, aggr="add"):

        super(GCNConv, self).__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.aggr = aggr

        self.bias = True
        self.normalize = True
        self.improved = False
        self.add_self_loops = True

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, hidden_dim), torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_dim, out_dim))

        # self.edge_embedding1 = torch.nn.Embedding(num_bond_type, out_dim)  # 边嵌入
        # self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        # 使用 Xavier 均匀初始化（Xavier Uniform Initialization）初始化两个嵌入层的权重。
        # torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)  # 边嵌入初始化
        # torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        if self.bias:
            self.bias = Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def norm(self, edge_index, edge_weight, num_nodes,
             improved=False, add_self_loops=True, flow="source_to_target", dtype=None):
        ### assuming that self-loops have been already added in edge_index

        fill_value = 2. if improved else 1.

        # assert flow in ['source_to_target', 'target_to_source']
        # num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if add_self_loops:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)

        if edge_weight is None:  # 所有边都是一样的重要
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        row, col = edge_index
        # 有向
        # idx = col if flow == 'source_to_target' else row
        # deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='mean')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    # TODO:MUTAG
    # def forward(self, x, edge_index, edge_attr=None, edge_label=None):
    #     device = x.device
    #     edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    #     # num_edge_features = edge_attr.size(1) if edge_attr is not None else 1
    #     num_edge_features = num_bond_type
    #
    #     if edge_label is not None:
    #         # 确保 edge_label 是 int64 类型
    #         edge_label = edge_label.to(dtype=torch.int64, device=device)
    #         edge_attr = torch.zeros((edge_label.size(0), num_edge_features), device=device, dtype=x.dtype)
    #         edge_attr.scatter_(1, edge_label.unsqueeze(1), 1)
    #
    #     # num_self_loops = x.size(0)  # 节点数目
    #     self_loop_attr = torch.zeros(x.size(0), num_edge_features, device=device, dtype=x.dtype)
    #     edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0) if edge_attr is not None else self_loop_attr
    #
    #     edge_types = torch.argmax(edge_attr, dim=1)  # 形状: (num_edges + num_nodes,)
    #     edge_embeddings = self.edge_embedding1(edge_types)
    #
    #     norm = self.norm(edge_index, x.size(0), x.dtype)
    #     x = self.linear(x)  # 线性变换
    #     return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    # 调用时机：self.propagate 在处理每条边时，会调用 message 函数来生成消息。
    # 具体来说，message 函数在 propagate 内部被调用，针对每条边 (i, j)（i 是目标节点，j 是源节点）生成消息。

    def forward(self, x, edge_index, edge_weight=None):

        num_nodes = x.size(0)

        if self.normalize:
            edge_index, edge_weight = self.norm(edge_index, edge_weight, num_nodes, dtype=x.dtype)
        else:
            edge_index, edge_weight = add_remaining_self_loops(edge_index, None, 1., num_nodes)

        x = self.linear(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    # def message_and_aggregate(self, adj_t, x):
    #     return spmm(adj_t, x, reduce=self.aggr)

    def update(self, aggr_out):
        return aggr_out


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, hidden_dim=32, out_dim=32, JK="last", drop_ratio=0.0, gnn_type="gcn"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of MLPs 管理神经网络层数l
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            input_dim = emb_dim if layer == 0 else self.out_dim  # 动态设置输入维度

            if gnn_type == "gin":
                self.gnns.append(GINConv(input_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(input_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(input_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(input_dim))

        # List of batch norms 对批量进行归一化
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))

    def forward(self, data, isbatch=False):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_attr = getattr(data, 'edge_attr', None)
        # 判断是否有边权重
        edge_weight = getattr(data, 'edge_weight', None)

        # device = x.device
        # edge_label = getattr(data, 'edge_label', None)  # 安全获取 edge_label

        # print(data.x)  # 检查 one-hot 编码
        # data.atom_types = torch.argmax(data.x, dim=1) + 1  # 获取原子类型索引 从1开始
        # print("原子索引范围：", data.atom_types)  # 检查索引范围

        # weight 是 nn.Embedding 模块的可学习参数，表示嵌入矩阵。
        # self.x_embedding1.weight.data = torch.cat(
        #     [torch.zeros(1, self.emb_dim).to(device), self.x_embedding1.weight.data[1:]], 0)

        # x = self.x_embedding1(data.atom_types)  # 这里传入的就是X特征矩阵 节点表示 形状: (num_nodes, emb_dim)

        h_list = [x]
        for layer in range(self.num_layer):  # 3层GCN
            # h = self.gnns[layer](h_list[layer], edge_index, edge_attr, edge_label)
            h = self.gnns[layer](h_list[layer], edge_index, edge_weight)  # 这里的edge_attr就是权重
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio)
            h_list.append(h)

        # Different implementations of Jk-concat 解决过平滑问题
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":  # 最简单的 JK 方式，等价于不使用 JK 机制。
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        if isbatch:
            # if emb:
            return global_mean_pool(node_representation, batch), node_representation

        return global_mean_pool(node_representation, None), node_representation


class GNNClassifier(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, hidden_dim, num_tasks=2, JK="last", drop_ratio=0.0,
                 graph_pooling="mean", gnn_type="gcn"):
        super(GNNClassifier, self).__init__()
        # gcn_3l
        self.gnn = GNN(num_layer=num_layer, emb_dim=emb_dim, hidden_dim=hidden_dim, JK=JK, drop_ratio=drop_ratio,
                       gnn_type=gnn_type)
        # mlp_3_64hidden => num_tasks 分类 因为输入的是节点表示，所以是32
        self.mlp = MLP(num_layer=num_layer, emb_dim=32, hidden_dim=hidden_dim, graph_class=num_tasks)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            gate_nn = torch.nn.Linear(emb_dim if JK != "concat" else (num_layer + 1) * emb_dim, 1)
            self.pool = GlobalAttention(gate_nn=gate_nn)
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1]) if graph_pooling[-1].isdigit() else 2
            self.pool = Set2Set(emb_dim if JK != "concat" else (num_layer + 1) * emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.mult = 2 if graph_pooling[:-1] == "set2set" else 1

    def forward(self, data, isbatch=False):
        # x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if isbatch:
            batch = data.batch
            # edge_label = getattr(data, 'edge_label', None)  # 安全获取 edge_label

            # x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            embed, node_embed = self.gnn(data, isbatch=isbatch)
            graph_representation = self.pool(node_embed, batch)
        else:
            embed, node_embed = self.gnn(data, isbatch=isbatch)
            graph_representation = self.pool(node_embed, None)

        return self.mlp(graph_representation)  # 返回没有归一化的输出

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, weights_only=True))


def train_gnn_classifier(model, train_dataset, val_dataset, device, epochs=350, lr=0.01):
    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    # Adam 不仅能够计算每个参数的自适应学习率，还会利用梯度的一阶矩估计（均值）和二阶矩估计（方差）来动态调整学习率。
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # SGD 作为最基础的优化算法，其参数更新依据的是当前 batch 数据计算出的梯度，公式为：θ=θ−μ⋅∇L(θ)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_roc_auc': [], 'val_roc_auc': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_preds, train_labels, train_probs = 0.0, [], [], []

        for i, batch_data in enumerate(train_loader):
            batch = batch_data.to(device)
            batch.y = torch.where(batch.y == 2.0, torch.tensor(0.0), batch.y)
            batch.y = batch.y.squeeze().float()  # 去除一个维度

            # 调试 batch 信息
            assert batch.batch.max() < batch.num_graphs, f"Batch index {batch.batch.max()} exceeds num_graphs {batch.num_graphs}"
            # print(f"Batch: x {batch.x.shape}, batch max {batch.batch.max()}, num_graphs {batch.num_graphs}")

            optimizer.zero_grad()
            out = model(batch, isbatch=True)  # logits（即未经过 softmax 处理的得分） = out
            # print(f"out shape: {out.shape}")
            # print(f"batch.y shape: {batch.y.shape}")

            # pred = torch.argmax(out, dim=1)
            # pred = pred.float()
            # print("pred:", pred)
            # print("y:", batch.y)
            # torch.nn.CrossEntropyLoss 期望的输入 input 是 logits（即未经过 softmax 处理的得分）
            loss = criterion(out, batch.y.squeeze().long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
            train_preds.extend(out.argmax(dim=1).cpu().numpy())
            train_labels.extend(batch.y.cpu().numpy())
            train_probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().detach().numpy())

        train_loss /= len(train_dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_roc_auc = roc_auc_score(train_labels, train_probs)

        model.eval()
        val_loss, val_preds, val_labels, val_probs = 0.0, [], [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch.y = torch.where(batch.y == 2.0, torch.tensor(0.0), batch.y)
                batch.y = batch.y.squeeze().float()  # 去除一个维度
                assert batch.batch.max() < batch.num_graphs, f"Batch index {batch.batch.max()} exceeds num_graphs {batch.num_graphs}"
                out = model(batch, isbatch=True)
                # pred = torch.argmax(out, dim=1)
                loss = criterion(out, batch.y.squeeze().long())

                val_loss += loss.item() * batch.num_graphs
                val_preds.extend(out.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())
                val_probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().detach().numpy())

        val_loss /= len(val_dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_roc_auc = roc_auc_score(val_labels, val_probs)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_roc_auc'].append(train_roc_auc)  # 存储 ROC-AUC
        history['val_roc_auc'].append(val_roc_auc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, ROC-AUC: {train_roc_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}')
        # 当一组批量所计算的误差值低于之前最小的误差值时，则更新模型参数，并重置 patience 计数器。
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # model.load_state_dict(best_model_state)

    return best_model_state, history


def evaluate_single_graph(classifier, graph, device):
    classifier.eval()
    classifier.to(device)
    graph = graph.to(device)
    with torch.no_grad():
        logists = classifier(graph)
        pred_prob = torch.softmax(logists, dim=1).squeeze()  # 转换为概率，形状 [2]
        # print("Predicted Probabilities:", pred_prob)
        true_label = graph.y.item()
        predicted_label = torch.argmax(pred_prob, dim=0).item()
        predicted_label = 1.0 if predicted_label == 1.0 else 2.0
    return true_label, predicted_label


def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 示例数据集（需要替换为实际数据集）
    # # TODO dd: Data(x=[242, 82], edge_index=[2, 1106], y=[1] 1 2, node_label=[242])
    data_name = "dd"
    train_dataset = DD('train')
    valid_dataset = DD('valid')
    test_dataset = DD('test')

    print("single data:", train_dataset[0])
    # # 检查数据集大小
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    node_in_dim = train_dataset[0].x.shape[1]
    print("node_in_dim:", node_in_dim)
    print("\n")
    # 打印数据集标签的类别
    all_labels = [data.y.item() for data in train_dataset]
    num_classes = len(set(all_labels))
    print("num_classes:", num_classes)
    num_tasks = num_classes

    # 初始化模型
    classifier = GNNClassifier(
        num_layer=3,
        emb_dim=node_in_dim,
        hidden_dim=32,
        num_tasks=num_tasks
    )
    # # TODO: Train dd classifier
    best_model_state, history = train_gnn_classifier(
        classifier,
        train_dataset,
        valid_dataset,
        device
    )

    # 保存最佳模型
    save_to = '../best_gnnclassifier/best_gnn_classifier_' + data_name + '.pt'
    torch.save(best_model_state, save_to)
    print(f"GNNClassifier saved to {save_to}")
    classifier.load_state_dict(best_model_state)
    # classifier.load_state_dict(torch.load(save_to, weights_only=True))

    # 逐图测试嵌入
    print("val single graph pred_prob auc...")
    # 评估模型性能
    true_labels_val = []
    predicted_labels_val = []
    for graph in valid_dataset:
        true_label, predicted_label = evaluate_single_graph(classifier, graph, device)
        true_labels_val.append(true_label)
        predicted_labels_val.append(predicted_label)

    # 计算准确率
    accuracy = accuracy_score(true_labels_val, predicted_labels_val)
    print(f"Accuracy: {accuracy:.4f}")

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels_val, predicted_labels_val)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\n")

    # 逐图测试嵌入
    print("test single graph pred_prob auc...")
    # 评估模型性能
    true_labels_test = []
    predicted_labels_test = []
    for graph in test_dataset:
        true_label, predicted_label = evaluate_single_graph(classifier, graph, device)
        true_labels_test.append(true_label)
        predicted_labels_test.append(predicted_label)
    # print("true_labels_test：", true_labels_test)
    # print("predicted_labels_test：", predicted_labels_test)

    # 计算准确率
    accuracy = accuracy_score(true_labels_test, predicted_labels_test)
    print(f"Accuracy: {accuracy:.4f}")

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels_test, predicted_labels_test)
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    main()
