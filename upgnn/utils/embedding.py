import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

# 通用
num_atom_type = 120  # including the extra mask tokens （原子类型的总数）。118原子类型的总数+额外掩码原子类型。
num_chirality_tag = 3  # 手性标签（chirality tags）的总数。 手性标签的总数（例如，3 种：无手性、R 构型、S 构型）。

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


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
    # 模型初始化方法
    def __init__(self, emb_dim, aggr="add", isuse_edge_attr=False):
        super(GCNConv, self).__init__()
        self.hidden_dim = 32
        # self.edge_weight = torch.nn.Linear(emb_dim, emb_dim)
        self.linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, self.hidden_dim), torch.nn.ReLU(),
                                          torch.nn.Linear(self.hidden_dim, emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)  # 边嵌入
        # self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        # 使用 Xavier 均匀初始化（Xavier Uniform Initialization）初始化两个嵌入层的权重。
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)  # 边嵌入初始化
        # torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr
        self.isuse_edge_attr = isuse_edge_attr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # 模型主调方法
    def forward(self, x, edge_index, edge_attr=None, edge_label=None):

        device = x.device

        # add_self_loops 函数，为每个节点添加自环边（self-loop）添加自环目的：GCN 需要自环来包含节点自身的特征。
        # 这会将 edge_index 的形状从 [2, num_edges] 扩展为 [2, num_edges + num_nodes]（例如，从 [2, 38] 到 [2, 38 + 17 = 55]）。
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        num_edge_features = edge_attr.size(1) if edge_attr is not None else num_bond_type
        # num_edge_features = num_bond_type

        if edge_label is not None:

            # 确保 edge_label 是 int64 类型
            edge_label = edge_label.to(dtype=torch.int64, device=device)

            # 生成一个 edge_attr 全0张量，用于存储 edge_label 的 one-hot 表示。
            edge_attr = torch.zeros((edge_label.size(0), num_edge_features), device=device, dtype=x.dtype)
            # scatter_ 操作，用于根据索引 (index) 将源值 (src) 填充到目标张量 (edge_attr) 的指定维度 (dim) 中。
            # 热编码（one-hot encoding），存储在 edge_attr 中。具体来说：
            # 对于每条边（edge_attr 的每一行），根据 edge_label 的值，在对应的列（由 edge_label 指定）设置值为 1，其余列保持为 0。
            # edge_attr = tensor([
            #     [1, 0, 0, 0],  # 边 0：类型 0（单键）
            #     [0, 1, 0, 0],  # 边 1：类型 1（双键）
            #     [0, 0, 1, 0],  # 边 2：类型 2（三键）
            #     [0, 0, 0, 1]   # 边 3：类型 3（芳香键）
            #     ])
            edge_attr.scatter_(1, edge_label.unsqueeze(1), 1)

        elif edge_attr is not None:
            edge_attr = edge_attr.to(device=device)

        # num_self_loops = x.size(0)  # 节点数目
        self_loop_attr = torch.zeros((x.size(0), num_edge_features), device=device, dtype=x.dtype)
        # torch.cat 期望所有参数都是 Tensor 类型。
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0) if edge_attr is not None else self_loop_attr

        edge_types = torch.argmax(edge_attr, dim=1)  # 形状: (num_edges + num_nodes,)
        edge_embeddings = self.edge_embedding1(edge_types)

        # 计算归一化系数
        norm = self.norm(edge_index, x.size(0), x.dtype)
        # 在消息传递前变换节点特征 -> 线性变换
        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)  # 调用 message

    def message(self, x_j, edge_attr, norm):  # 在 propagate 中，为每条边生成消息
        # 是否使用边属性仅针对于在训练模型时 加入边属性 对模型性能是否有提升
        if self.isuse_edge_attr:
            # edge_weight = self.edge_weight(edge_attr)  # 添加线性层增强边特征
            # return norm.view(-1, 1) * (x_j + edge_weight)
            return norm.view(-1, 1) * (x_j + edge_attr)
        else:
            return norm.view(-1, 1) * x_j  # 仅使用邻居节点特征

    def update(self, aggr_out):  # 在 propagate 中，聚合后调用。
        return self.linear(aggr_out)


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

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0.0, gnn_type="gcn"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)  # 将输入数据统一处理 将原子类型映射为嵌入向量
        # self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)  # 将手性标签 映射为嵌入向量

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)  # 随机初始化
        # torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs 管理神经网络层数
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, data, emb=False):
        # 获取单个图数据的所有属性值
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_label = getattr(data, 'edge_label', None)  # 安全获取 edge_label

        device = self.x_embedding1.weight.data.device
        # print(data.x)  # 检查 one-hot 编码
        data.atom_types = torch.argmax(data.x, dim=1) + 1  # 获取原子类型索引 从1开始
        # print(data.atom_types)  # 检查索引范围

        # weight 是 nn.Embedding 模块的可学习参数，表示嵌入矩阵。
        self.x_embedding1.weight.data = torch.cat(
            [torch.zeros(1, self.emb_dim).to(device), self.x_embedding1.weight.data[1:]], 0)

        # 如果数据集没有手性标签num_chirality_tag，self.x_embedding2（手性嵌入）不适用。
        # self.x_embedding2.weight.data = torch.cat(
        #     [torch.zeros(1, self.emb_dim).to(device), self.x_embedding2.weight.data[1:]], 0)

        x = self.x_embedding1(data.atom_types)  # 节点表示 形状: (num_nodes, emb_dim)

        h_list = [x]
        for layer in range(self.num_layer):  # 3层GCN
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr, edge_label)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                # h = F.dropout(h, self.drop_ratio, training=self.training)
                h = F.dropout(h, self.drop_ratio)
            else:
                # h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                h = F.dropout(F.relu(h), self.drop_ratio)
            h_list.append(h)

        ### Different implementations of Jk-concat 解决过平滑问题
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

        if emb:
            return global_mean_pool(node_representation, None), node_representation

        return global_mean_pool(node_representation, None)


class GNN_graphpred(torch.nn.Module):  # 预训练图预测
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gcn"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file, weights_only=False))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


# TODO:ogb
# def forward(self, x, edge_index, edge_attr): #edge_attr 是边特征矩阵，形状为 (num_edges, num_edge_features)
#     device = x.device
#     edge_index  = add_self_loops(edge_index, num_nodes=x.size(0))[0]
#
#     # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#
#     # add features corresponding to self-loop edges.
#
#     num_edge_features = edge_attr.size(1)
#
#     # 创建自环边特征，维度与 edge_attr 一致
#     self_loop_attr = torch.zeros(x.size(0), num_edge_features, device=edge_attr.device, dtype=edge_attr.dtype)
#     self_loop_attr[:, 0] = 0  # 自环边的 bond_type 设为 0
#     self_loop_attr[:, 1] = 0  # 自环边的 bond_direction 设为 0（无方向）
#     edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
#
#
#     # 计算边嵌入：bond_type 和 bond_direction 的嵌入相加
#     edge_emb1 = self.edge_embedding1(edge_attr[:, 0].long())  # bond_type 嵌入
#     edge_emb2 = self.edge_embedding2(edge_attr[:, 1].long())  # bond_direction 嵌入
#     edge_embeddings = edge_emb1 + edge_emb2  # 组合嵌入
#
#
#     norm = self.norm(edge_index, x.size(0), x.dtype)
#     x = F.relu(x)
#     # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)
#     # 图中每个节点经过图卷积操作后的新特征表示，包含了来自邻居节点和自环的信息。
#     return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)  # renew node Embedding

# # TODO:MUTAG
# def forward(self, x, edge_index, edge_attr):
#     device = x.device
#     orig_edge_attr = edge_attr
#     edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#     num_edge_features = orig_edge_attr.size(1) if orig_edge_attr is not None else 1
#     num_self_loops = x.size(0)
#     self_loop_attr = torch.zeros(num_self_loops, num_edge_features, device=device, dtype=x.dtype)
#     edge_attr = torch.cat((orig_edge_attr, self_loop_attr), dim=0) if orig_edge_attr is not None else self_loop_attr
#
#     edge_embeddings = self.edge_embedding1(edge_attr[:, 0].long())
#
#     row, col = edge_index
#     deg = degree(col, x.size(0), dtype=x.dtype)
#     deg_inv_sqrt = deg.pow(-0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#     norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#
#     x = F.relu(x)
#     return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)
