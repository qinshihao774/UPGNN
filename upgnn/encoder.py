import torch

from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax, add_remaining_self_loops, spmm
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import global_mean_pool

num_atom_type = 120  # mutag 8
# num_atom_type = 120  # including the extra mask tokens （原子类型的总数）。118原子类型的总数+额外掩码原子类型。
num_chirality_tag = 3  # 手性标签（chirality tags）的总数。 手性标签的总数（例如，3 种：无手性、R 构型、S 构型）。
num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens 4
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
             improved=False, add_self_loops=True, dtype=None,
             flow: str = "source_to_target"):
        ### assuming that self-loops have been already added in edge_index

        fill_value = 2. if improved else 1.

        assert flow in ['source_to_target', 'target_to_source']
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if add_self_loops:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        row, col = edge_index
        idx = col if flow == 'source_to_target' else row
        # deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='mean')
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
            # if isinstance(edge_index, Tensor):
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = self.norm(edge_index, edge_weight, num_nodes, dtype=x.dtype)
            else:
                edge_index, edge_weight = cache[0], cache[1]

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

    def forward(self, data, isbatch=True):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
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
            h = self.gnns[layer](h_list[layer], edge_index, edge_weight)
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
