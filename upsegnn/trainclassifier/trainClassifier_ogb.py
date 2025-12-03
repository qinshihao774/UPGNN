import torch
import random
import numpy as np
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, add_remaining_self_loops
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.loader import DataLoader

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from upsegnn.downstream_model import MLP

patience = 12

# 通用
num_atom_type = 120  # including the extra mask tokens （原子类型的总数）。118原子类型的总数+额外掩码原子类型。
num_chirality_tag = 3  # 手性标签（chirality tags）的总数。 手性标签的总数（例如，3 种：无手性、R 构型、S 构型）。

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
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
        # self.edge_dim = edge_dim

        self.bias = True
        self.normalize = True
        self.improved = False
        self.add_self_loops = True
        self.cached = False

        self._cached_edge_index = None
        self._cached_adj_t = None

        # self.edge_linear = torch.nn.Linear(edge_dim, out_dim)

        self.linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, hidden_dim), torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_dim, out_dim))

        # self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)  # 边嵌入
        # self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        # 使用 Xavier 均匀初始化（Xavier Uniform Initialization）初始化两个嵌入层的权重。
        # torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)  # 边嵌入初始化
        # torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        if self.bias:
            self.bias = Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias', None)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     super().reset_parameters()
    #     self.linear.reset_parameters()
    #     zeros(self.bias)
    #     self._cached_edge_index = None
    #     self._cached_adj_t = None

    def norm(self, edge_index, edge_weight, num_nodes,
             improved=False, add_self_loops=True, flow="source_to_target", dtype=None):

        fill_value = 2. if improved else 1.

        # assert flow in ['source_to_target', 'target_to_source']
        # num_nodes = maybe_num_nodes(edge_index, num_nodes)

        # if isinstance(edge_index, SparseTensor):
        #     assert edge_index.size(0) == edge_index.size(1)
        #
        #     adj_t = edge_index
        #
        #     if not adj_t.has_value():
        #         adj_t = adj_t.fill_value(1., dtype=dtype)
        #     if add_self_loops:
        #         adj_t = torch_sparse.fill_diag(adj_t, fill_value)
        #
        #     deg = torch_sparse.sum(adj_t, dim=1)
        #     deg_inv_sqrt = deg.pow_(-0.5)
        #     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        #     adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        #     adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))
        #
        #     return adj_t

        if add_self_loops:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)

        if edge_weight is None:  # 所有边都是一样的重要
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        row, col = edge_index
        # idx = col if flow == 'source_to_target' else row
        # deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='mean')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):

        # print('edge_attr:', edge_attr)
        # 这里的边权重再某些情况下可与edge_attr 互换

        num_nodes = x.size(0)
        # edge_attr = edge_attr.float()

        # self_loop_attr = torch.zeros(num_nodes, edge_attr.size(1), device=edge_attr.device, dtype=edge_attr.dtype)
        # self_loop_attr[:, 0] = 4  # 自环边类型
        # edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        # if self.normalize:
        #     cache = self._cached_edge_index
        #     if cache is None:
        #         edge_index, edge_weight = self.norm(edge_index, edge_weight, num_nodes, dtype=x.dtype)
        #         if self.cached:
        #             self._cached_edge_index = (edge_index, edge_weight)
        #     else:
        #         edge_index, edge_weight = cache[0], cache[1]

        # if self.normalize:
        #     if edge_attr is not None:
        #         edge_index, edge_weight = self.norm(edge_index, edge_attr, num_nodes, dtype=x.dtype)
        #     elif edge_attr is None and edge_weight is not None:
        #         edge_index, edge_weight = self.norm(edge_index, edge_weight, num_nodes, dtype=x.dtype)
        #     else:
        #         edge_index, edge_weight = self.norm(edge_index, None, num_nodes, dtype=x.dtype)
        # else:
        #     edge_index, edge_weight = add_remaining_self_loops(edge_index, None, 1., num_nodes)

        if self.normalize:
            edge_index, edge_weight = self.norm(edge_index, edge_weight, num_nodes, dtype=x.dtype)
        else:
            edge_index, edge_weight = add_remaining_self_loops(edge_index, None, 1., num_nodes)

        x = self.linear(x)

        # # 处理边特征
        # edge_embeddings = None
        # if edge_attr is not None:
        #     edge_embeddings = self.edge_linear(edge_attr)  # [num_edges, out_dim]

        # out = self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_embeddings=edge_embeddings)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

        # if edge_embeddings is not None:
        #     return x_j + edge_embeddings
        # elif edge_weight is not None:
        #     return edge_weight.view(-1, 1) * x_j
        # return x_j

    #
    # def message(self, x_j, edge_attr, norm):
    #     # edge_weight = self.edge_weight(edge_attr)  # 添加线性层增强边特征
    #     # return norm.view(-1, 1) * (x_j + edge_weight)
    #
    #     return norm.view(-1, 1) * (x_j + edge_attr)
    #     # return norm.view(-1, 1) * x_j  # 仅使用邻居节点特征

    # def update(self, aggr_out):
    #     return self.linear(aggr_out)

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

    def __init__(self, num_layer, emb_dim, hidden_dim=32, out_dim=32, JK="last", drop_ratio=0.0,
                 gnn_type="gcn"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)  # 将原子类型映射为嵌入向量
        # self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)  # 将手性标签 映射为嵌入向量

        # torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)  # 随机初始化
        # torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs 管理神经网络层数
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

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))

    def forward(self, data, isbatch=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

        x = x.float()

        h_list = [x]
        for layer in range(self.num_layer):  # 3层GCN
            h = self.gnns[layer](h_list[layer], edge_index, edge_weight)  # 这里的edge_attr就是权重
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
        # mlp_3_64hidden =》 num_tasks 分类 因为输入的是节点表示，所以是32
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
        if isbatch:
            batch = data.batch
            embed, node_embed = self.gnn(data, isbatch=isbatch)
            graph_representation = self.pool(node_embed, batch)
        else:
            embed, node_embed = self.gnn(data, isbatch=isbatch)
            graph_representation = self.pool(node_embed, None)

        return self.mlp(graph_representation)  # 返回没有归一化的输出

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, weights_only=True))


def train_gnn_classifier(model, train_dataset, val_dataset, device, num_epochs=600, lr=0.0005):
    model = model.to(device)
    # GNN 在训练过程中能够区分批量中的单个图信息结构。
    # 在 PyG 中，DataLoader 将多个图（Data 对象）组合成一个 Batch 对象。
    # 每个 Batch 对象包含了一组图的节点特征、边信息、标签等，并通过一个特殊的属性 batch（一个张量）来标识每个节点属于哪个图。
    # 创建了一个数据加载器，每次迭代返回一个 Batch 对象（batch_data），其中包含 batch_size=4 个图的合并数据。

    # train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, follow_batch=['edge_attr'])
    # val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, follow_batch=['edge_attr'])

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

    best_val_roc_auc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_preds, train_labels, train_probs = 0.0, [], [], []

        for i, batch_data in enumerate(train_loader):
            # batch_data 是一个 Batch 对象
            batch = batch_data.to(device)
            # batch.y = torch.where(batch.y == -1, torch.tensor(0.0), batch.y)
            batch.y = batch.y.squeeze(1).long()  # 去除一个维度

            # 调试 batch 信息
            assert batch.batch.max() < batch.num_graphs, f"Batch index {batch.batch.max()} exceeds num_graphs {batch.num_graphs}"
            # print(f"Batch: x {batch.x.shape}, batch max {batch.batch.max()}, num_graphs {batch.num_graphs}")

            optimizer.zero_grad()
            out = model(batch, isbatch=True)  # logits（即未经过 softmax 处理的得分） = out

            # try:
            #     loss = criterion(out, batch.y)
            # except Exception as e:
            #     print(f"Error calculating loss: {e}")
            #     print(f"out: {out}")
            #     print(f"batch.y: {batch.y}")
            #     continue

            # torch.nn.CrossEntropyLoss 期望的输入 input 是 logits（即未经过 softmax 处理的得分）

            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
            train_preds.extend(out.argmax(dim=1).cpu().numpy())
            # print("train_preds:" , train_preds)
            train_labels.extend(batch.y.cpu().numpy())
            # print("train_labels:" , train_labels)
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
                batch.y = torch.where(batch.y == -1, torch.tensor(0.0), batch.y)
                batch.y = batch.y.squeeze(1).long()  # 去除一个维度
                assert batch.batch.max() < batch.num_graphs, f"Batch index {batch.batch.max()} exceeds num_graphs {batch.num_graphs}"
                out = model(batch, isbatch=True)
                # pred = torch.argmax(out, dim=1)
                loss = criterion(out, batch.y)

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

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, ROC-AUC: {train_roc_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}')
        # 当一组批量所计算的误差值低于之前最小的误差值时，则更新模型参数，并重置 patience 计数器。
        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            best_model_state = model.state_dict()

            # patience_counter = 0
        # else:  # 在最佳状态后 延后30组批量
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f'早停触发，在第 {epoch + 1} 轮停止')
        #         break
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
        # print("True Label:", true_label)
        # print("Predicted Label:", predicted_label)
    return true_label, predicted_label


def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO ogb: Data(edge_index=[2, 50], edge_attr=[50, 3], x=[24, 9], y=[1, 1], num_nodes=24)
    data_name = "ogb"
    dataset = torch.load("../data/ogb/ogb_graph.pt", weights_only=False)
    print(f"数据集大小: {len(dataset)}")

    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx['train']]
    valid_dataset = dataset[split_idx['valid']]
    test_dataset = dataset[split_idx['test']]

    print("single data:", train_dataset[0])
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    node_in_dim = train_dataset[0].x.shape[1]
    print("node_in_dim:", node_in_dim)
    # 边属性维度
    edge_in_dim = train_dataset[0].edge_attr.shape[1]
    print("edge_in_dim:", edge_in_dim)


    # 文件在哪执行 路径就根据执行文件的路径进行设置
    # make_balanced_dataset(train_dataset, 'train')
    # make_balanced_dataset(valid_dataset, 'valid')

    new_train_dataset = torch.load("../data/ogb/train_dataset_balanced.pt", weights_only=False)
    new_valid_dataset = torch.load("../data/ogb/valid_dataset_balanced.pt", weights_only=False)
    # print(f"数据集大小: {len(new_train_dataset)}")
    # print(f"数据集大小: {len(new_valid_dataset)}")

    # 检查数据集标签
    all_labels = [data.y.item() for data in new_train_dataset]
    num_classes = len(set(all_labels))
    print("num_classes:", num_classes)
    num_tasks = num_classes

    # 统计标签数量和比例
    label_counts = Counter(all_labels)
    total_samples = len(all_labels)
    #
    print("\n训练集标签分布：")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        proportion = count / total_samples * 100
        print(f"标签 {label}: {count} 个, 占比 {proportion:.2f}%")

    # 初始化模型  如果模型出现Nan 或者 inf 注意模型初始化参数比如 层数 中间维度
    classifier = GNNClassifier(
        num_layer=2,
        emb_dim=node_in_dim,
        hidden_dim=32,
        num_tasks=num_tasks
    )

    # best_model_state, history = train_gnn_classifier(
    #     classifier,
    #     new_train_dataset,
    #     new_valid_dataset,
    #     device
    # )
    save_to = '../best_gnnclassifier/best_gnn_classifier_' + data_name + '.pt'
    # 保存最佳模型
    # torch.save(best_model_state, save_to)
    # print(f"GNNClassifier saved to {save_to}")

    classifier.load_state_dict(torch.load(save_to, weights_only=True))

    # 逐图测试嵌入
    print("val single graph pred_prob auc...")
    # 评估模型性能
    true_labels_val = []
    predicted_labels_val = []
    for graph in new_valid_dataset:
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

    # 逐图测试嵌入
    print("test single graph pred_prob auc...")
    # 评估模型性能
    true_labels_test = []
    predicted_labels_test = []
    for graph in test_dataset:
        true_label, predicted_label = evaluate_single_graph(classifier, graph, device)
        true_labels_test.append(true_label)
        predicted_labels_test.append(predicted_label)

    # 计算准确率
    # print("true_labels_test：", true_labels_test)
    # print("predicted_labels_test：", predicted_labels_test)

    accuracy = accuracy_score(true_labels_test, predicted_labels_test)
    print(f"Accuracy: {accuracy:.4f}")

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels_test, predicted_labels_test)
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    main()
