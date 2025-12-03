import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from upsegnn.utils.embedding import GNN
from upsegnn.downstream_model import MLP
class GNNClassifier(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, hidden_dim, num_tasks=2, JK="last", drop_ratio=0.0,
                 graph_pooling="mean", gnn_type="gcn"):
        super(GNNClassifier, self).__init__()
        # gcn_3l
        self.gnn = GNN(num_layer=num_layer, emb_dim=emb_dim, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type)
        # mlp_3_64hidden =》 num_tasks 分类
        self.mlp = MLP(num_layer=num_layer, emb_dim=emb_dim, hidden_dim=hidden_dim, graph_class=num_tasks)

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

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        embed, node_embed = self.gnn(data, emb=True)
        graph_representation = self.pool(node_embed, batch)
        return self.mlp(graph_representation) #返回没有归一化的输出

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, weights_only=True))
