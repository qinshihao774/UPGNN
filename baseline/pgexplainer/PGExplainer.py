import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import *
from munch import Munch
from torch import Tensor
from torch.nn import ReLU
from torch_scatter import scatter
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from torch_geometric.nn import Linear
from torch_geometric.utils import k_hop_subgraph
from upsegnn.dataset.mutag import Mutag
from upsegnn.model import fn_softedgemask
from upsegnn.trainclassifier.trainClassifier_mutag import GNNClassifier

from base import InstanceExplainAlgorithm
from utils import set_masks, clear_masks


class PGExplainer(InstanceExplainAlgorithm):
    r"""
    An implementation of PGExplainer in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>'.
    """
    coeffs = {
        'reg_size': 0.05,
        'reg_ent': 1.0,
        'temp0': 5.0,
        'temp1': 2.0,
        'EPS': 1e-15,
        'edge_reduction': 'sum',
        'bias': 0.001,
        'knn_loss': 1.
    }

    def __init__(self, model: nn.Module, device, epochs: int = 50, gnn_task: str = 'graph', **kwargs):
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
        self.load_state_dict(torch.load(file_path, map_location=self.device, weights_only=False))

    def train_loop(
            self,
            data: Data,
            model_to_explain: nn.Module,
            epoch: int,
            use_edge_weight: bool = False,
            apply_sigmoid: bool = True,
            **kwargs
    ) -> Munch:
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        if (embeds := data.get('ori_embeddings')) is None:
            with torch.no_grad():
                embeds = [model_to_explain.gnn(x, edge_index, emb=True)[1]]  # Use gnn method

        if (batch := data.get('batch')) is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
        edge_batch = batch[edge_index[0]]

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
            mask = F.sigmoid(mask)  # Apply sigmoid for explanation phase
            return mask

        def get_gnn_pred(mask: Tensor) -> Tensor:
            set_masks(model_to_explain, mask, edge_index)
            pred = model_to_explain(data)
            clear_masks(model_to_explain)
            return pred

        if not isinstance(embeds, List):
            embeds = [embeds]

        hard_masks = data.get('hard_masks', [None] * len(embeds))
        masks = [get_explainer_pred(embed, hard_mask) for embed, hard_mask in zip(embeds, hard_masks)]
        masked_pred = [get_gnn_pred(mask) for mask in masks]

        if self.gnn_task == 'node':
            corn_node_id = data.get('corn_node_id')
            assert len(masked_pred) == len(corn_node_id)
            masked_pred = [pred[node_id] for pred, node_id in zip(masked_pred, corn_node_id)]

        target_label = data.y if hasattr(data, 'y') else data.get('target_label')
        if not isinstance(target_label, List):
            target_label = [target_label]
        loss_dict = self.__loss__(masked_pred, target_label, masks, edge_batch, apply_sigmoid)

        return loss_dict

    # def forward(
    #         self,
    #         data: Data,
    #         **kwargs
    # ) -> Tuple[Tensor, Tensor]:
    #     assert (embeds := data.get('ori_embeddings')) is not None
    #     edge_index = data.edge_index.to(self.device)
    #     if (corn_node_id := data.get('corn_node_id')) is not None and kwargs.get("sample_subgraph"):
    #         edge_index = k_hop_subgraph(
    #             corn_node_id,
    #             kwargs.get('num_hops', 3),
    #             data.edge_index,
    #             relabel_nodes=False
    #         )[1]
    #
    #     if (batch := data.get('batch')) is None:
    #         # 修复：添加dtype和device
    #         batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
    #     edge_batch = batch[edge_index[0]]
    #
    #     if not isinstance(embeds, List): embeds = [embeds]
    #     hard_masks = data.get('hard_masks', [None] * len(embeds))
    #     masks = [self.get_explainer_pred(embed.to(self.device), edge_index, corn_node_id, edge_batch, hard_mask)
    #              for embed, hard_mask in zip(embeds, hard_masks)]
    #
    #     edge_mask = masks[0]  # Take first mask for single embedding
    #     hard_edge_mask = (edge_mask > kwargs.get('threshold', 40)).float()
    #     return edge_mask, hard_edge_mask

    def forward(
            self,
            data: Data,
            **kwargs
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""
        Generating explanation for given graph.
        Args:
            data: a PYG Data which storage information of graph to be explained.
        Returns:
            Union[Tensor, Tensor]: edge_mask, feat_mask
        """
        assert (embeds := data.get('ori_embeddings')) is not None

        edge_index = data.edge_index
        if (corn_node_id := data.get('corn_node_id')) is not None and kwargs.get("sample_subgraph"):
            edge_index = k_hop_subgraph(
                corn_node_id,
                kwargs.get('num_hops', 3),
                data.edge_index,
                relabel_nodes=False
            )[1]

        if (batch := data.get('batch')) is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)  # 修复：添加dtype和device
        edge_batch = batch[edge_index[0]]

        if not isinstance(embeds, List): embeds = [embeds]
        hard_masks = data.get('hard_masks', [None] * len(embeds))
        masks = [self.get_explainer_pred(embed, edge_index, corn_node_id, edge_batch, hard_mask)
                 for embed, hard_mask in zip(embeds, hard_masks)]

        return masks

    @property
    def name(self):
        return 'pgexplainer'

    def get_explainer_pred(self, embed: Tensor, edge_index, corn_node_id, edge_batch, hard_mask: Tensor) -> Tensor:
        # Sample possible explanation
        expl_inputs = self._create_inputs(
            embed,
            edge_index,
            corn_node_id,
            edge_batch,
            hard_mask).unsqueeze(dim=0)
        logits = self.explainer(expl_inputs)[0]
        mask = self._concrete_sample(logits, training=False).squeeze()
        return mask


def create_hard_edge_mask(edge_mask: Tensor, k: int = 7) -> Tensor:
    """
    Create a hard edge mask by selecting the top k edges with the highest values in edge_mask.

    Args:
        edge_mask (Tensor): A 1D tensor containing importance scores for edges.
        k (int): Number of top edges to select (default is 7).

    Returns:
        Tensor: A binary tensor (hard_edge_mask) with 1s for the top k edges and 0s elsewhere.
    """
    device = edge_mask.device
    num_edges = edge_mask.size(0)

    # If the number of edges is less than or equal to k, select all edges
    if num_edges <= k:
        hard_edge_mask = torch.ones_like(edge_mask, dtype=torch.bool, device=device)
    else:
        # Get the indices of the top k values
        _, top_indices = torch.topk(edge_mask, k, dim=0)
        # Initialize hard_edge_mask with zeros
        hard_edge_mask = torch.zeros_like(edge_mask, dtype=torch.bool, device=device)
        # Set the top k indices to True
        hard_edge_mask[top_indices] = True

    return hard_edge_mask


# def evaluate_fidelity_plus(explainer, classifier, dataset, device):
#     classifier.eval()
#     explainer.eval()
#     total_loss = 0.0
#     num_graphs = 0
#
#     for graph in dataset:
#         graph = graph.to(device)
#         if graph.num_nodes == 0 or graph.edge_index.size(1) == 0:
#             print("Warning: 跳过空图，无节点或无边")
#             continue
#
#         if graph.y.dim() > 1:
#             graph.y = graph.y.squeeze(-1).long()
#         elif graph.y.dim() == 0:
#             graph.y = graph.y.unsqueeze(0).long()
#
#         with torch.no_grad():
#             _, node_embed = classifier.gnn(graph, emb=True)
#             pred_orig = classifier(graph)
#             graph.ori_embeddings = node_embed
#
#             masks = explainer(graph)
#             edge_mask = masks[0]  # 取第一个（通常只有一个嵌入）
#
#             masked_data = generate_subgraph(graph, edge_mask, graph.edge_index, graph.edge_attr)
#
#             if masked_data is None or masked_data.num_nodes == 0:
#                 print("Warning: 掩码后子图为空，跳过该图计算")
#                 continue
#
#             pred_masked = classifier(masked_data)
#             loss = F.mse_loss(F.softmax(pred_orig, dim=-1), F.softmax(pred_masked, dim=-1))
#             total_loss += loss.item()
#             num_graphs += 1
#
#     avg_fidelity = total_loss / max(num_graphs, 1)
#     print(f"平均 Fidelity+: {avg_fidelity:.4f}")
#     return avg_fidelity
#
#
# def evaluate_fidelity_minus(explainer, classifier, dataset, device):
#     classifier.eval()
#     explainer.eval()
#     total_loss = 0.0
#     num_graphs = 0
#
#     for graph in dataset:
#         graph = graph.to(device)
#         if graph.num_nodes == 0 or graph.edge_index.size(1) == 0:
#             print("Warning: 跳过空图，无节点或无边")
#             continue
#
#         if graph.y.dim() > 1:
#             graph.y = graph.y.squeeze(-1).long()
#         elif graph.y.dim() == 0:
#             graph.y = graph.y.unsqueeze(0).long()
#
#         with torch.no_grad():
#             _, node_embed = classifier.gnn(graph, emb=True)
#             pred_orig = classifier(graph)
#             graph.ori_embeddings = node_embed
#
#             masks = explainer(graph)
#             edge_mask = masks[0]  # 取第一个（通常只有一个嵌入）
#
#             masked_data = generate_subgraph(graph, edge_mask, graph.edge_index, graph.edge_attr)
#
#             if masked_data is None or masked_data.num_nodes == 0:
#                 print("Warning: 掩码后子图为空，跳过该图计算")
#                 continue
#
#             pred_masked = classifier(masked_data)
#             loss = F.mse_loss(F.softmax(pred_orig, dim=-1), F.softmax(pred_masked, dim=-1))
#             total_loss += loss.item()
#             num_graphs += 1
#
#     avg_fidelity = total_loss / max(num_graphs, 1)
#     print(f"平均 Fidelity+: {avg_fidelity:.4f}")
#     return avg_fidelity

def evaluate_fidelity_plus(explainer, classifier, dataset, device):
    """
    Calculate Fidelity+ metric to evaluate the impact of identified substructures on model predictions.
    Args:
        explainer: The explainer model.
        classifier: The target classifier model.
        dataset: List of PyG Data objects (each representing a graph).
        device: torch.device, specifying the computation device.
    Returns:
        avg_fidelity: Average Fidelity+ score.
    """
    classifier.eval()
    classifier.to(device)
    explainer.eval()
    explainer.to(device)

    num_graphs = 0
    total_fidelity = 0.0

    for graph in dataset:
        graph = graph.to(device)
        if graph.num_nodes == 0 or graph.edge_index.size(1) == 0:
            print("Warning: Skipping empty graph (no nodes or edges)")
            continue

        # Ensure graph.y is a 1D tensor
        if graph.y.dim() > 1:
            graph.y = graph.y.squeeze(-1).long()
        elif graph.y.dim() == 0:
            graph.y = graph.y.unsqueeze(0).long()

        # Original graph prediction: f(G)
        _, node_embed = classifier.gnn(graph, isbatch=False)
        graph.ori_embeddings = node_embed
        pred_orig = classifier(graph)
        pred_orig_prob = F.softmax(pred_orig, dim=-1)
        pred_orig_label = pred_orig.argmax(dim=-1)

        # Generate explanation
        masks = explainer(graph)
        edge_mask = masks[0]  # 取第一个（通常只有一个嵌入）
        # edge_mask = edge_mask.detach()  # Detach to avoid gradient tracking
        # hard_edge_mask = create_hard_edge_mask(edge_mask)  # 修改：使用top-7 edges

        # Generate subgraph
        # masked_data = generate_subgraph(graph, hard_edge_mask, edge_index, edge_attr, edge_label)
        masked_data = fn_softedgemask(graph, edge_mask, isFidelitPlus=True)  # 需实现：用掩码过滤节点/边

        if masked_data is None or masked_data.num_nodes == 0:
            print("Warning: Empty subgraph after masking, skipping...")
            continue

        # Subgraph prediction: f(G \ S_i)
        pred_masked = classifier(masked_data)
        pred_masked_prob = F.softmax(pred_masked, dim=-1)

        # Calculate Fidelity+ score
        if pred_orig_label == graph.y:
            fidelity_score = torch.norm(pred_orig_prob - pred_masked_prob, p=2).item()
        else:
            fidelity_score = 0.0
        total_fidelity += fidelity_score
        num_graphs += 1

    avg_fidelity = total_fidelity / max(num_graphs, 1)
    print(f"Average Fidelity+: {avg_fidelity:.4f}")
    return avg_fidelity


def evaluate_fidelity_minus(explainer, classifier, dataset, device):
    """
    Calculate Fidelity+ metric to evaluate the impact of identified substructures on model predictions.
    Args:
        explainer: The explainer model.
        classifier: The target classifier model.
        dataset: List of PyG Data objects (each representing a graph).
        device: torch.device, specifying the computation device.
    Returns:
        avg_fidelity: Average Fidelity+ score.
    """
    classifier.eval()
    classifier.to(device)
    explainer.eval()
    explainer.to(device)

    num_graphs = 0
    total_fidelity = 0.0

    for graph in dataset:
        graph = graph.to(device)
        if graph.num_nodes == 0 or graph.edge_index.size(1) == 0:
            print("Warning: Skipping empty graph (no nodes or edges)")
            continue

        # Ensure graph.y is a 1D tensor
        if graph.y.dim() > 1:
            graph.y = graph.y.squeeze(-1).long()
        elif graph.y.dim() == 0:
            graph.y = graph.y.unsqueeze(0).long()

        # Original graph prediction: f(G)
        _, node_embed = classifier.gnn(graph, isbatch=False)
        graph.ori_embeddings = node_embed
        pred_orig = classifier(graph)
        pred_orig_prob = F.softmax(pred_orig, dim=-1)
        pred_orig_label = pred_orig.argmax(dim=-1)

        # Generate explanation
        masks = explainer(graph)
        edge_mask = masks[0]  # 取第一个（通常只有一个嵌入）
        # edge_mask = edge_mask.detach()  # Detach to avoid gradient tracking
        # hard_edge_mask = create_hard_edge_mask(edge_mask)  # 修改：使用top-7 edges

        # Generate subgraph
        # masked_data = generate_subgraph(graph, hard_edge_mask, edge_index, edge_attr, edge_label)
        masked_data = fn_softedgemask(graph, edge_mask, isFidelitPlus=False)  # 需实现：用掩码过滤节点/边

        if masked_data is None or masked_data.num_nodes == 0:
            print("Warning: Empty subgraph after masking, skipping...")
            continue

        # Subgraph prediction: f(G \ S_i)
        pred_masked = classifier(masked_data)
        pred_masked_prob = F.softmax(pred_masked, dim=-1)

        # Calculate Fidelity+ score
        if pred_orig_label == graph.y:
            fidelity_score = torch.norm(pred_orig_prob - pred_masked_prob, p=2).item()
        else:
            fidelity_score = 0.0
        total_fidelity += fidelity_score
        num_graphs += 1

    avg_fidelity = total_fidelity / max(num_graphs, 1)
    print(f"Average Fidelity-: {avg_fidelity:.4f}")
    return avg_fidelity


def generate_subgraph(graph, edge_mask: Tensor, edge_index: Tensor, edge_attr: Tensor = None):
    """Generate subgraph based on edge mask."""
    device = graph.x.device
    # threshold = edge_mask.mean()
    # # print(f"threshold: {threshold:.4f}")
    # hard_edge_mask = (edge_mask > threshold).float()
    # #
    # # # 根据硬边掩码筛选边
    # selected_edges = hard_edge_mask.bool()

    # Use top-k instead of threshold

    k = 7
    num_edges = edge_mask.size(0)
    # if num_edges <= k:
    #     selected_edges = torch.ones_like(edge_mask, dtype=torch.bool)
    # else:
    #     _, top_indices = torch.topk(edge_mask, k)
    #     selected_edges = torch.zeros_like(edge_mask, dtype=torch.bool)
    #     selected_edges[top_indices] = True
    #

    # If the number of edges is less than or equal to k, select all edges
    if num_edges <= k:
        hard_edge_mask = torch.ones_like(edge_mask, dtype=torch.bool, device=device)
    else:
        # Get the indices of the top k values
        _, top_indices = torch.topk(edge_mask, k, dim=0)
        # Initialize hard_edge_mask with zeros
        hard_edge_mask = torch.zeros_like(edge_mask, dtype=torch.bool, device=device)
        # Set the top k indices to True
        hard_edge_mask[top_indices] = True

    # print(f"Selected {selected_edges.sum().item()} edges out of {num_edges}")

    sub_edge_index = edge_index[:, hard_edge_mask]

    # 筛选对应的边属性
    sub_edge_attr = None
    if edge_attr is not None:
        sub_edge_attr = edge_attr[hard_edge_mask]  # 形状：[num_selected_edges, num_edge_features]


    else:
        # If edge_attr is None, check if edge_label exists and use it to create edge_attr
        if hasattr(graph, 'edge_label') and graph.edge_label is not None:

            sub_edge_label = graph.edge_label[hard_edge_mask]

            # Assume edge_label is integer labels, one-hot encode them

            num_edge_types = int(graph.edge_label.max().item() + 1)  # Number of unique edge labels

            sub_edge_attr = F.one_hot(sub_edge_label.long(), num_classes=num_edge_types).float()

        elif sub_edge_index.size(1) > 0:

            # Fallback to dummy attr if no edge_label

            sub_edge_attr = torch.ones((sub_edge_index.size(1), 6), dtype=torch.float,
                                       device=device)  # Assuming 4 features for MUTAG-like datasets

    # 提取子图中的节点
    subset = torch.unique(sub_edge_index)
    if subset.numel() == 0:
        return None  # 空子图

    # 使用 torch_geometric.utils.subgraph 生成子图
    sub_edge_index, sub_edge_attr = subgraph(
        subset, sub_edge_index, edge_attr=sub_edge_attr, relabel_nodes=True, num_nodes=graph.num_nodes
    )
    sub_x = graph.x[subset]
    sub_y = graph.y  # 图级标签保持不变

    # 创建子图对象
    sub_graph = Data(
        x=sub_x.to(device),
        edge_index=sub_edge_index.to(device),
        edge_attr=sub_edge_attr.to(device) if sub_edge_attr is not None else None,
        y=sub_y.to(device),
        num_nodes=subset.size(0),
        batch=torch.zeros(sub_x.size(0), dtype=torch.long, device=device)  # 单图的 batch 属性
    )

    return sub_graph


def map_labels(dataset):
    """Map -1 labels to 0."""
    new_dataset = []
    for data in dataset:
        data.y = torch.where(data.y == -1, torch.tensor(0, dtype=data.y.dtype), data.y).squeeze()
        new_dataset.append(data)
    return new_dataset


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_name = 'mutag'
    dataset_root = '../../upsegnn/data/mutag'

    train_dataset = Mutag(root=dataset_root, split='train')
    valid_dataset = Mutag(root=dataset_root, split='val')
    test_dataset = Mutag(root=dataset_root, split='test')

    train_dataset = map_labels(train_dataset)
    valid_dataset = map_labels(valid_dataset)
    test_dataset = map_labels(test_dataset)

    print("data:", train_dataset[0])
    node_in_dim = train_dataset[0].x.shape[1]
    print("node_in_dim:", node_in_dim)
    all_labels = [data.y.item() for data in train_dataset]
    num_classes = len(set(all_labels))
    print(f"Number of classes: {num_classes}")

    classifier = GNNClassifier(
        num_layer=3,
        emb_dim=node_in_dim,
        hidden_dim=32,
        num_tasks=num_classes
    )
    classifier_path = '../../upsegnn/best_gnnclassifier/best_gnn_classifier_' + data_name + '.pt'
    classifier.load_state_dict(torch.load(classifier_path, weights_only=True))
    classifier.to(device)

    explainer = PGExplainer(classifier, device, gnn_task='graph')
    epochs = explainer.epochs

    optimizer = torch.optim.Adam(explainer.parameters(), lr=0.001)
    explainer.train()
    classifier.eval()
    for epoch in range(epochs):
        total_loss = 0.0
        for graph in train_dataset:
            graph = graph.to(device)
            if graph.y.dim() > 1:
                graph.y = graph.y.squeeze(-1).long()
            elif graph.y.dim() == 0:
                graph.y = graph.y.unsqueeze(0).long()

            with torch.no_grad():
                _, node_embed = classifier.gnn(graph, isbatch=False)
                graph.ori_embeddings = node_embed.to(device)

            loss_dict = explainer.train_loop(
                data=graph,
                model_to_explain=classifier,
                epoch=epoch,
                use_edge_weight=False,  # Use set_masks instead of edge_weights
                apply_sigmoid=True
            )

            loss = sum(loss_dict.values())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_dataset):.4f}")

    os.makedirs(f'./pge_pretrained/{data_name}', exist_ok=True)
    torch.save(explainer.state_dict(), f'./pge_pretrained/{data_name}/{data_name}.pkl')
    explainer.load_parameters(path='./pge_pretrained', dataset=data_name)

    print("Testing graph classification explanation...")
    true_labels = []
    predicted_labels = []
    skipped_graphs = 0

    for graph in valid_dataset:
        try:
            graph = graph.to(device)
            if graph.y.dim() > 1:  # 张量
                graph.y = graph.y.squeeze(-1).long()
            elif graph.y.dim() == 0:  # 标量
                graph.y = graph.y.unsqueeze(0).long()

            with torch.no_grad():
                _, node_embed = classifier.gnn(graph, isbatch=False)
                graph.ori_embeddings = node_embed
                # graph.hard_mask = explainer.get_hard_mask(graph)

            # Generate explanation
            masks = explainer(graph)
            edge_mask = masks[0]  # 取第一个（通常只有一个嵌入）

            # Generate subgraph
            masked_data = generate_subgraph(graph, edge_mask, graph.edge_index, graph.edge_attr)

            if masked_data is None or masked_data.num_nodes == 0 or masked_data.edge_index.size(1) == 0:
                print(f"Warning: Empty subgraph for graph {graph}, skipping...")
                skipped_graphs += 1
                continue

            logits = classifier(masked_data)
            pred_prob = F.softmax(logits, dim=-1)  # 概率和为1
            # pred_prob = torch.sigmoid(logits).squeeze() #  二分类概率各自属于[0,1]区间，但是各个分类概率互不影响

            predicted_label = torch.argmax(pred_prob).item()
            true_label = graph.y.item()

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

        except Exception as e:
            print(f"Error processing graph: {e}")
            skipped_graphs += 1
            continue

    print(f"Skipped graphs: {skipped_graphs}")

    if true_labels:
        accuracy = accuracy_score(true_labels, predicted_labels)
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
    else:
        print("Warning: No valid predictions to compute metrics.")

    print("Evaluating fidelity...")
    evaluate_fidelity_plus(explainer, classifier, valid_dataset, device)
    evaluate_fidelity_minus(explainer, classifier, valid_dataset, device)
    print("--------------done!-----------------")
