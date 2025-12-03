from dig.xgraph.method.base_explainer import ExplainerBase
from dig.xgraph.method.utils import symmetric_edge_mask_indirect_graph

from upsegnn import trainClassifier_ogb
from upsegnn.dataset.mutag import Mutag
from upsegnn.trainclassifier import trainClassifier_mutag
from sklearn.metrics import accuracy_score, confusion_matrix
from upsegnn.model import mask_fn_edgemask

import torch
import os
from torch import Tensor
from torch_geometric.utils import subgraph, add_remaining_self_loops
from dig.version import debug
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
from typing import Union
from torch_geometric.data import Data

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

EPS = 1e-5


class GNNExplainer(ExplainerBase):
    r"""GNNExplainer for identifying important subgraph structures and features for graph classification.
    Args:
        model (torch.nn.Module): The GNN model to explain.
        epochs (int, optional): Number of training epochs. (default: 100)
        lr (float, optional): Learning rate. (default: 0.01)
        coff_edge_size (float, optional): Coefficient for edge mask size penalty. (default: 0.01)
        coff_edge_ent (float, optional): Coefficient for edge mask entropy penalty. (default: 0.01)
        explain_graph (bool, optional): Whether to explain graph classification. (default: True)
        indirect_graph_symmetric_weights (bool, optional): Symmetrize edge weights for indirect graphs. (default: False)
    """

    def __init__(self,
                 model: torch.nn.Module,
                 epochs: int = 25,
                 lr: float = 0.0032,
                 coff_edge_size: float = 0.01,
                 coff_edge_ent: float = 0.01,
                 coff_node_feat_size: float = 0.0,  # Disabled for graph classification
                 coff_node_feat_ent: float = 0.0,  # Disabled for graph classification
                 explain_graph: bool = True,
                 indirect_graph_symmetric_weights: bool = False):
        # GNNExplainer接收一个已训练的model（GNN分类器） 但是本身不训练任何参数。
        super().__init__(model, epochs, lr, explain_graph)
        self.coff_edge_size = coff_edge_size
        self.coff_edge_ent = coff_edge_ent
        self.coff_node_feat_size = coff_node_feat_size
        self.coff_node_feat_ent = coff_node_feat_ent
        self._symmetric_edge_mask_indirect_graph = indirect_graph_symmetric_weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def __loss__(self, raw_preds: Tensor, y_label: Union[Tensor, int]) -> Tensor:
        """Compute loss for graph classification explanation."""
        if not self.explain_graph:
            raise ValueError("This explainer is configured for graph classification only.")

        # Ensure y_label is a 1D tensor with batch size 1
        if y_label.dim() > 1:
            y_label = y_label.squeeze(-1).long()  # Squeeze only the last dimension if needed
        elif y_label.dim() == 0:
            y_label = y_label.unsqueeze(0).long()  # Add batch dimension if scalar
        loss = cross_entropy_with_logit(raw_preds, y_label)

        # Edge mask regularization
        m = self.edge_mask.sigmoid()
        loss = loss + self.coff_edge_size * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coff_edge_ent * ent.mean()

        return loss

    def gnn_explainer_alg(self, graph: Tensor, y_label: Tensor) -> Tensor:
        """Train to obtain edge mask for graph explanation."""
        # 这就是优化循环！它使用Adam优化self.edge_mask（边掩码），在self.epochs轮内迭代
        optimizer = torch.optim.Adam([self.edge_mask], lr=0.001)  # Only optimize edge mask

        for epoch in range(1, self.epochs + 1):
            raw_preds = self.model(graph)
            loss = self.__loss__(raw_preds, y_label)

            if epoch % 20 == 0 and debug:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.edge_mask, clip_value=2.0)
            optimizer.step()

        return self.edge_mask.sigmoid()

    def forward(self, x: Tensor, edge_index: Tensor, target_label: Tensor = None, **kwargs):
        """Generate edge masks for graph classification."""
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()
        data = kwargs.get("g")

        edge_label = data.edge_label if hasattr(data, 'edge_label') else None
        edge_attr = data.edge_attr.to(self.device) if hasattr(data, 'edge_attr') else None

        # Add self-loops and move data to device
        fill_value = 0 if edge_attr is not None else 1.0
        self_loop_edge_index, self_loop_edge_attr = add_remaining_self_loops(
            edge_index, edge_attr=edge_attr, fill_value=fill_value, num_nodes=x.shape[0]
        )

        if edge_label is not None:
            num_self_loops = self_loop_edge_index.size(1) - edge_index.size(1)
            self_loop_edge_label = torch.cat(
                [edge_label, torch.zeros(num_self_loops, dtype=edge_label.dtype, device=self.device)])
        else:
            self_loop_edge_label = None

        # Ensure target_label is a 1D tensor with batch size 1
        if target_label is not None:
            if target_label.dim() > 1:
                target_label = target_label.squeeze(-1).long()
            elif target_label.dim() == 0:
                target_label = target_label.unsqueeze(0).long()

        graph = Data(
            x=x.to(self.device),
            edge_index=self_loop_edge_index.to(self.device),
            edge_attr=self_loop_edge_attr.to(self.device) if self_loop_edge_attr is not None else None,
            edge_label=self_loop_edge_label.to(self.device) if self_loop_edge_label is not None else None,
            y=target_label.to(self.device) if target_label is not None else None,
            batch=torch.zeros(x.size(0), dtype=torch.long, device=self.device) if x.shape[0] is not None else None
        )

        # Initialize and set masks
        self.__clear_masks__()
        self.__set_masks__(x, self_loop_edge_index)

        # Generate edge mask
        edge_mask = self.gnn_explainer_alg(graph, graph.y).detach()

        if self._symmetric_edge_mask_indirect_graph:
            edge_mask = symmetric_edge_mask_indirect_graph(self_loop_edge_index, edge_mask)

        # Generate hard edge mask
        threshold = edge_mask.mean()
        hard_edge_mask = (edge_mask > threshold).float().detach()

        self.__clear_masks__()
        return edge_mask, hard_edge_mask, self_loop_edge_index, self_loop_edge_attr, self_loop_edge_label

    def __repr__(self):
        return f'{self.__class__.__name__}(explain_graph={self.explain_graph})'


def generate_subgraph(graph, hard_edge_mask: Tensor, edge_index: Tensor, edge_attr: Tensor = None,
                      edge_label: Tensor = None):
    """Generate subgraph based on edge mask."""
    device = graph.x.device

    selected_edges = hard_edge_mask.bool()
    sub_edge_index = edge_index[:, selected_edges]

    sub_edge_attr = edge_attr[selected_edges] if edge_attr is not None else None
    sub_edge_label = edge_label[selected_edges] if edge_label is not None else None

    subset = torch.unique(sub_edge_index)
    if subset.numel() == 0:
        return None  # Empty subgraph

    sub_edge_index, sub_edge_attr = subgraph(
        subset, sub_edge_index, edge_attr=sub_edge_attr, relabel_nodes=True, num_nodes=graph.num_nodes
    )

    sub_x = graph.x[subset]
    sub_y = graph.y.squeeze(-1).long() if graph.y.dim() > 1 else graph.y.long()  # Ensure y is 1D

    sub_graph = Data(
        x=sub_x.to(device),
        edge_index=sub_edge_index.to(device),
        edge_attr=sub_edge_attr.to(device) if sub_edge_attr is not None else None,
        edge_label=sub_edge_label.to(device) if sub_edge_label is not None else None,
        y=sub_y.to(device),
        num_nodes=subset.size(0),
        batch=torch.zeros(sub_x.size(0), dtype=torch.long, device=device)
    )

    return sub_graph


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
        _, node_embed = classifier.gnn(graph, emb=True)
        pred_orig = classifier(graph)
        pred_orig_prob = F.softmax(pred_orig, dim=-1)
        pred_orig_label = pred_orig.argmax(dim=-1)

        # Generate explanation
        edge_mask, hard_edge_mask, edge_index, edge_attr, edge_label = explainer(
            graph.x, graph.edge_index, target_label=graph.y, g=graph
        )

        # Generate subgraph
        # masked_data = generate_subgraph(graph, hard_edge_mask, edge_index, edge_attr, edge_label)
        masked_data = mask_fn_edgemask(graph, hard_edge_mask, isFidelitPlus=True)  # 需实现：用掩码过滤节点/边

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
        _, node_embed = classifier.gnn(graph, emb=True)
        pred_orig = classifier(graph)
        pred_orig_prob = F.softmax(pred_orig, dim=-1)
        pred_orig_label = pred_orig.argmax(dim=-1)

        # Generate explanation
        edge_mask, hard_edge_mask, edge_index, edge_attr, edge_label = explainer(
            graph.x, graph.edge_index, target_label=graph.y, g=graph
        )

        # Generate subgraph
        # masked_data = generate_subgraph(graph, hard_edge_mask, edge_index, edge_attr, edge_label)
        masked_data = mask_fn_edgemask(graph, hard_edge_mask, isFidelitPlus=False)  # 需实现：用掩码过滤节点/边

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


def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    """Custom cross-entropy loss that handles tensor shapes."""
    if y_true.dim() > 1:
        y_true = y_true.squeeze(-1).long()  # Squeeze only the last dimension if needed
    elif y_true.dim() == 0:
        y_true = y_true.unsqueeze(0).long()  # Add batch dimension if scalar
    return cross_entropy(y_pred, y_true, **kwargs)


if __name__ == "__main__":
    data_name = 'ogb'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    dataset = torch.load('./data/ogb/ogb_graph.pt', weights_only=False)
    train_dataset = torch.load("./data/ogb/train_dataset_balanced.pt", weights_only=False)
    valid_dataset = torch.load("./data/ogb/valid_dataset_balanced.pt", weights_only=False)

    split_idx = dataset.get_idx_split()
    # train_dataset = dataset[split_idx['train']]
    # valid_dataset = dataset[split_idx['valid']]
    test_dataset = dataset[split_idx['test']]

    node_in_dim = train_dataset[0].x.shape[1]
    all_labels = [data.y.item() for data in train_dataset]
    num_classes = len(set(all_labels))
    print(f"Number of classes: {num_classes}")
    print(train_dataset[0])

    # Initialize and load classifier
    classifier = trainClassifier_ogb.GNNClassifier(
        num_layer=3,
        emb_dim=node_in_dim,
        hidden_dim=32,
        num_tasks=num_classes
    )
    classifier_path = './best_gnnclassifier/best_gnn_classifier_' + data_name + '.pt'
    classifier.load_state_dict(torch.load(classifier_path, weights_only=True))
    classifier.to(device)

    # Initialize explainer
    explainer = GNNExplainer(classifier)

    # Test explanation on train dataset
    print("Testing graph classification explanation...")
    true_labels = []
    predicted_labels = []
    skipped_graphs = 0

    for graph in train_dataset:
        try:
            graph = graph.to(device)
            # Ensure graph.y is a 1D tensor with batch size 1
            if graph.y.dim() > 1:
                graph.y = graph.y.squeeze(-1).long()
            elif graph.y.dim() == 0:
                graph.y = graph.y.unsqueeze(0).long()

            # Generate explanation
            edge_mask, hard_edge_mask, edge_index, edge_attr, edge_label = explainer(
                graph.x, graph.edge_index, target_label=graph.y, g=graph
            )
            # print("edge_mask:", edge_mask)

            # Generate subgraph
            masked_data = generate_subgraph(graph, hard_edge_mask, edge_index, edge_attr, edge_label)

            if masked_data is None or masked_data.num_nodes == 0 or masked_data.edge_index.size(1) == 0:
                print(f"Warning: Empty subgraph for graph {graph}, skipping...")
                skipped_graphs += 1
                continue

            logits = classifier(masked_data)
            pred_prob = F.softmax(logits, dim=-1)  # Use softmax for multi-class
            predicted_label = torch.argmax(pred_prob).item()
            true_label = graph.y.item()

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

        except Exception as e:
            print(f"Error processing graph: {e}")
            skipped_graphs += 1
            continue

    print(f"Skipped graphs: {skipped_graphs}")

    # Compute metrics
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
