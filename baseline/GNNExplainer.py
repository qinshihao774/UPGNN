from dig.xgraph.method.base_explainer import ExplainerBase
from dig.xgraph.method.utils import symmetric_edge_mask_indirect_graph

from upsegnn import trainClassifier_ogb
from upsegnn.dataset.mutag import Mutag
from upsegnn.trainclassifier import trainClassifier_mutag
from sklearn.metrics import accuracy_score, confusion_matrix
from upsegnn.model import mask_fn_edgemask

import torch
from torch import Tensor
from torch_geometric.utils import subgraph, add_remaining_self_loops
from dig.version import debug
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
from typing import Union
from torch_geometric.data import Data

from upsegnn.utils.datasetutils import load_data

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
                 epochs: int = 300,
                 lr: float = 0.002,
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
        # print(data.x)
        # print(data.edge_label)
        if data is None:
            raise ValueError("Must provide 'data' keyword argument with the graph data object.")  # 防御性检查

        # edge_label = data.edge_label if hasattr(data, 'edge_label') else None  # 安全访问
        edge_label = getattr(data, 'edge_label', None)  # 使用 getattr 安全访问
        # edge_label = data._storage.get('edge_label', None)
        # edge_label = []

        edge_attr = getattr(data, 'edge_attr', None)
        # edge_attr = []
        #
        # for edge in edge_index.t():
        #     src, dst = edge
        #     # 节点特征的差值
        #     edge_feature = data.x[src] - data.x[dst]
        #     edge_attr.append(edge_feature)
        #
        # data.edge_attr = torch.stack(edge_attr)

        # Add self-loops and move data to device
        self_loop_edge_index, self_loop_edge_attr = add_remaining_self_loops(
            edge_index,edge_attr=edge_attr, num_nodes=x.shape[0]
        )

        if edge_label is not None:
            num_self_loops = self_loop_edge_index.size(1) - edge_index.size(1)
            self_loop_edge_label = torch.cat(
                [edge_label, torch.zeros(num_self_loops, dtype=edge_label.dtype, device=self.device)])
        else:
            self_loop_edge_label = None

        #
        # graph = type('', (), {})()
        # graph.x = x.to(self.device)
        # graph.edge_index = self_loop_edge_index.to(self.device)
        # graph.edge_attr = self_loop_edge_attr.to(self.device) if self_loop_edge_attr is not None else None
        # graph.edge_label = self_loop_edge_label.to(
        #     self.device) if self_loop_edge_label is not None else None  # 添加 edge_label
        # graph.y = target_label.to(self.device) if target_label is not None else None
        # graph.batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device) if x.shape[0] is not None else None

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

        if self._symmetric_edge_mask_indirect_graph:  # False
            edge_mask = symmetric_edge_mask_indirect_graph(self_loop_edge_index, edge_mask)

        # Generate hard edge mask
        threshold = edge_mask.mean()  # Adjusted for better subgraph selection
        hard_edge_mask = (edge_mask > threshold).float().detach()

        self.__clear_masks__()
        return edge_mask, hard_edge_mask, self_loop_edge_index, self_loop_edge_attr, self_loop_edge_label

        # 用于（量化边贡献）、（二值掩码（0 或 1），1 表示重要边）、用于节点信息传递的边索引和属性
        # return edge_mask, hard_edge_mask, self_loop_edge_index, self_loop_edge_attr

    def __repr__(self):
        return f'{self.__class__.__name__}(explain_graph={self.explain_graph})'


def generate_subgraph(graph, hard_edge_mask: Tensor, edge_index: Tensor, edge_attr: Tensor = None,
                      edge_label: Tensor = None,
                      ):
    """Generate subgraph based on edge mask."""
    device = graph.x.device
    hard_edge_mask = hard_edge_mask

    # 根据硬边掩码筛选边
    selected_edges = hard_edge_mask.bool()
    sub_edge_index = edge_index[:, selected_edges]

    # 筛选对应的边属性
    sub_edge_attr = edge_attr[selected_edges] if edge_attr is not None else None
    # 筛选对应的边标签
    sub_edge_label = edge_label[selected_edges] if edge_label is not None else None
    # sub_edge_label = graph.edge_label[selected_edges] if hasattr(graph, 'edge_label') else None

    # 提取子图中的节点
    subset = torch.unique(sub_edge_index)
    if subset.numel() == 0:
        return None  # 空子图

    # 使用 torch_geometric.utils.subgraph 生成子图
    sub_edge_index, sub_edge_attr = subgraph(
        subset, sub_edge_index, edge_attr=sub_edge_attr, relabel_nodes=True, num_nodes=graph.num_nodes
    )

    # 对于 edge_label，也需类似处理（subgraph 不直接支持 edge_label）
    # if sub_edge_label is not None:
    # 手动 relabel edge_label 的索引（但由于 relabel_nodes=True，edge_label 可能不需要 relabel，因为它是边标签而非索引）
    # sub_edge_label = sub_edge_label[selected_edges]  # 简化，假设不需 relabel

    sub_x = graph.x[subset]
    sub_y = graph.y  # 图级标签保持不变

    # 创建 PyG Data 对象
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


# def evaluate_fidelity_plus(explainer, classifier, dataset, device):
#     """
#     计算 Fidelity+ 指标，衡量图解释器识别关键子结构对模型预测的影响程度
#     参数:
#         explainer: 包含解释器（explainer）和编码器（embed_model）的模型
#         classifier: 目标分类器，输入图输出预测结果
#         dataset: PyG Data 对象列表（每个元素是图数据）
#         device: torch.device，指定计算设备
#     返回:
#         avg_fidelity: 平均 Fidelity+ 分数
#     """
#     # 模型、分类器置为评估模式
#     # model.eval()
#     # model.to(device)
#     classifier.eval()
#     classifier.to(device)
#
#     # total_loss = 0.0  # 累计所有图的损失
#     # num_graphs = 0  # 实际参与计算的有效图数量
#
#     num_graphs = 0  # 实际参与计算的有效图数量
#     total_fidelity = 0.0
#
#     for graph in dataset:
#         graph = graph.to(device)
#         # 跳过空图（无节点或无边，无法计算）
#         if graph.num_nodes == 0 or graph.edge_index.size(1) == 0:
#             print("Warning: 跳过空图，无节点或无边")
#             continue
#
#         # with torch.no_grad():
#         # 1. 原始图预测：f(G)
#         # 若分类器需要节点嵌入，先通过编码器获取；若直接用图数据，可简化
#         _, node_embed = classifier.gnn(graph, emb=True)  # 获取节点嵌入
#         pred_orig = classifier(graph)  # 原始图预测结果
#         pred_orig_prob = F.softmax(pred_orig, dim=-1)
#         pred_orig_label = pred_orig.argmax(dim=-1)
#
#         # 2. 获取关键子结构掩码（节点/边掩码）：识别 S_i
#         # node_mask, _ = mlp_explainer.get_explain(data=graph, embeds=node_embed, istrain=False)
#         # Generate explanation
#         edge_mask, hard_edge_mask, edge_index, edge_attr, edge_label = explainer(graph.x, graph.edge_index,
#                                                                                  target_label=graph.y,
#                                                                                  g=graph)
#
#         # print(f"edge_mask={edge_mask}, hard_edge_mask={hard_edge_mask}")
#
#         # Generate subgraph
#         masked_data = generate_subgraph(graph, edge_mask, edge_index, edge_attr, edge_label)
#
#         # 3. 生成移除关键子结构的子图：G \ S_i
#         # masked_data = mask_fn(graph, node_mask)  # mask_fn 需实现：用掩码过滤节点/边
#         # masked_data = masked_data.to(device)  # 确保子图在目标设备
#
#         # 处理掩码后空图（移除关键子结构后无有效节点，跳过或特殊处理）
#         if masked_data.num_nodes == 0:
#             print("Warning: 掩码后子图为空，跳过该图计算")
#             continue
#
#         # 4. 子图预测：f(G \ S_i)
#         pred_masked = classifier(masked_data)
#         pred_masked_prob = F.softmax(pred_masked, dim=-1)
#
#         # 计算单图 F+ 得分
#         if pred_orig_label == graph.y:
#             fidelity_score = torch.norm(pred_orig_prob - pred_masked_prob, p=2).item()
#         else:
#             fidelity_score = 0.0
#         total_fidelity += fidelity_score
#         num_graphs += 1
#
#     # 计算平均 Fidelity+（避免除以 0，用 max 保证分母至少为 1）
#     avg_fidelity = total_fidelity / max(num_graphs, 1)
#
#     print(f"平均 Fidelity+: {avg_fidelity:.4f}")


def evaluate_fidelity_plus(explainer, classifier, dataset, device):
    """
    计算 Fidelity+ 指标，衡量图解释器识别关键子结构对模型预测的影响程度
    参数:
        model: 包含解释器（explainer）和编码器（embed_model）的模型
        classifier: 目标分类器，输入图输出预测结果
        dataset: PyG Data 对象列表（每个元素是图数据）
        device: torch.device，指定计算设备
        loss_fn: 损失函数，默认用 MSE，可替换为交叉熵等（需匹配任务）
    返回:
        avg_fidelity: 平均 Fidelity+ 分数
    """
    # 模型、分类器置为评估模式
    explainer.eval()
    explainer.to(device)
    classifier.eval()
    classifier.to(device)

    # total_loss = 0.0  # 累计所有图的损失
    num_graphs = 0  # 实际参与计算的有效图数量
    total_fidelity = 0.0

    for graph in dataset:
        graph = graph.to(device)
        # 跳过空图（无节点或无边，无法计算）
        if graph.num_nodes == 0 or graph.edge_index.size(1) == 0:
            print("Warning: 跳过空图，无节点或无边")
            continue

        # 主要作用是临时禁用梯度计算
        # 核心功能是告诉 PyTorch 在其包裹的代码块中不要为张量操作构建计算图，也不要计算或存储梯度
        # 效果：减少内存消耗（因为不保存中间梯度），加速计算（因为不执行梯度相关的操作）。
        # with torch.no_grad():

        # 1. 原始图预测：f(G)
        # 若分类器需要节点嵌入，先通过编码器获取；若直接用图数据，可简化
        _, node_embed = classifier.gnn(graph, emb=True)  # 获取节点嵌入
        pred_orig = classifier(graph)  # 原始图预测结果 不需要梯度，因此使用 torch.no_grad() 提高效率。
        pred_orig_prob = F.softmax(pred_orig, dim=-1)
        pred_orig_label = pred_orig.argmax(dim=-1)

        # 2. 获取关键子结构掩码（节点/边掩码）：识别 S_i
        _, hard_edge_mask, _, _, _ = explainer(graph.x, graph.edge_index,
                                               target_label=graph.y,
                                               g=graph)
        # node_mask, _ = explainer(data=graph, embeds=node_embed, istrain=False)
        # print("node_mask:", node_mask)

        # 3. 生成移除关键子结构的子图：G \ S_i
        masked_data = mask_fn_edgemask(graph, hard_edge_mask, isFidelitPlus=True)  # 需实现：用掩码过滤节点/边

        # 4. 子图预测：f(G \ S_i)
        pred_masked = classifier(masked_data)
        pred_masked_prob = F.softmax(pred_masked, dim=-1)

        # 计算单图 F+ 得分
        if pred_orig_label == graph.y:
            fidelity_score = torch.norm(pred_orig_prob - pred_masked_prob, p=2).item()
        else:
            fidelity_score = 0.0
        total_fidelity += fidelity_score
        num_graphs += 1

    # 计算平均 Fidelity+（避免除以 0，用 max 保证分母至少为 1）
    avg_fidelity = total_fidelity / max(num_graphs, 1)
    print(f"平均 Fidelity+: {avg_fidelity:.4f}")
    return avg_fidelity


def evaluate_fidelity_minus(explainer, classifier, dataset, device):
    """
    计算 Fidelity+ 指标，衡量图解释器识别关键子结构对模型预测的影响程度
    参数:
        model: 包含解释器（explainer）和编码器（embed_model）的模型
        classifier: 目标分类器，输入图输出预测结果
        dataset: PyG Data 对象列表（每个元素是图数据）
        device: torch.device，指定计算设备
        loss_fn: 损失函数，默认用 MSE，可替换为交叉熵等（需匹配任务）
    返回:
        avg_fidelity: 平均 Fidelity+ 分数
    """
    # 模型、分类器置为评估模式
    explainer.eval()
    explainer.to(device)
    classifier.eval()
    classifier.to(device)

    # total_loss = 0.0  # 累计所有图的损失
    num_graphs = 0  # 实际参与计算的有效图数量
    total_fidelity = 0.0

    for graph in dataset:
        graph = graph.to(device)
        # 跳过空图（无节点或无边，无法计算）
        if graph.num_nodes == 0 or graph.edge_index.size(1) == 0:
            print("Warning: 跳过空图，无节点或无边")
            continue

        # 主要作用是临时禁用梯度计算
        # 核心功能是告诉 PyTorch 在其包裹的代码块中不要为张量操作构建计算图，也不要计算或存储梯度
        # 效果：减少内存消耗（因为不保存中间梯度），加速计算（因为不执行梯度相关的操作）。
        # with torch.no_grad():

        # 1. 原始图预测：f(G)
        # 若分类器需要节点嵌入，先通过编码器获取；若直接用图数据，可简化
        # _, node_embed = classifier.gnn(graph, emb=True)  # 获取节点嵌入
        pred_orig = classifier(graph)  # 原始图预测结果 不需要梯度，因此使用 torch.no_grad() 提高效率。
        pred_orig_prob = F.softmax(pred_orig, dim=-1)
        pred_orig_label = pred_orig.argmax(dim=-1)

        # 2. 获取关键子结构掩码（节点/边掩码）：识别 S_i
        _, hard_edge_mask, _, _, _ = explainer(graph.x, graph.edge_index,
                                               target_label=graph.y,
                                               g=graph)
        # node_mask, _ = explainer(data=graph, embeds=node_embed, istrain=False)
        # print("node_mask:", node_mask)

        # 3. 生成移除关键子结构的子图：G \ S_i
        masked_data = mask_fn_edgemask(graph, hard_edge_mask, isFidelitPlus=False)  # 需实现：用掩码过滤节点/边

        # 4. 子图预测：f(G \ S_i)
        pred_masked = classifier(masked_data)
        pred_masked_prob = F.softmax(pred_masked, dim=-1)

        # 计算单图 F+ 得分
        if pred_orig_label == graph.y:
            fidelity_score = torch.norm(pred_orig_prob - pred_masked_prob, p=2).item()
        else:
            fidelity_score = 0.0
        total_fidelity += fidelity_score
        num_graphs += 1

    # 计算平均 Fidelity+（避免除以 0，用 max 保证分母至少为 1）
    avg_fidelity = total_fidelity / max(num_graphs, 1)
    print(f"平均 Fidelity-: {avg_fidelity:.4f}")
    return avg_fidelity

def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    if y_true.dim() == 0:  # 如果是标量
        y_true = y_true.unsqueeze(0)  # 转换为形状[1]
    return cross_entropy(y_pred, y_true.long(), **kwargs)


def map_labels(dataset):
    """Mutag Map -1 labels to 0."""
    new_dataset = []
    for data in dataset:
        data.y = torch.where(data.y == -1, torch.tensor(0, dtype=data.y.dtype), data.y).squeeze()
        new_dataset.append(data)
    return new_dataset


if __name__ == "__main__":

    data_name = 'mutag'
    # train_dataset, valid_dataset, test_dataset = load_data(data_name)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO:Load mutag datasets
    train_dataset = Mutag(root='../upsegnn/data/mutag', split='train')
    valid_dataset = Mutag(root='../upsegnn/data/mutag', split='val')
    test_dataset = Mutag(root='../upsegnn/data/mutag', split='test')
    train_dataset = map_labels(train_dataset)
    valid_dataset = map_labels(valid_dataset)
    test_dataset = map_labels(test_dataset)

    node_in_dim = train_dataset[0].x.shape[1]
    all_labels = [data.y.item() for data in train_dataset]
    num_classes = len(set(all_labels))
    print(f"Number of classes: {num_classes}")
    # 打印其中一个 数据
    print(train_dataset[0])

    # Initialize and load model
    classifier = trainClassifier_mutag.GNNClassifier(
        num_layer=3,
        emb_dim=node_in_dim,
        hidden_dim=128,
        num_tasks=num_classes
    )
    #
    # classifier = trainClassifier_obg.GNNClassifier(
    #     num_layer=3,
    #     emb_dim=node_in_dim,
    #     hidden_dim=32,
    #     num_tasks=num_classes
    # )

    Classifier_path = '../upsegnn/best_gnnclassifier/best_gnn_classifier_' + data_name + '.pt'
    classifier.load_state_dict(torch.load(Classifier_path, weights_only=True))
    classifier.to(device)

    # Initialize explainer
    explainer = GNNExplainer(classifier)

    # Test explanation on train dataset
    print("Testing graph classification explanation...")
    true_labels = []
    predicted_labels = []
    skipped_graphs = 0
    # for epoch in range(100):
    for graph in train_dataset:
        try:
            # Move graph to device
            graph = graph.to(device)
            graph.y = graph.y.squeeze().float()
            # Generate explanation
            edge_mask, hard_edge_mask, edge_index, edge_attr, edge_label = explainer(graph.x, graph.edge_index,
                                                                                     target_label=graph.y, g=graph)

            # print(f"edge_mask={edge_mask}, hard_edge_mask={hard_edge_mask}")

            # Generate subgraph
            # masked_data = generate_subgraph(graph, edge_mask, edge_index, edge_attr)
            masked_data = generate_subgraph(graph, hard_edge_mask, edge_index, edge_attr)

            # Evaluate with masked graph
            # with torch.no_grad():

            if masked_data.num_nodes == 0 or masked_data.edge_index.size(1) == 0:
                print(f"Warning: Empty subgraph for graph {graph}, skipping...")
                skipped_graphs += 1
                continue

            logits = classifier(masked_data)
            pred_prob = torch.sigmoid(logits).squeeze()

            true_label = int(graph.y.item())
            predicted_label = torch.argmax(pred_prob).item()

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
