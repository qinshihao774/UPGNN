import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops
import warnings

warnings.filterwarnings("ignore")

# 全局禁用 Inspector（防止任何 explain_message 触发）
try:
    from torch_geometric.inspector import Inspector

    Inspector.enable = False
    print("Inspector 已全局禁用")
except:
    pass


class GNNExplainerManual:
    """手动实现的 GNNExplainer，100% 兼容你的自定义 GNN，不触发任何 Inspector"""

    def __init__(self, model, epochs=300, lr=0.01, device=None):
        self.model = model.eval()
        self.epochs = epochs
        self.lr = lr
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def topk_edgemask_subgraph(self, data, edge_mask, rate=0.7, isFidelitPlus=False):
        num_edges = data.edge_index.size(1)
        k = max(1, int(rate * num_edges))
        threshold = edge_mask.topk(k).values.min()
        hard_mask = (edge_mask >= threshold).float()
        final_mask = hard_mask if not isFidelitPlus else (1 - hard_mask)
        masked_data = data.clone()
        masked_data.edge_weight = final_mask
        return masked_data

    def explain(self, data: Data):
        data = data.to(self.device)
        self.model.eval()

        # 添加自环
        edge_index, _ = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)
        num_edges_with_loop = edge_index.size(1)

        # 初始化边掩码（可学习）
        edge_mask = torch.nn.Parameter(torch.ones(num_edges_with_loop, device=self.device) * 0.5)

        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()

            # 前向（用 masked edge_weight）
            masked_data = data.clone()
            masked_data.edge_index = edge_index
            masked_data.edge_weight = torch.sigmoid(edge_mask)

            # with torch.no_grad():
            pred = self.model(masked_data)

            # 损失：预测分布 + 大小正则 + 熵正则
            # pred_prob = F.softmax(pred, dim=-1)
            # loss_pred = -pred_prob[:, data.y.item()].mean()
            loss_pred = F.cross_entropy(pred, data.y)  # 替换你手写的

            m = torch.sigmoid(edge_mask)
            loss_size = 0.01 * m.mean()
            loss_ent = -m * torch.log(m + 1e-8) - (1 - m) * torch.log(1 - m + 1e-8)
            loss_ent = 0.01 * loss_ent.mean()

            loss = loss_pred + loss_size + loss_ent
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")

        final_mask = torch.sigmoid(edge_mask)[:data.edge_index.size(1)]  # 去掉自环
        return final_mask.detach()


# ==================== Fidelity 计算 ====================
# def compute_fidelity_plus(classifier, explainer, dataset, device):
#     classifier.eval()
#     total, cnt = 0.0, 0
#     for graph in dataset:
#         graph = graph.to(device)
#         if graph.num_nodes <= 1 or graph.edge_index.size(1) == 0:
#             continue
#         try:
#             with torch.no_grad():
#                 pred_orig = classifier(graph)
#                 if pred_orig.argmax(-1).item() != graph.y.item():
#                     continue
#             mask = explainer.explain(graph)
#             sub = explainer.topk_edgemask_subgraph(graph, mask, isFidelitPlus=True)
#             if sub.edge_index.size(1) == 0:
#                 continue
#             with torch.no_grad():
#                 pred_sub = classifier(sub)
#             score = torch.norm(F.softmax(pred_orig, -1) - F.softmax(pred_sub, -1), p=2).item()
#             total += score
#             cnt += 1
#         except:
#             continue
#     return total / max(cnt, 1)
#
#
# def compute_fidelity_minus(classifier, explainer, dataset, device):
#     classifier.eval()
#     total, cnt = 0.0, 0
#     for graph in dataset:
#         graph = graph.to(device)
#         if graph.num_nodes <= 1 or graph.edge_index.size(1) == 0:
#             continue
#         try:
#             with torch.no_grad():
#                 pred_orig = classifier(graph)
#                 if pred_orig.argmax(-1).item() != graph.y.item():
#                     continue
#             mask = explainer.explain(graph)
#             sub = explainer.topk_edgemask_subgraph(graph, mask, isFidelitPlus=False)
#             if sub.edge_index.size(1) == 0:
#                 continue
#             with torch.no_grad():
#                 pred_sub = classifier(sub)
#             score = torch.norm(F.softmax(pred_orig, -1) - F.softmax(pred_sub, -1), p=2).item()
#             total += score
#             cnt += 1
#         except:
#             continue
#     return total / max(cnt, 1)


# ==================== 主程序 ====================
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
    # explainer.eval()

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
        edge_mask = explainer.explain(graph)
        # edge_mask = torch.clamp(edge_mask, min=0.01, max=0.99)
        # masked_data = fn_softedgemask(graph, edge_mask, isFidelitPlus=True)  # 需实现：用掩码过滤节点/边
        masked_data = explainer.topk_edgemask_subgraph(graph, edge_mask, isFidelitPlus=True)  # 需实现：用掩码过滤节点/边

        if masked_data.num_nodes == 0 or masked_data.edge_index.size(1) == 0:
            continue  # 跳过空子图

        with torch.no_grad():
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
    # explainer.eval()

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
        edge_mask = explainer.explain(graph)
        # edge_mask = torch.clamp(edge_mask, min=0.01, max=0.99)
        # masked_data = fn_softedgemask(graph, edge_mask, isFidelitPlus=False)  # 需实现：用掩码过滤节点/边
        masked_data = explainer.topk_edgemask_subgraph(graph, edge_mask, isFidelitPlus=False)  # 需实现：用掩码过滤节点/边

        if masked_data.num_nodes == 0 or masked_data.edge_index.size(1) == 0:
            continue

        with torch.no_grad():
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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from utils import select_func


    dataset_name = 'ogb'
    classifier, train_dataset, valid_dataset, test_dataset = select_func(dataset_name, device)
    # print(f"开始解释 {dataset_name} 数据集...")
    explainer = GNNExplainerManual(model=classifier, epochs=50, lr=0.01, device=device)
    # # 测试前10个图是否能解释成功
    # success = 0
    # for i, g in enumerate(test_dataset[:10]):
    #     try:
    #         print(f"\n解释第 {i} 个图...")
    #         mask = explainer.explain(g)
    #         print(f"Success: 边掩码 shape = {mask.shape}")
    #         success += 1
    #     except Exception as e:
    #         print(f"Failed: 图 {i} 失败: {e}")
    #
    # print(f"\n成功解释 {success}/10 个图")

    # Fidelity
    print("\n计算 Fidelity...")
    fp = compute_fidelity_plus(classifier, explainer, test_dataset, device)
    fm = compute_fidelity_minus(classifier, explainer, test_dataset, device)
    print(f"\nFidelity+: {fp:.4f}")
    print(f"Fidelity-: {fm:.4f}")
