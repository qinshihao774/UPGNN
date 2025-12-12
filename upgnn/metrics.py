import random
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx


def evaluate_embeddings(model, dataset, device):
    """
    使用逻辑回归评估整个数据集的嵌入，计算 AUC 分数，以衡量编码器性能。
    参数:
        embed_model: 生成嵌入的 GNN 模型
        dataset: Data 对象列表（例如 valid_dataset）
        device: torch.device
    """
    embed_model = model.embed_model
    embed_model.eval()
    X, y = [], []

    for graph in dataset:
        # print(f"Graph.y: {graph.y}")

        graph = graph.to(device)
        with torch.no_grad():
            embeds = embed_model(graph)
        X.append(embeds.cpu().numpy())
        # 确保每个标签都是一维数组
        y.append(np.expand_dims(graph.y.cpu().numpy(), axis=0))

    X = np.concatenate(X)
    y = np.concatenate(y).ravel()

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"警告：数据集中仅有一个类别 ({unique_classes[0]})，无法计算 AUC。")
        return

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)
    print(f"线性评估 AUC: {auc:.4f}")


def evaluate_single_graph(classifier, explainer, data, device):
    classifier.eval()
    explainer.eval()
    data = data.to(device)
    with torch.no_grad():
        embed, node_embed = classifier.gnn(data, isbatch=False)
        edge_mask = explainer.explainer(data=data, node_embed=node_embed)
        # masked_data_minus = explainer.fn_softedgemask(data, edge_mask, isFidelitPlus=False)
        # masked_data_minus = explainer.topk_edge_mask(data, edge_mask, rate=0.8, isFidelitPlus=False)
        masked_data_minus = explainer.hard_subgraph_mask(data, edge_mask, rate=0.8, keep_important=True)
        logists = classifier(masked_data_minus)
        pred_prob = torch.sigmoid(logists).squeeze()  # 转换为概率，形状 [2]
        true_label = data.y.item()
        predicted_label = torch.argmax(pred_prob).item()

        return true_label, predicted_label


def calculate_sparsity(classifier, explainer, dataset, device, is_dataloader=False):
    """
    计算验证集中解释子图相对于原图的稀疏度。

    参数：
        dataset: 数据集，可以是 DataLoader 或 Data 对象的列表。
        is_dataloader: 如果为 True，则 validation_set 是 DataLoader；否则为列表。

    返回：
        sparsities: 每个图的稀疏度列表。
        average_sparsity: 平均稀疏度。

    假设：
        - 每个图数据 (Data) 有属性：
          - original_edge_index: 原图的边索引，形状 [2, num_original_edges]
          - subgraph_edge_index: 解释子图的边索引，形状 [2, num_subgraph_edges]
        - 稀疏度定义为：1 - (子图边数 / 原图边数)，值越大表示越稀疏。
        - 假设无向图，边数 = edge_index.size(1) / 2
    """
    explainer.eval()
    # explainer.to(device)
    classifier.eval()
    # classifier.to(device)

    sparsities = []

    if is_dataloader:
        # 如果是 DataLoader，遍历批次（假设 batch_size=1 以简化；否则需处理批次）
        for batch in dataset:
            batch = batch.to(device)
            for data in batch:  # 如果 batch_size >1，需要解批
                _, node_embed = classifier.gnn(data, isbatch=False)
                edge_mask = explainer.explainer(data=data, node_embed=node_embed)

                top_k = 10  # Adjustable number of top edges to select
                _, top_indices = torch.topk(edge_mask, k=min(top_k, edge_mask.size(0)), largest=True)
                selected_edges = torch.zeros_like(edge_mask, dtype=torch.bool)
                selected_edges[top_indices] = True
                subgraph_edge_index = data.edge_index[:, selected_edges]

                # Compute number of edges (undirected graph: divide by 2)
                original_num_edges = data.edge_index.size(1) / 2
                subgraph_num_edges = subgraph_edge_index.size(1) / 2

                if original_num_edges == 0:
                    sparsity = 0.0  # 避免除零
                else:
                    sparsity = 1 - (subgraph_num_edges / original_num_edges)
                sparsities.append(sparsity)
    else:
        # 如果是列表，直接遍历
        for data in dataset:
            data = data.to(device)
            with torch.no_grad():
                _, node_embed = classifier.gnn(data, isbatch=False)
                edge_mask = explainer.explainer(data=data, node_embed=node_embed)

            top_k = 10  # Adjustable number of top edges to select
            _, top_indices = torch.topk(edge_mask, k=min(top_k, edge_mask.size(0)), largest=True)
            selected_edges = torch.zeros_like(edge_mask, dtype=torch.bool)
            selected_edges[top_indices] = True
            subgraph_edge_index = data.edge_index[:, selected_edges]

            # Compute number of edges (undirected graph: divide by 2)
            original_num_edges = data.edge_index.size(1) / 2
            subgraph_num_edges = subgraph_edge_index.size(1) / 2

            if original_num_edges == 0:
                sparsity = 0.0
            else:
                sparsity = 1 - (subgraph_num_edges / original_num_edges)
            sparsities.append(sparsity)

    average_sparsity = sum(sparsities) / len(sparsities) if sparsities else 0.0

    print(f"Average Sparsity: {average_sparsity:.4f}")
    # print(f"Sparsities: {sparsities}")

    # return sparsities, average_sparsity


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
    explainer.eval()
    # classifier.to(device)
    # explainer.to(device)

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

            # 2. 生成关键子结构掩码（假设 explainer 返回 edge_mask 用于 Fidelity+）
            # _, node_embed = classifier.gnn(graph, emb=True)
            _, node_embed = classifier.gnn(graph, isbatch=False)
            edge_mask = explainer.explainer(data=graph, node_embed=node_embed)
            # print(f"Edge mask mean: {edge_mask.mean():.4f}, min: {edge_mask.min():.4f}, max: {edge_mask.max():.4f}")
            # masked_data = explainer.fn_softedgemask(graph, edge_mask, isFidelitPlus=True)  # 非关键结构
            # masked_data = explainer.topk_edge_mask(graph, edge_mask, rate=0.5, isFidelitPlus=True)  # 非关键结构
            masked_data = explainer.hard_subgraph_mask(graph, edge_mask, rate=0.7, keep_important=False)  # 非关键结构

            if masked_data.num_nodes == 0 or masked_data.edge_index.size(1) == 0:
                continue  # 跳过空子图

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
    explainer.eval()
    # classifier.to(device)
    # explainer.to(device)

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

            # 2. 生成关键子结构掩码
            # _, node_embed = classifier.gnn(graph, emb=True)
            _, node_embed = classifier.gnn(graph, isbatch=False)
            edge_mask = explainer.explainer(data=graph, node_embed=node_embed)
            # print(f"Edge mask mean: {edge_mask.mean():.4f}, min: {edge_mask.min():.4f}, max: {edge_mask.max():.4f}")
            # masked_data = explainer.fn_softedgemask(graph, edge_mask, isFidelitPlus=False)  # 关键结构
            # masked_data = explainer.topk_edge_mask(graph, edge_mask, rate=0.5, isFidelitPlus=False)  # 关键结构
            masked_data = explainer.hard_subgraph_mask(graph, edge_mask, rate=0.7, keep_important=True)  # 关键结构

            if masked_data.num_nodes == 0 or masked_data.edge_index.size(1) == 0:
                continue

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


def fidelity_plus(model, data, subgraph) -> float:
    model.eval()
    with torch.no_grad():
        orig_out = model(data.x, data.edge_index, data.batch)
        orig_prob = orig_out.softmax(1)
        true_label = orig_out.argmax(1).item()
        p_orig = orig_prob[0, true_label].item()

        # 移除解释子图（重要部分置0）
        node_idx = subgraph.node_index
        mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
        mask[node_idx] = True
        x_masked = data.x.clone()
        x_masked[~mask] = 0
        edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        edge_index_masked = data.edge_index[:, ~edge_mask]  # 保留非重要边

        batch_masked = data.batch.clone()
        batch_masked[~mask] = 0
        out_masked = model(x_masked, edge_index_masked, batch_masked)
        p_masked = out_masked.softmax(1)[0, true_label].item()

        return p_orig - p_masked  # 下降越多越好


def fidelity_minus(model, data, subgraph):
    model.eval()
    with torch.no_grad():
        orig_out = model(data.x, data.edge_index, data.batch)
        orig_prob = orig_out.softmax(1)
        true_label = data.y.item()  # 假设 graph-level 标签
        p_orig = orig_prob[0, true_label].item()

        # 只保留子图
        node_idx = subgraph.node_index
        mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
        mask[node_idx] = True

        x_sub = data.x.clone()
        x_sub[~mask] = 0
        edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        edge_index_sub = data.edge_index[:, edge_mask]
        batch_padded = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        batch_padded[torch.where(mask)[0]] = data.batch[torch.where(mask)[0]]

        out_sub = model(x_sub, edge_index_sub, batch_padded)
        p_sub = out_sub.softmax(1)[0, true_label].item() if out_sub.size(0) > 0 else 0.0

        fid_minus = p_orig - p_sub  # 标准差值，低好（接近0）
        return fid_minus, p_sub


# ==================== 4. 可视化函数 ====================
def visualize_explanation(data, subgraph, fid_plus, fid_minus, sparsity):
    G_full = to_networkx(data, to_undirected=True)
    G_sub = to_networkx(subgraph, to_undirected=True)

    plt.figure(figsize=(15, 6))

    # 原图 + 高亮
    plt.subplot(131)
    pos = nx.kamada_kawai_layout(G_full)
    nx.draw(G_full, pos, node_color='lightgray', edge_color='gray', node_size=300, with_labels=False)
    nx.draw_networkx_nodes(G_full, pos, nodelist=subgraph.node_index.tolist(), node_color='red', node_size=500)
    nx.draw_networkx_edges(G_full, pos, edgelist=[(a.item(), b.item()) for a, b in subgraph.edge_index.t()],
                           edge_color='red', width=3)
    plt.title("Original + Highlighted Subgraph")

    # 纯子图
    plt.subplot(132)
    pos_sub = nx.kamada_kawai_layout(G_sub)
    nx.draw(G_sub, pos_sub, node_color='salmon', node_size=600, edge_color='red', width=3, with_labels=True)
    plt.title("Explanation Subgraph")

    # 指标柱状图
    plt.subplot(133)
    metrics = ['Fidelity+', 'Fidelity-', 'Sparsity']
    values = [fid_plus, fid_minus, sparsity]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    plt.bar(metrics, values, color=colors, alpha=0.8)
    plt.ylim(0, 1)
    plt.title("Explanation Quality Metrics")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=12)

    plt.suptitle(f"MUTAG Graph {data.y.item()} | Nodes: {data.num_nodes} → {subgraph.num_nodes}", fontsize=16)
    plt.tight_layout()
    plt.show()
