import random
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


# from torch_geometric.explain import unfaithfulness  # 评估解释的指标
# from upsegnn.model import fn_softedgemask


# def evaluate_fidelity_plus(model, classifier, dataset, device):
#     """
#     计算 Fidelity+ 指标，衡量图解释器识别关键子结构对模型预测的影响程度
#     参数:
#         model: 包含解释器（explainer）和编码器（embed_model）的模型
#         classifier: 目标分类器，输入图输出预测结果
#         dataset: PyG Data 对象列表（每个元素是图数据）
#         device: torch.device，指定计算设备
#         loss_fn: 损失函数，默认用 MSE，可替换为交叉熵等（需匹配任务）
#     返回:
#         avg_fidelity: 平均 Fidelity+ 分数
#     """
#     # 模型、分类器置为评估模式
#     model.eval()
#     classifier.eval()
#     classifier.to(device)
#
#     # embed_model = model.embed_model.to(device)
#     embed_model = classifier.gnn.to(device)
#     mlp_explainer = model.explainer.to(device)
#
#     # total_loss = 0.0  # 累计所有图的损失
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
#         #
#         # # 原始预测
#         # with torch.no_grad():
#         #     original_pred = classifier(graph)
#         # print("原始预测结果：", original_pred)
#         # 这里简单假设是分类任务，取预测类别（你可根据实际任务调整，比如回归任务就不需要 argmax）
#         # original_pred_label = original_pred.argmax(dim=1)
#         # print("原始预测结果：", original_pred_label)
#         # 计算原始预测正确的样本数量（如果是单样本图任务，比如图分类，这里就是 1 或 0，根据实际情况灵活改）
#         # if (original_pred_label == graph.y):
#         #     original_correct += 1
#         # else:
#         # raise ValueError("图数据需要包含 'y' 属性来进行预测准确率计算")
#
#         with torch.no_grad():
#             # 1. 原始图预测：f(G)
#             # 若分类器需要节点嵌入，先通过编码器获取；若直接用图数据，可简化
#             _, node_embed = embed_model(graph, emb=True)  # 获取节点嵌入
#             pred_orig = classifier(graph)  # 原始图预测结果
#             pred_orig_prob = F.softmax(pred_orig, dim=-1)
#             pred_orig_label = pred_orig.argmax(dim=-1)
#
#             # 2. 获取关键子结构掩码（节点/边掩码）：识别 S_i
#             node_mask, _ = mlp_explainer.get_explain(data=graph, embed=node_embed, logits=pred_orig)
#             # print("node_mask:", node_mask)
#
#             # 3. 生成移除关键子结构的子图：G \ S_i
#             masked_data = mask_fn_nodemask(graph, node_mask, isFidelitPlus=True)  # mask_fn 需实现：用掩码过滤节点/边
#
#             # if masked_data.num_nodes == 0:
#             #     print("Warning: 掩码后子图为空，跳过该图计算")
#             #     continue
#
#         #         # 掩码后的预测
#         #         with torch.no_grad():
#         #             masked_pred = classifier(masked_data)
#         #             print("masked_pred:", masked_pred)
#         #             masked_pred_label = masked_pred.argmax(dim=1)
#         #
#         #             if (masked_pred_label == graph.y):
#         #                 masked_correct += 1
#         #
#         #         # 计算 F+，这里根据你给的公式思路，类似 F = 1 - Acc(X')/Acc(X)，这里实现的是这个逻辑，你可根据实际需求调整公式定义
#         #         if original_correct == 0:
#         #             # 避免除以 0，可根据实际情况处理，这里简单返回一个特殊值或抛异常，也可调整计算逻辑
#         #             raise ValueError("原始预测准确率为 0，无法计算 F+")
#         #
#         # f_plus = 1 - float(masked_correct) / float(original_correct)
#         # print(f"Fidelity+: {f_plus:.4f}")
#         # return f_plus
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
#     # return total_fidelity


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
        masked_data_minus = explainer.fn_softedgemask(data, edge_mask, isFidelitPlus=False)
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
                num_graphs += 1
                continue  # 如果预测错误，贡献为 0

            # 2. 生成关键子结构掩码（假设 explainer 返回 edge_mask 用于 Fidelity+）
            # _, node_embed = classifier.gnn(graph, emb=True)
            _, node_embed = classifier.gnn(graph, isbatch=False)
            edge_mask = explainer.explainer(data=graph, node_embed=node_embed)
            # print(f"Edge mask mean: {edge_mask.mean():.4f}, min: {edge_mask.min():.4f}, max: {edge_mask.max():.4f}")
            masked_data = explainer.fn_softedgemask(graph, edge_mask, isFidelitPlus=True)  # 非关键结构

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
                num_graphs += 1
                continue

            # 2. 生成关键子结构掩码
            # _, node_embed = classifier.gnn(graph, emb=True)
            _, node_embed = classifier.gnn(graph, isbatch=False)
            edge_mask = explainer.explainer(data=graph, node_embed=node_embed)
            # print(f"Edge mask mean: {edge_mask.mean():.4f}, min: {edge_mask.min():.4f}, max: {edge_mask.max():.4f}")
            masked_data = explainer.fn_softedgemask(graph, edge_mask, isFidelitPlus=False)  # 关键结构

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
