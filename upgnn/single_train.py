import io
import os
import numpy as np
import logging
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from model import set_seed, generate_explanation, train, Pretrain_Explainer, retune
from metrics import evaluate_single_graph, calculate_sparsity, compute_fidelity_minus, compute_fidelity_plus
from sklearn.metrics import accuracy_score, confusion_matrix
from upgnn.trainclassifier import trainClassifier_proteins, trainClassifier_nci1, trainClassifier_ba2motif, \
    trainClassifier_dd, trainClassifier_mutag, trainClassifier_mutagenicity, trainClassifier_bbbp, \
    trainClassifier_frankenstein, trainClassifier_ogb
from utils.datasetutils import load_data
from datetime import datetime

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ==================== 日志================
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = f"{log_dir}/trainlog_{datetime.now().strftime('%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # 同时打印到屏幕
    ]
)
logger = logging.getLogger()

## In[Settings]
# set_seed(42)
data_name = 'bbbp'
save_path = 'pretrained'
train_dataset, valid_dataset, test_dataset = load_data(data_name)
# train_dataset, valid_dataset, test_dataset = load_data("mutagenicity")
Classifier_path = './best_gnnclassifier/best_gnn_classifier_' + data_name + '.pt'
# Classifier_path = './best_gnnclassifier/best_gnn_classifier_mutagenicity.pt'
# pretrained_Explainer_path = './pretrained/pretrained_explainer_graph_' + data_name + '.pt'
# pretrained_Explainer_path = './pretrained/pretrained_explainer_exclude_' + data_name + '.pt'
# pretrained_Explainer_path = './pretrained/pretrained_explainer_all.pt'
# pretrained_Explainer_path = './pretrained/pretrained_explainer_exclude_mutag.pt'
# pretrained_Explainer_path = './pretrained/pretrained_explainer_graph.pt'

os.makedirs(save_path, exist_ok=True)  # exist_ok=True 创建目录时，如果目录已经存在，则不报错
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## In[Dataset]
# # 检查数据集大小
print(f"{data_name}_dataset single data:", train_dataset[0])
print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")

# 检查数据集标签
all_labels = [data.y.item() for data in train_dataset]
num_classes = len(set(all_labels))
print("num_classes:", num_classes)
num_tasks = num_classes

## In[统一打印]
node_in_dim = train_dataset[0].x.shape[1]  # 节点维度
print(f'node_dim：{node_in_dim}')
edge_in_dim = train_dataset[0].edge_index.shape[1]
print(f'edge_dim：{edge_in_dim}')

# DataLoader用于从数据集加载数据并将其组织成小批量的工具类
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# val_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# [classifier]
# ba2motif
# classifier = trainClassifier_ba2motif.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=12,
#                                                     num_tasks=num_tasks).to(device)
# mutag
# classifier = trainClassifier_mutag.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=32,
#                                                  num_tasks=num_tasks).to(device)
# dd
# classifier = trainClassifier_dd.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=32,
#                                               num_tasks=num_tasks).to(device)
# ogb
# classifier = trainClassifier_ogb.GNNClassifier(num_layer=2, emb_dim=node_in_dim, hidden_dim=32,
#                                                num_tasks=num_tasks).to(device)
# mutagenicity
# classifier = trainClassifier_mutagenicity.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=32,
#                                                         num_tasks=num_tasks).to(device)
# proteins
# classifier = trainClassifier_proteins.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=128,
#                                                     num_tasks=num_tasks).to(device)
# nci1
# classifier = trainClassifier_nci1.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=32,
#                                                 num_tasks=num_tasks).to(device)
# frankenstein
# classifier = trainClassifier_frankenstein.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=300,
#                                                num_tasks=num_tasks).to(device)
# bbbp
classifier = trainClassifier_bbbp.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=16,
                                                num_tasks=num_tasks).to(device)

classifier.load_state_dict(torch.load(Classifier_path, weights_only=True))
PE = Pretrain_Explainer(classifier, 5, 256, device, explain_graph=True, loss_type='NCE')  # 初始化解释器

# TODO: single dataset train...
num = int(len(train_dataset) * 0.1)
train_sub_dataset = Subset(train_dataset, range(num))  # 前 10%
train(PE, train_sub_dataset, valid_dataset, logger, device, epochs=5)
# torch.save(PE.explainer.state_dict(), pretrained_Explainer_path)
# PE.explainer.load_state_dict(torch.load(pretrained_Explainer_path, weights_only=True))
# PE.generate_explanation(test_dataset, device)

print("--------------done!-----------------")
with torch.no_grad():
    # # 逐图验证
    print("Evaluating fidelity...")
    avg_fidelity_plus = compute_fidelity_plus(classifier, PE, test_dataset, device)
    print(f"Average Fidelity+: {avg_fidelity_plus:.4f}")
    avg_fidelity_minus = compute_fidelity_minus(classifier, PE, test_dataset, device)
    print(f"Average Fidelity-: {avg_fidelity_minus:.4f}")



    # calculate_sparsity(classifier, PE, test_dataset, device, is_dataloader=False)
    print("--------------done!-----------------")

    # # 逐图测试嵌入
    # print("Evaluating single graph pred_prob auc...")
    # # 评估模型性能
    # true_labels_val = []
    # predicted_labels_val = []
    # for data in valid_dataset:
    #     true_label, predicted_label = evaluate_single_graph(classifier, PE, data, device)
    #     true_labels_val.append(true_label)
    #     # predicted_label = 1.0 if predicted_label == 1 else -1.0
    #     predicted_labels_val.append(predicted_label)
    #
    # # 计算准确率
    # accuracy = accuracy_score(true_labels_val, predicted_labels_val)
    # print(f"Accuracy: {accuracy:.4f}")
    #
    # # 计算混淆矩阵
    # conf_matrix = confusion_matrix(true_labels_val, predicted_labels_val)
    # print("Confusion Matrix:", conf_matrix)
    #
    # print("--------------done!-----------------")

    # # 逐图测试嵌入
    # print("Testing single graph pred_prob auc...")
    # # 评估模型性能
    # true_labels_test = []
    # predicted_labels_test = []
    # for data in test_dataset:
    #     true_label, predicted_label = evaluate_single_graph(classifier, PE, data, device)
    #     true_labels_test.append(true_label)
    #     # predicted_label = 1.0 if predicted_label == 1 else -1.0  #mutag
    #     predicted_labels_test.append(predicted_label)
    #
    # # 计算准确率
    # accuracy = accuracy_score(true_labels_test, predicted_labels_test)
    #
    # print(f"Accuracy: {accuracy:.4f}")
    #
    # # 计算混淆矩阵
    # conf_matrix = confusion_matrix(true_labels_test, predicted_labels_test)
    # print("Confusion Matrix:", conf_matrix)

# Confusion Matrix:
# [[3405  578]   ->   [TP FN]
# [ 102   28]]   ->   [FP TN]

# print("Retuning  ... ")
# # train_sub_dataset = Subset(train_dataset, range(100))  # 前 100 个
# retune(PE, train_sub_dataset, device)
# # retune(PE, train_dataset, device)
# PE.generate_explanation(test_dataset, device)
# with torch.no_grad():
#     # # 逐图验证
#     print("Evaluating fidelity...")
#     avg_fidelity_plus = compute_fidelity_plus(classifier, PE, test_dataset, device)
#     print(f"Average Fidelity+: {avg_fidelity_plus:.4f}")
#     avg_fidelity_minus = compute_fidelity_minus(classifier, PE, test_dataset, device)
#     print(f"Average Fidelity-: {avg_fidelity_minus:.4f}")
#     # calculate_sparsity(classifier, PE, test_dataset, device, is_dataloader=False)
#     print("--------------done!-----------------")
#
#     # 逐图测试嵌入
#     print("Testing single graph pred_prob auc...")
#     # 评估模型性能
#     true_labels_test = []
#     predicted_labels_test = []
#     for data in test_dataset:
#         true_label, predicted_label = evaluate_single_graph(classifier, PE, data, device)
#         true_labels_test.append(true_label)
#         # predicted_label = 1.0 if predicted_label == 1 else -1.0  #mutag
#         predicted_labels_test.append(predicted_label)
#
#     # 计算准确率
#     accuracy = accuracy_score(true_labels_test, predicted_labels_test)
#
#     print(f"Accuracy: {accuracy:.4f}")
#
#     # 计算混淆矩阵
#     conf_matrix = confusion_matrix(true_labels_test, predicted_labels_test)
#     print("Confusion Matrix:", conf_matrix)
