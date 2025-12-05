import io
import os
import random

import numpy as np
import logging
import torch
from torch_geometric.loader import DataLoader
from model import set_seed, generate_explanation, train, Pretrain_Explainer, retune
from torch.utils.data import Subset
from metrics import evaluate_single_graph, calculate_sparsity, compute_fidelity_minus, compute_fidelity_plus

from sklearn.metrics import accuracy_score, confusion_matrix

from utils.datasetutils import select_func
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
set_seed(42)
dataset_list = ['ba2motif', 'bbbp', 'mutag', 'nci1', 'proteins', 'dd', 'mutagenicity', 'ogb', 'frankenstein']

save_path = 'pretrained'
# train_dataset, valid_dataset, test_dataset = load_data(data_name)
# Classifier_path = './best_gnnclassifier/best_gnn_classifier_' + data_name + '.pt'
# pretrained_Explainer_path = './pretrained/trained_explainer_graph_' + data_name + '.pt'
pretrained_Explainer_path = './pretrained/pretrained_explainer_all.pt'
# pretrained_gnn_path = './pretrained/trained_gnnEncoder_graph_' + data_name + '.pt'
# pretrained_model_path = './pretrained/trained_model_graph_' + data_name + '.pt'
os.makedirs(save_path, exist_ok=True)  # exist_ok=True 创建目录时，如果目录已经存在，则不报错
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## In[Dataset]
# DataLoader用于从数据集加载数据并将其组织成小批量的工具类
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# val_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

classifier = None
PE = Pretrain_Explainer(classifier, 5, 256, device, explain_graph=True, loss_type='NCE')  # 初始化解释器

# # TODO: mutiple dataset pretrain...
# for dataset in dataset_list:
#     print(f"dataset is {dataset}, Pretraining...")
#     classifier, train_dataset, valid_dataset, test_dataset = select_func(dataset)
#     PE.model = classifier
#     train(PE, train_dataset, valid_dataset, logger, pretrained_Explainer_path, device, epochs=5)
# # PE.generate_explanation(test_dataset, device)
# torch.save(PE.explainer.state_dict(), pretrained_Explainer_path)
# print("\n--------------pretrained explainer save successfully!-----------------\n")

# TODO: mutiple dataset pretrained without retune...
PE.explainer.load_state_dict(torch.load(pretrained_Explainer_path, weights_only=True))
print("\n--------------Pretrained Without Retune...-----------------\n")
for dataset in dataset_list:
    classifier, train_dataset, valid_dataset, test_dataset = select_func(dataset, device=device)
    PE.model = classifier

    with torch.no_grad():
        # 逐图验证
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

        # 逐图测试嵌入
        print("Testing single graph pred_prob auc...")
        # 评估模型性能
        true_labels_test = []
        predicted_labels_test = []
        for data in test_dataset:
            true_label, predicted_label = evaluate_single_graph(classifier, PE, data, device)
            true_labels_test.append(true_label)
            # predicted_label = 1.0 if predicted_label == 1 else -1.0  #mutag
            predicted_labels_test.append(predicted_label)

        # 计算准确率
        accuracy = accuracy_score(true_labels_test, predicted_labels_test)

        print(f"Accuracy: {accuracy:.4f}")

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(true_labels_test, predicted_labels_test)
        print("Confusion Matrix:", conf_matrix)

# # TODO: mutiple dataset pretrained with retune...
print("--------------Pretrained And Retune...-----------------")
for dataset in dataset_list:
    PE.explainer.load_state_dict(torch.load(pretrained_Explainer_path, weights_only=True))
    print(f"\ndataset is {dataset},Refining...")
    classifier, train_dataset, valid_dataset, test_dataset = select_func(dataset, device=device)
    PE.model = classifier

    train_sub_dataset = Subset(train_dataset, range(100))  # 前 100 个
    retune(PE, train_sub_dataset, device, epochs=3)
    # PE.generate_explanation(test_dataset, device)

    with torch.no_grad():
        # 逐图验证
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

        # 逐图测试嵌入
        print("Testing single graph pred_prob auc...")
        # 评估模型性能
        true_labels_test = []
        predicted_labels_test = []
        for data in test_dataset:
            true_label, predicted_label = evaluate_single_graph(classifier, PE, data, device)
            true_labels_test.append(true_label)
            # predicted_label = 1.0 if predicted_label == 1 else -1.0  #mutag
            predicted_labels_test.append(predicted_label)

        # 计算准确率
        accuracy = accuracy_score(true_labels_test, predicted_labels_test)

        print(f"Accuracy: {accuracy:.4f}")

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(true_labels_test, predicted_labels_test)
        print("Confusion Matrix:", conf_matrix)

print("--------------done!-----------------")

# Confusion Matrix:
# [[3405  578]   ->   [TP FN]
# [ 102   28]]   ->   [FP TN]
# ------------------------------------------------------------------------------------------------
