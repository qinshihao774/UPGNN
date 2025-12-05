import math
import warnings
import os
import numpy as np
import torch.nn as nn 
import torch_geometric as pyg
import torch_geometric.nn as pygnn 
from torch.utils.data import Dataset
from torch import Tensor
from typing import Union, Tuple, List, Optional, Callable, Any
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
import torch
from os import path as osp
from torch_geometric.data import Data
from upgnn.dataset.ba2motif import BA2Motif
from upgnn.dataset.dd import DD
from upgnn.dataset.frankenstein import FrankensteinTXT
from upgnn.dataset.mutag import Mutag
from upgnn.dataset.mutagenicity import Mutagenicity
from upgnn.dataset.nci1 import NCI1
from upgnn.dataset.proteins import PROTEINS
from upgnn.dataset.synthetic import Synthetic
from upgnn.trainclassifier import trainClassifier_proteins, trainClassifier_nci1, trainClassifier_ba2motif, \
    trainClassifier_dd, trainClassifier_mutag, trainClassifier_mutagenicity, trainClassifier_frankenstein, \
    trainClassifier_bbbp,trainClassifier_ogb

from ogb.graphproppred import PygGraphPropPredDataset

##################################################################
###################### GNN Helper ################################
##################################################################


class GraphDataset(Dataset):
    def __init__(self, data, slices, idx):
        self.data = data
        self.slices = slices
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        g_idx = self.idx[i]
        # 切片 x
        x = self.data.x[self.slices['x'][g_idx]:self.slices['x'][g_idx + 1]]
        # 切片 edge_index
        edge_index = self.data.edge_index[:, self.slices['edge_index'][g_idx]:self.slices['edge_index'][g_idx + 1]]
        # 标签
        y = self.data.y[g_idx].clone()
        y = y.view(1).long()  # → tensor([1]), dtype=long

        # edge_attr（如果有）
        edge_attr = None
        if 'edge_attr' in self.slices:
            edge_attr = self.data.edge_attr[self.slices['edge_attr'][g_idx]:self.slices['edge_attr'][g_idx + 1]]

        graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
        graph.num_nodes = x.size(0)
        return graph

    @property
    def num_node_features(self):
        return self.data.x.shape[1]

    @property
    def num_classes(self):
        return int(self.data.y.max().item()) + 1  # 0/1 → 2


def map_labels(dataset, old_label, new_label):
    """
    将数据集中的指定标签值映射为新的标签值

    参数:
        dataset: 输入数据集，包含多个data对象
        old_label: 需要被替换的旧标签值
        new_label: 替换后的新标签值
    返回:
        处理后的新数据集
    """
    new_dataset = []
    for data in dataset:
        # 创建与原标签同类型的新标签张量
        replacement = torch.tensor(new_label, dtype=data.y.dtype)

        # 替换指定标签值
        if data.y.dim() > 1 and data.y.shape[1] == 1:
            # 对于形状为[N, 1]的标签，替换后压缩为[N]
            data.y = torch.where(data.y == old_label, replacement, data.y).squeeze()
        else:
            # 对于其他形状的标签直接替换
            data.y = torch.where(data.y == old_label, replacement, data.y).view(-1)

        new_dataset.append(data)

    return new_dataset


def squeeze_labels(dataset):
    new_dataset = []
    for data in dataset:
        # 保证 y 是 [N] 形状的 LongTensor
        y = data.y.long()
        # 如果全是同一个值 → 压缩为标量 [1]
        if y.unique().numel() == 1:
            data.y = y[0].view(1)  # → tensor([1])
        else:
            data.y = y  # 保持原样
        new_dataset.append(data)
    return new_dataset



def set_masks(
    model: nn.Module,
    mask: Union[torch.Tensor, Parameter],
    edge_index: torch.Tensor,
    apply_sigmoid: bool = True,
):
    """Adding mask matrix to each module of to-be-explained gnn model."""

    # remove self-loops
    loop_mask = edge_index[0] != edge_index[1]

    for module in model.modules():
        if isinstance(module, MessagePassing):
            if not isinstance(mask, Parameter) and '_edge_mask' in module._parameters:
                mask = Parameter(mask)
            module.explain        = True  # explain mode
            module._edge_mask     = mask  # edge mask of every edge
            module._loop_mask     = loop_mask  # whether to remove self-loops
            module._apply_sigmoid = apply_sigmoid  # whether to apply sigmoid func

def clear_masks(model: nn.Module):
    """Clear all mask matrix of each module of to-be-explained gnn model."""
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.explain    = False
            module._edge_mask = None
            module._loop_mask = None
            module._apply_sigmoid = True
            if '_edge_mask' in module._parameters:
                module._parameters.__delitem__('_edge_mask')

def get_embeddings(
    model: torch.nn.Module,
    use_hook: bool = False,
    *args,
    **kwargs,
) -> List[torch.Tensor]:
    """Returns the output embeddings of all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers in
    :obj:`model`.

    Internally, this method registers forward hooks on all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers of a :obj:`model`,
    and runs the forward pass of the :obj:`model` by calling
    :obj:`model(*args, **kwargs)`.

    Args:
        model (torch.nn.Module): The message passing model.
        use_hook: Whether to use hook to get embeddings.
        *args: Arguments passed to the model.
        **kwargs (optional): Additional keyword arguments passed to the model.
    """
    embeddings: List[torch.Tensor] = []

    if use_hook:
        def hook(model: nn.Module, inputs: Any, outputs: Any):
            # Clone output in case it will be later modified in-place:
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            assert isinstance(outputs, torch.Tensor)
            embeddings.append(outputs.clone())

        hook_handles = []
        for module in model.modules():  # Register forward hooks:
            if isinstance(module, MessagePassing):
                hook_handles.append(module.register_forward_hook(hook))

        if len(hook_handles) == 0:
            warnings.warn("The 'model' does not have any 'MessagePassing' layers")

        training = model.training
        model.eval()
        with torch.no_grad():
            model(*args, **kwargs)
        model.train(training)

        for handle in hook_handles:  # Remove hooks:
            handle.remove()
    else:
        embeddings.append(model.embedding(*args, **kwargs)[0])

    return embeddings

def get_predicts(graph: Union[List[Data], Data, pyg.data.Batch],
                 discriminator: nn.Module,
                 device: torch.device,
                 index: Union[torch.Tensor, int] = None,
                 return_type: str = 'prob',
                 use_edge_weight: bool = False,
                 apply_sigmoid:   bool = False) -> torch.Tensor:
    if isinstance(graph, list):
        assert len(graph) != 0
        graph = pyg.data.Batch.from_data_list(graph)
    graph.to(device)
    if use_edge_weight:
        edge_weight = graph.get('edge_weight')
        if edge_weight is None:
            edge_weight = graph.get('edge_mask', torch.ones_like(graph.edge_index[0]))

        set_masks(discriminator,
                  edge_weight,
                  graph.edge_index,
                  apply_sigmoid)

    logits = discriminator(graph.x, graph.edge_index, batch=graph.get('batch'))

    if index is not None:
        logits = logits[index]

    if return_type == 'raw':
        outs = logits
    elif return_type == 'prob':
        if graph.y is not None:
            outs = logits.softmax(dim=-1)[torch.arange(logits.shape[0], device=logits.device), graph.y]
        else:
            outs = logits.softmax(dim=-1)
    elif return_type == 'all_prob':
        outs = logits.softmax(dim=-1)
    elif return_type == 'label':
        outs = logits.argmax(dim=-1)
    elif return_type == 'all_embed':
        outs = get_embeddings(discriminator, False, x=graph.x, edge_index=graph.edge_index)[-1]
    elif return_type == 'embed':
        all_embeds = get_embeddings(discriminator, False, x=graph.x, edge_index=graph.edge_index)[-1]
        outs = pygnn.global_mean_pool(all_embeds, batch=graph.batch)
    elif return_type == 'all':
        all_embeds = get_embeddings(discriminator, False, x=graph.x, edge_index=graph.edge_index)[-1]
        embeds = pygnn.global_mean_pool(all_embeds, batch=graph.batch)
        outs = {"logits": logits, "embeds": embeds, "probs": logits.softmax(dim=-1), "all_embeds": all_embeds}
    else:
        raise ValueError(f"{return_type} does not support now!")

    if use_edge_weight:
        clear_masks(discriminator)

    return outs

def unbatch_data(batch_data: Data) -> List[Data]:
    r"""
    Unbatch data from BatchData into a list of Data.
    Args:
        batch_data(BatchData): batch data information
    """
    assert (batch := batch_data.batch) is not None and batch.max() >= 0
    edge_batch = batch[batch_data.edge_index[0]]
    data_list = []
    num_nodes_per_graph = torch.bincount(batch)
    num_edges_per_graph = torch.bincount(edge_batch)
    start_node_index    = torch.repeat_interleave(
        torch.cat((torch.tensor([0], device=batch.device), torch.cumsum(num_nodes_per_graph, dim=-1)[:-1]),
                  dim=-1),
        num_edges_per_graph)
    edge_index          = batch_data.edge_index - start_node_index
    for graph_id in range(batch.max().item() + 1):
        data_list.append(
            Data(x              = batch_data.x[batch == graph_id],
                 edge_index     = edge_index[:, edge_batch == graph_id],
                 edge_mask      = batch_data.edge_mask[edge_batch == graph_id],
                 pred_edge_mask = batch_data.pred_edge_mask[0][edge_batch == graph_id],
                 corn_node_id   = batch_data.get('corn_node_id')[graph_id]
                                  if batch_data.get('corn_node_id') is not None else None,
                 target_label   = batch_data.target_label[graph_id])
        )
    return data_list

def pack_explanatory_subgraph(data: Data, importance: Tensor, top_ratio: float = 0.5) -> Data:
    r"""
    Getting new graph with importance score.
    Args:
        data(PYG data):       source graph
        importance(Tensor):   edge importance score from explainer
        top_ratio(float):     how much edges to keep
    """
    topk         = max(math.floor(top_ratio * data.num_edges), 1)
    edge_indices = torch.topk(importance, k=topk)[1]
    x, edge_index, edge_attr = relabel_nodes(data, edge_indices)

    return Data(
        x=x,
        y=data.y,
        N=data.num_nodes,
        edge_index   = edge_index,
        edge_attr    = edge_attr,
        target_label = data.target_label,
        ori_embeddings = data.get('ori_embeddings')
    )

def relabel_nodes(data: Data, edge_indices: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    edge_index = data.edge_index[:, edge_indices]
    sub_nodes  = edge_index.unique()
    row, col   = edge_index

    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((data.num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]

    return data.x[sub_nodes], edge_index, data.edge_attr[edge_indices]

def combine_mask(mask: List[Tensor], top_ratio_every_time: List[float]) -> Tensor:
    lt_num = len(mask)
    for i in range(lt_num - 1, 0, -1):
        top_ratio = top_ratio_every_time[i - 1]
        Gi_edge_idx = len(mask[i - 1])
        topk = max(math.floor(top_ratio * Gi_edge_idx), 1)
        Gi_pos_edge_idx = np.argsort(-np.abs(mask[i - 1].cpu()))[:topk]
        diff_every_time = max(mask[i - 1]) - min(mask[i]) + 1e-4
        for k, index in enumerate(Gi_pos_edge_idx):
            mask[i - 1][int(index)] = mask[i][k] + diff_every_time
    return mask[0]

def load_data(datatype):
    if datatype == 'syn':
        # TODO: Synthetic 创建训练、验证、测试集  y =  ?
        train_dataset = Synthetic(osp.join("../data/", f'Synthetic/'), mode='train')
        valid_dataset = Synthetic(osp.join("../data/", f'Synthetic/'), mode='val')
        test_dataset = Synthetic(osp.join("../data/", f'Synthetic/'), mode='test')

    elif datatype == 'ogb':

        dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='../data/ogb')
        # # TODO: mutag 创建训练、验证、测试集   y= 0 1
        # dataset = torch.load('./../data/ogb/ogb_graph.pt', weights_only=False)
        train_dataset = torch.load("../data/ogb/train_dataset_balanced.pt", weights_only=False)
        valid_dataset = torch.load("../data/ogb/valid_dataset_balanced.pt", weights_only=False)
        # print(f"data size: {len(dataset)}")

        split_idx = dataset.get_idx_split()
        # train_dataset = dataset[split_idx['train']]
        # valid_dataset = dataset[split_idx['valid']]
        test_dataset = dataset[split_idx['test']]

        train_dataset = squeeze_labels(train_dataset)
        valid_dataset = squeeze_labels(valid_dataset)
        test_dataset = squeeze_labels(test_dataset)

    elif datatype == 'mutag':
        # # TODO: mutag 创建训练、验证、测试集   y= -1 1
        train_dataset = Mutag(root='../data/mutag', split='train')
        valid_dataset = Mutag(root='../data/mutag', split='val')
        test_dataset = Mutag(root='../data/mutag', split='test')

        # train_dataset = map_labels(train_dataset, -1, 0)
        # # print("train_dataset：", train_dataset)
        # # mutag：eg   Data(x=[17, 7], edge_index=[2, 38], y=1.0, node_label=[17], edge_label=[38], node_type=[17])
        # valid_dataset = map_labels(valid_dataset, -1, 0)
        # test_dataset = map_labels(test_dataset, -1, 0)

    elif datatype == 'mutagenicity':
        # TODO: mutagenicity 创建训练、验证、测试集   y= 0 1
        train_dataset = Mutagenicity(mode='train')
        valid_dataset = Mutagenicity(mode='valid')
        test_dataset = Mutagenicity(mode='test')

    elif datatype == 'proteins':
        # TODO: proteins 创建训练、验证、测试集   y= 0 1
        train_dataset = PROTEINS('train')
        valid_dataset = PROTEINS('valid')
        test_dataset = PROTEINS('test')

    elif datatype == 'nci1':
        # TODO: nci1 创建训练、验证、测试集   y= 0 1
        train_dataset = NCI1('train')
        valid_dataset = NCI1('valid')
        test_dataset = NCI1('test')

    elif datatype == 'ba2motif':
        # TODO: ba2motif 创建训练、验证、测试集   y= 0是环[5,2]  1是房子[6,2]
        train_dataset = BA2Motif('../data/ba2motif', 'train')
        valid_dataset = BA2Motif('../data/ba2motif', 'valid')
        test_dataset = BA2Motif('../data/ba2motif', 'test')

    elif datatype == 'dd':
        # TODO: dd 创建训练、验证、测试集   y= 1 2
        train_dataset = DD('train')
        valid_dataset = DD('valid')
        test_dataset = DD('test')

        train_dataset = map_labels(train_dataset, 2, 0)
        # print("train_dataset：", train_dataset)
        # mutag：eg   Data(x=[17, 7], edge_index=[2, 38], y=1.0, node_label=[17], edge_label=[38], node_type=[17])
        valid_dataset = map_labels(valid_dataset, 2, 0)
        test_dataset = map_labels(test_dataset, 2, 0)

    elif datatype == 'frankenstein':
        # TODO: frankenstein 创建训练、验证、测试集   y= -1 1
        train_dataset = FrankensteinTXT(root='../data/frankenstein', split='train')
        valid_dataset = FrankensteinTXT(root='../data/frankenstein', split='valid')
        test_dataset = FrankensteinTXT(root='../data/frankenstein', split='test')

        train_dataset = map_labels(train_dataset, -1, 0)
        # print("train_dataset：", train_dataset)
        # mutag：eg   Data(x=[17, 7], edge_index=[2, 38], y=1.0, node_label=[17], edge_label=[38], node_type=[17])
        valid_dataset = map_labels(valid_dataset, -1, 0)
        test_dataset = map_labels(test_dataset, -1, 0)


    elif datatype == 'bbbp':
        # TODO: bbbp 创建训练、验证、测试集   y= 0 1
        # ------------------- 1. 路径配置 -------------------
        ROOT = '../data/bbbp'
        DATA_PT = f'{ROOT}/processed/data.pt'  # 完整合并数据
        TRAIN_IDX = f'{ROOT}/processed/train_idx.pt'
        VALID_IDX = f'{ROOT}/processed/valid_idx.pt'
        TEST_IDX = f'{ROOT}/processed/test_idx.pt'

        # ------------------- 2. 加载 data + slices -------------------
        data, slices = torch.load(DATA_PT, weights_only=False)

        # ------------------- 3. 加载划分索引 -------------------
        train_idx = torch.load(TRAIN_IDX, weights_only=False)
        valid_idx = torch.load(VALID_IDX, weights_only=False)
        test_idx = torch.load(TEST_IDX, weights_only=False)

        # ------------------- 5. 创建三个子集 -------------------
        train_dataset = GraphDataset(data, slices, train_idx)
        valid_dataset = GraphDataset(data, slices, valid_idx)
        test_dataset = GraphDataset(data, slices, test_idx)

    return train_dataset, valid_dataset, test_dataset


def select_func(data_name, device):
    train_dataset, valid_dataset, test_dataset = load_data(data_name)
    # dataset_root = '../../upgnn/data/ba2motif'
    # Classifier_path = './best_gnnclassifier/best_gnn_classifier_' + data_name + '.pt'
    Classifier_path = '../best_gnnclassifier/best_gnn_classifier_' + data_name + '.pt'

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

    if data_name == 'ba2motif':
        classifier = trainClassifier_ba2motif.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=12,
                                                            num_tasks=num_tasks).to(device)
    elif data_name == 'mutag':
        classifier = trainClassifier_mutag.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=32,
                                                         num_tasks=num_tasks).to(device)
    elif data_name == 'nci1':
        classifier = trainClassifier_nci1.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=32,
                                                        num_tasks=num_tasks).to(device)
    elif data_name == 'proteins':
        classifier = trainClassifier_proteins.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=128,
                                                            num_tasks=num_tasks).to(device)
    elif data_name == 'dd':
        classifier = trainClassifier_dd.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=32,
                                                      num_tasks=num_tasks).to(device)
    elif data_name == 'mutagenicity':
        classifier = trainClassifier_mutagenicity.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=32,
                                                                num_tasks=num_tasks).to(device)
    elif data_name == 'ogb':
        classifier = trainClassifier_ogb.GNNClassifier(num_layer=2, emb_dim=node_in_dim, hidden_dim=32,
                                                       num_tasks=num_tasks).to(device)
    elif data_name == 'frankenstein':
        classifier = trainClassifier_frankenstein.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=300,
                                                                num_tasks=num_tasks).to(device)
    elif data_name == 'bbbp':
        classifier = trainClassifier_bbbp.GNNClassifier(num_layer=3, emb_dim=node_in_dim, hidden_dim=16,
                                                        num_tasks=num_tasks).to(device)

    classifier.load_state_dict(torch.load(Classifier_path, weights_only=True))  # 加载预训练分类器

    return classifier, train_dataset, valid_dataset, test_dataset