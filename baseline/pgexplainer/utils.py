import math
import warnings
import torch
import os
import random
import numpy as np
import torch.nn as nn 
import torch_geometric as pyg
import torch_geometric.nn as pygnn 

from torch.nn import Parameter
from torch import Tensor
from torch_geometric.nn import MessagePassing
from typing import Union, Tuple, List, Optional, Callable, Any
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

##################################################################
###################### GNN Helper ################################
##################################################################


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