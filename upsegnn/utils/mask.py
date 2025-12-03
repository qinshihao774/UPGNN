## In[Import]
import torch.nn as nn
from torch_geometric.nn import MessagePassing

## In[Set mask]
def set_mask(mask, model):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._explain = True
            module._edge_mask = mask
            assert module._explain is True and module._edge_mask.equal(mask)

## In[Clear mask]
def clear_mask(model):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._explain = False
            module._edge_mask = None
            assert module._explain is False and module._edge_mask is None