from typing import List

import torch
from torch import Tensor

from torch_geometric.utils import degree

# 这个函数的作用是根据 batch 向量将一个张量 src 拆分成多个子张量，每个子张量对应一个原始样本。
# 因为批量数据集是批量处理的，所以需要拆分张量，此处batch对应一个长整型张量，为 src 中的每个元素分配一个样本索引
# 例如，如果 batch 的值为 [0, 0, 0, 1, 1, 2, 2]，则 src 的第一个元素对应于 batch 的第一个索引，第二个元素对应于 batch 的第二个索引，依此类推。
# 索引值不同的情况下，src 中的元素会分配给不同的样本。 相同的话，src中的元素会分配给同一个样本。
def unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]: # 按维度0进行拆分
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`

    Example:

        >>> src = torch.arange(7)
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    print('src.shape:',src.shape)
    print('batch.shape:',batch.shape)
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


# 这个函数专门用于处理图的边索引（edge_index），将批处理的边索引重新拆分为每个图的边索引。
def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_index(edge_index, batch)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]))
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)