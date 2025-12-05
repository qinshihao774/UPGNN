import numpy as np
import scipy.sparse as sp
import torch
import scipy
import pickle as pkl
from scipy.sparse import coo_matrix

def sparsity(exp):
    selected = torch.where(exp > 0.5, 1, 0).sum().cpu()
    return selected / exp.shape[0]

def adj_to_edge_index(adj):
    """
    Convert an adjacency matrix to an edge index
    :param adj: Original adjacency matrix
    :return: Edge index representation of the graphs
    """
    converted = []
    for d in adj:
        edge_index = np.argwhere(d > 0.).T
        converted.append(edge_index)

    return converted


def preprocess_features(features):
    """
    Preprocess the features and transforms them into the edge index representation
    :param features: Orginal feature representation
    :return: edge index representation
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features).astype(np.float32)
    try:
        return features.todense() # [coordinates, data, shape], []
    except:
        return features

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        values = values.astype(np.float32)
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_adj(adj):
    """
    Transforms adj matrix into edge index.
    Is different to adj_to_edge_index in terms of how the final representation can be used
    :param adj: adjacency matrix
    :return: edge index
    """
    return sparse_to_tuple(sp.coo_matrix(adj))