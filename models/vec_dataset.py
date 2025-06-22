import copy
import logging

import numpy as np
import torch
import torch_geometric
# import torch.utils.data as data
import scipy
import pynndescent
from torch_geometric.utils import to_undirected, negative_sampling, remove_self_loops
from torch_scatter import scatter_sum
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from utils.utils import index2adjacency
from sklearn.neighbors import NearestNeighbors


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(configs):
    dataset = VECDataset(root_path=configs.root_path, name=configs.dataset, gc_type="gaussian_static", knn_k=configs.knn_k)
    edge_index, edge_weight = dataset.edge_index, dataset.weight
    data = torch_geometric.data.Data(x=dataset.feature, edge_index=edge_index, y=dataset.y, edge_attr=edge_weight)
    data.num_nodes = dataset.num_nodes
    data_dict = {}
    data_dict['feature'] = dataset.feature.clone()
    data_dict['num_features'] = dataset.num_features
    data_dict['edge_index'] = edge_index.clone()
    data_dict['degrees'] = dataset.degrees
    data_dict['edge_weight'] = edge_weight.clone()
    data_dict['num_nodes'] = dataset.num_nodes
    data_dict['labels'] = dataset.labels
    data_dict['num_classes'] = dataset.num_classes
    data_dict['neg_edge_index'] = dataset.neg_edge_index
    data_dict['adj'] = dataset.adj
    data_dict['similarities'] = dataset.similarities
    data_dict['similarities_complete'] = dataset.similarities_complete
    data_dict['y'] = dataset.y.clone()
    return data, data_dict


def load_data_mat_simple(mat_path, gc_type, scalable=False, use_feats=True, knn_k=10):

    data = scipy.io.loadmat(mat_path)
    X = np.array(data['fea']).astype(np.float32)
    y = np.array(data['gnd']).astype(np.float32).flatten()
    X = MinMaxScaler().fit_transform(X)
    spectral_config = {}
    spectral_config["n_nbg"] = knn_k
    spectral_config["scale"] = 1
    edge_index, edge_weights, similarities, similarities_complete = gaussian_graph_simple(X, spectral_config=spectral_config)
    return torch.from_numpy(X), y, edge_index, edge_weights, similarities, similarities_complete

def get_nearest_neighbors(
    X: torch.Tensor, Y: torch.Tensor = None, k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the distances and the indices of the k nearest neighbors of each data point.

    Parameters
    ----------
    X : torch.Tensor
        Batch of data points.
    Y : torch.Tensor, optional
        Defaults to None.
    k : int, optional
        Number of nearest neighbors to calculate. Defaults to 3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Distances and indices of each data point.
    """
    if Y is None:
        Y = X
    if len(X) < k:
        k = len(X)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    Dis, Ids = nbrs.kneighbors(X)
    return Dis, Ids

def get_gaussian_kernel(
    D: torch.Tensor, scale, Ids: np.ndarray, device: torch.device, is_local: bool = True
) -> torch.Tensor:
    """
    Computes the Gaussian similarity function according to a given distance matrix D and a given scale.

    Parameters
    ----------
    D : torch.Tensor
        Distance matrix.
    scale :
        Scale.
    Ids : np.ndarray
        Indices of the k nearest neighbors of each sample.
    device : torch.device
        Defaults to torch.device("cpu").
    is_local : bool, optional
        Determines whether the given scale is global or local. Defaults to True.

    Returns
    -------
    torch.Tensor
        Matrix W with Gaussian similarities.
    """

    if not is_local:
        # global scale
        W = torch.exp(-torch.pow(D, 2) / (scale**2))
    else:
        # local scales
        W = torch.exp(
            -torch.pow(D, 2).to(device)
            / (torch.tensor(scale).float().to(device).clamp_min(1e-7) ** 2)
        )
    W_complete = copy.deepcopy(W)
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + torch.t(W)) / 2.0
    W_complete = (W_complete + torch.t(W_complete)) / 2.0
    return sym_W, W_complete



def gaussian_graph_simple(X, spectral_config=None):
    X = torch.from_numpy(X).to(device)
    # X = torch.from_numpy(X)
    is_local = False
    if spectral_config is None:
        spectral_config = {}
        spectral_config["n_nbg"] = 10
        spectral_config["scale"] = 1
    n_neighbors = spectral_config["n_nbg"]
    # scale = 50**0.5
    scale = spectral_config["scale"]
    Dx = torch.cdist(X, X)
    print(X.shape, Dx.shape)
    Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
    # scale = compute_scale(Dis, k=scale_k, is_local=is_local)
    W, W_complete = get_gaussian_kernel(
        Dx, scale, indices, device=device, is_local=is_local
    )
    print(W.shape)
    edge_index, edge_weights = dense_to_sparse(W)
    return edge_index, edge_weights, W, W_complete


class VECDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, name, gc_type='gaussian_static', knn_k=10):
        path = os.path.join(root_path, f"{name}.mat")
        x, y, edge_index, weights, similarities, similarities_complete = load_data_mat_simple(path, gc_type, knn_k=knn_k)
        self.num_nodes = x.shape[0]
        self.feature = x
        self.num_features = x.shape[1]
        self.edge_index, self.weight = to_undirected(edge_index=edge_index, edge_attr=weights, reduce='mean')
        self.edge_index, self.weight = remove_self_loops(edge_index=self.edge_index, edge_attr=self.weight)
        self.degrees = scatter_sum(self.weight, self.edge_index[0])
        _, y = np.unique(np.array(y), return_inverse=True)
        self.y = torch.from_numpy(y)
        self.labels = y.tolist()
        self.num_classes = np.max(np.unique(self.labels))+1
        self.neg_edge_index = negative_sampling(self.edge_index)
        self.adj = index2adjacency(self.num_nodes, self.edge_index, self.weight, is_sparse=True)
        self.similarities = similarities
        self.similarities_complete = similarities_complete

