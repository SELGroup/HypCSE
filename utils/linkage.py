"""Decoding utils."""

import time

import networkx as nx
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

from mst import mst
from unionfind import unionfind
from utils.lca import hyp_lca


### Single linkage using MST trick

# @profile
def sl_np_mst(similarities):
    n = similarities.shape[0]
    ij, _ = mst.mst(similarities, n)
    uf = unionfind.UnionFind(n)
    uf.merge(ij)
    return uf.tree

def sl_from_embeddings(xs, S):
    xs0 = xs[None, :, :]
    xs1 = xs[:, None, :]
    sim_mat = S(xs0, xs1)  # (n, n)
    return sl_np_mst(sim_mat.numpy())

### Single linkage using naive union find

# @profile
def nn_merge_uf_fast_np(xs, S, partition_ratio=None, verbose=False):
    """ Uses Cython union find and numpy sorting

    partition_ratio: either None, or real number > 1
    similarities will be partitioned into buckets of geometrically increasing size
    """
    n = xs.shape[0]
    # Construct distance matrix (negative similarity; since numpy only has increasing sorting)
    xs0 = xs[None, :, :]
    xs1 = xs[:, None, :]
    dist_mat = -S(xs0, xs1)  # (n, n)
    i, j = np.meshgrid(np.arange(n, dtype=int), np.arange(n, dtype=int))

    # Keep only unique pairs (upper triangular indices)
    idx = np.tril_indices(n, -1)
    ij = np.stack([i[idx], j[idx]], axis=-1)
    dist_mat = dist_mat[idx]

    # Sort pairs
    if partition_ratio is None:
        idx = np.argsort(dist_mat, axis=0)
    else:
        k, ks = ij.shape[0], []
        while k > 0:
            k = int(k // partition_ratio)
            ks.append(k)
        ks = np.array(ks)[::-1]
        if verbose:
            print(ks)
        idx = np.argpartition(dist_mat, ks, axis=0)
    ij = ij[idx]

    # Union find merging
    uf = unionfind.UnionFind(n)
    uf.merge(ij)
    return uf.tree

def fast_tree_decoding(leaves_embeddings, dist_fn, n_clusters, n_neighbors=100):
    n_instance = leaves_embeddings.shape[0]
    n_neighbors = min(n_neighbors, n_instance)
    # print(type(leaves_embeddings))
    # exit(0)
    knn_graph = kneighbors_graph(leaves_embeddings, n_neighbors=n_neighbors, include_self=False, metric=dist_fn)
    model = AgglomerativeClustering(linkage='single', connectivity=knn_graph, n_clusters=n_clusters, compute_full_tree=True)
    model.fit(leaves_embeddings)
    children_ = model.children_
    tree = nx.DiGraph()
    for i in range(0, n_instance):
        tree.add_node(i)
    for i in range(children_.shape[0]):
        parent = i + n_instance
        tree.add_node(parent)
        tree.add_edge(parent, children_[i][0])
        tree.add_edge(parent, children_[i][1])
    return tree
