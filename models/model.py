#  reload the HYPCSE model to ensure it being correct.
import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from embedding_leaves.embedders import ShallowEmbedder, HGCNEmbedder, HyboNetEmbedder, LorentzShallowEmbedder
from utils.lca import hyp_lca
from models.encoders import FermiDiracDecoder, LorentzGraphEncoder, LorentzDecoder
import networkx as nx
from manifold.poincare import project
from utils.linkage import nn_merge_uf_fast_np, sl_from_embeddings
from manifold.lorentz import Lorentz
from utils.utils import select_activation
from gsl.gsl_layers import SparseDropout
from models.layers import LorentzLinear
from torch.nn import Sequential, ReLU
from torch_scatter import scatter_sum
from torch_geometric.utils import to_undirected, remove_self_loops, degree
from sklearn.cluster import AgglomerativeClustering
from utils.linkage import fast_tree_decoding




class HYPCSE(torch.nn.Module):
    def __init__(self, embedder, in_features, hidden_dim_enc, num_nodes=None, n_layers=2, t2=1., r2=2.,
                 embed_dim=64, proj_dim=64, dropout=0.5, dropedge_rate=0.5, nonlin='relu', select_k=True, sparse=True, n_classes=None):
        super(HYPCSE, self).__init__()
        if embedder == 'LSENet':
            self.manifold = Lorentz()
            self.nonlin = select_activation(nonlin) if nonlin is not None else None
            self.t2 = t2
            self.r2 = r2
            self.scale = nn.Parameter(torch.tensor([0.999]), requires_grad=True)
            self.encoder = LorentzGraphEncoder(self.manifold, n_layers, in_features+1, hidden_dim_enc, embed_dim+1,
                                                use_att=False, use_bias=True, dropout=dropout, nonlin=self.nonlin)
            c = None

            self.sparse = sparse
            self.dropedge_rate = dropedge_rate
            if self.sparse:
                self.dropout_adj = SparseDropout(dprob=dropedge_rate)
            else:
                raise NotImplementedError

            self.proj_head = Sequential(LorentzLinear(self.manifold, embed_dim+1, proj_dim+1),
                                        ReLU(inplace=True),
                                        LorentzLinear(self.manifold, proj_dim+1, proj_dim+1))
            if n_classes is not None:
                self.decoder_nc = LorentzDecoder(c, self.manifold, embed_dim+1, n_classes)
        else:
            raise Exception("Not Implemented")
        self.select_k = select_k
        self.dc = FermiDiracDecoder(r=2., t=1.)

    def encode(self, x, adj):
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)
        h = self.encoder.encode(x, adj)
        h = self.normalize(h)
        return h

    def normalize(self, x):
        x = self.manifold.to_poincare(x)
        x = F.normalize(x, p=2, dim=-1) * self.scale.clamp(1e-2, 0.999)
        x = self.manifold.from_poincare(x)
        return x

    def forward(self, x, adj_, branch=None):
        if self.sparse:
            if branch == 'anchor':
                adj = copy.deepcopy(adj_)
            else:
                adj = adj_
            adj = adj.coalesce()
            dropout_values = F.dropout(adj.values(), p=self.dropedge_rate)
            adj = torch.sparse_coo_tensor(adj.indices(), dropout_values, adj.size()).to(adj.device)
        else:
            raise NotImplementedError
        x = self.encode(x, adj)
        z = self.proj_head(x)
        return z, x

    def se_loss(self, embeddings, adj, se_sparse=False, se_k=10, MIN_NORM=1e-10):
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = to_undirected(edge_index=edge_index, edge_attr=edge_weight, reduce='mean')
        edge_index, edge_weight = remove_self_loops(edge_index=edge_index, edge_attr=edge_weight)
        num_nodes = embeddings.size(0)
        node_degrees = scatter_sum(src=edge_weight, index=edge_index[0], dim_size=num_nodes)
        assert node_degrees.size(0) == embeddings.size(0)
        leaves_embeddings = self.manifold.to_poincare(embeddings, dim=-1)
        n, d = leaves_embeddings.size()
        m = edge_index.size(1)
        embeddings_i = leaves_embeddings[:,None,:].expand(n, n, d)
        embeddings_j = leaves_embeddings[None,:,:].expand(n, n, d)
        dist_ij = hyp_lca(embeddings_i, embeddings_j, return_coord=False) # n*n
        dist_ij = torch.squeeze(dist_ij, dim=-1)
        dist_ik = dist_ij[edge_index[0]]
        dist_jk = dist_ij[edge_index[1]]
        dist_ij = dist_ij[edge_index[0], edge_index[1]][:, None].expand(m, n)  # should be m * n
        Vk = node_degrees[None, :].expand(m, n)
        if self.r2 is None:
            lca_norm = torch.stack((1./(dist_ij + MIN_NORM), 1./(dist_ik + MIN_NORM), 1./(dist_jk + MIN_NORM)), dim=-1)
        else:
            lca_norm = torch.stack((self.r2-dist_ij, self.r2-dist_ik, self.r2-dist_jk), dim=-1)
        weights = torch.softmax(lca_norm / self.t2, dim=-1)  # m * n * 3
        if self.select_k:       # select k where k != i and k != j
            select_k = torch.ones_like(weights)
            select_k[torch.arange(m), edge_index[0], 0] = 0
            select_k[torch.arange(m), edge_index[1], 0] = 0
            weights = weights * select_k
            V_ij = (node_degrees[edge_index[0]] + node_degrees[edge_index[1]])  # V(T_i) + V(T_j), size: m
            volumes = torch.sum(Vk * weights[:,:,0], dim=-1) + V_ij # m
        else:
            volumes = torch.sum(Vk * weights[:, :, 0], dim=-1)
        volumes_log = torch.log2(volumes + MIN_NORM)
        se_loss = torch.sum(edge_weight * volumes_log)
        VOL_G = torch.sum(node_degrees)
        se_loss = se_loss / VOL_G  # not 2 * se_loss / VOL_G since edges are counted ordered.
        return se_loss


    def lp_loss(self, embeddings, edge_index, neg_edge_index, device=torch.device('cuda:0')):
        edges = torch.cat([edge_index, neg_edge_index], dim=-1)
        dist = self.manifold.dist(embeddings[edges[0]], embeddings[edges[1]])
        prob = self.dc.forward(dist)
        label = torch.cat([torch.ones(edge_index.shape[-1]), torch.zeros(neg_edge_index.shape[-1])]).to(device)
        lp_loss = F.binary_cross_entropy(prob, label)
        embeddings_0 = self.manifold.Frechet_mean(embeddings)
        return self.manifold.dist0(embeddings_0), lp_loss

    def dist0_loss(self, embeddings):
        embeddings_0 = self.manifold.Frechet_mean(embeddings)
        return self.manifold.dist0(embeddings_0)

    def decode_tree(self, leaves_embeddings, decoding_algo=None, n_cluster=None, fast_decoding=False):
        """Build a binary tree (nx graph) from leaves' embeddings. Assume points are normalized to same radius."""
        leaves_embeddings = project(leaves_embeddings).detach().cpu().to(torch.float64)
        sim_fn = lambda x, y: torch.sum(x * y, dim=-1)
        dist_fn = lambda x, y: - np.sum(x * y, axis=-1)
        if decoding_algo is None or decoding_algo=='exact':
            if fast_decoding:
                tree = fast_tree_decoding(leaves_embeddings.numpy(), dist_fn=dist_fn, n_clusters=n_cluster)
            else:
                parents = sl_from_embeddings(leaves_embeddings, sim_fn)
                tree = nx.DiGraph()
                for i, j in enumerate(parents[:-1]):
                    tree.add_edge(j, i)
            return tree

        elif decoding_algo in ["single", "average", "complete", "ward"]:
            raise NotImplementedError
        else:
            raise NotImplementedError

