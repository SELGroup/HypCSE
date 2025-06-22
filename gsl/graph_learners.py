import torch
import torch.nn as nn
import torch.nn.functional as F
from gsl.gsl_utils import top_k, apply_non_linearity, cal_similarity_graph, knn_fast, gaussian_knn
from gsl.gsl_layers import GraphConvolution
from utils.utils import index2adjacency
from torch_geometric.utils import to_undirected, remove_self_loops


class GNN_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, mlp_act, adj):
        super(GNN_learner, self).__init__()
        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(GraphConvolution(isize, isize))
        else:
            self.layers.append(GraphConvolution(isize, isize))
            for _ in range(nlayers - 2):
                self.layers.append(GraphConvolution(isize, isize))
            self.layers.append(GraphConvolution(isize, isize))

        self.input_dim = isize
        self.output_dim = isize
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.param_init()
        self.i = i
        self.sparse = sparse
        self.mlp_act = mlp_act
        self.adj = adj

    def internal_forward(self, h, adj):
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, features):
        if self.sparse:
            embeddings = self.internal_forward(features, self.adj)
            if self.knn_metric == 'cosine':
                rows, cols, values = knn_fast(embeddings, self.k, 1000)
            elif self.knn_metric == 'gaussian':
                rows, cols, values = gaussian_knn(embeddings, self.k)
            else:
                raise NotImplementedError
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            edge_index = torch.stack([rows_, cols_], dim=0)
            edge_index, values_ = remove_self_loops(edge_index, values_)
            adj = index2adjacency(N=features.shape[0], edge_index=edge_index, weight=values_)
            return adj.to(features.device)
        else:
            embeddings = self.internal_forward(features, self.adj)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities


class GNN_learner_adj(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, mlp_act):
        super(GNN_learner_adj, self).__init__()
        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(GraphConvolution(isize, isize))
        else:
            self.layers.append(GraphConvolution(isize, isize))
            for _ in range(nlayers - 2):
                self.layers.append(GraphConvolution(isize, isize))
            self.layers.append(GraphConvolution(isize, isize))

        self.input_dim = isize
        self.output_dim = isize
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.param_init()
        self.i = i
        self.sparse = sparse
        self.mlp_act = mlp_act

    def internal_forward(self, h, adj):
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, features, adj):
        if self.sparse:
            embeddings = self.internal_forward(features, adj)
            if self.knn_metric == 'cosine':
                rows, cols, values = knn_fast(embeddings, self.k, 1000)
            elif self.knn_metric == 'gaussian':
                rows, cols, values = gaussian_knn(embeddings, self.k)
            else:
                raise NotImplementedError
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            edge_index = torch.stack([rows_, cols_], dim=0)
            edge_index, values_ = remove_self_loops(edge_index, values_)
            adj_out = index2adjacency(N=features.shape[0], edge_index=edge_index, weight=values_)
            return adj_out.to(features.device)
        else:
            embeddings = self.internal_forward(features, adj)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities

class MLP_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, act):
        super(MLP_learner, self).__init__()

        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(nn.Linear(isize, isize))
        else:
            self.layers.append(nn.Linear(isize, isize))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(isize, isize))
            self.layers.append(nn.Linear(isize, isize))

        self.input_dim = isize
        self.output_dim = isize
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.param_init()
        self.i = i
        self.sparse = sparse
        self.act = act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)
        return h

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, features, adj=None):
        if self.sparse:
            embeddings = self.internal_forward(features)
            # rows, cols, values = knn_fast(embeddings, self.k, 1000)
            if self.knn_metric == 'cosine':
                rows, cols, values = knn_fast(embeddings, self.k, 1000)
            elif self.knn_metric == 'gaussian':
                rows, cols, values = gaussian_knn(embeddings, self.k)
            else:
                raise NotImplementedError
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            edge_index = torch.stack([rows_, cols_], dim=0)
            edge_index, values_ = remove_self_loops(edge_index, values_)
            # adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            # adj.edata['w'] = values_
            adj = index2adjacency(N=features.shape[0], edge_index=edge_index, weight=values_)
            return adj.to(features.device)
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities
