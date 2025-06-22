import torch.nn.functional as F
import torch
import numpy as np
from torch_geometric.utils import to_undirected, remove_self_loops, contains_self_loops
from torch_scatter import scatter_sum
import torch_geometric
from torch_geometric.utils import subgraph

EOS = 1e-10


def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        vals = torch.clamp(vals, min=0.01)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values

def gaussian_knn(X, k, scale=1, EPS=1e-10):
    X_std = (X - X.min(dim=0, keepdim=True).values) / (X.max(dim=0, keepdim=True).values - X.min(dim=0, keepdim=True).values).clamp(EPS)
    X_scaled = X_std * (X.max() - X.min()) + X.min()
    Dx = torch.cdist(X_scaled, X_scaled)
    SIM = torch.exp(-Dx **2 / 2*scale**2)
    vals, inds = SIM.topk(k=k+1, dim=-1)
    values = vals.view(-1)
    cols = inds.view(-1)
    rows = torch.arange(X.size(0)).view(-1, 1).repeat(1, k+1).view(-1).to(X.device)
    return rows, cols, values


def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph

def contrastive_loss_hyperbolic(x, x_aug, r3, t3, manifold, sym=True, eps=1e-12):
    batch_size, dim = x.size()
    x_expand = x[:,None,:].expand(batch_size, batch_size, dim)
    x_aug_expand = x_aug[None,:,:].expand(batch_size, batch_size, dim)
    dist_matrix = manifold.dist(x_expand, x_aug_expand)
    dist_matrix = torch.exp((r3 - dist_matrix) / t3)  # batch_size x batch_size

    pos_dist = dist_matrix[range(batch_size), range(batch_size)]
    if sym:
        loss_0 = pos_dist / (dist_matrix.sum(dim=0) - pos_dist).clamp(eps)
        loss_1 = pos_dist / (dist_matrix.sum(dim=1) - pos_dist).clamp(eps)

        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
        return loss
    else:
        loss_0 = pos_dist / (dist_matrix.sum(dim=0) - pos_dist).clamp(eps)  # sum on all anchor graph vertex representations.
        loss_0 = - torch.log(loss_0).mean()
        return loss_0


def torch_sparse_eye(num_nodes):
    indices = torch.arange(num_nodes).repeat(2, 1)
    values = torch.ones(num_nodes)
    # return torch.sparse.FloatTensor(indices, values)
    return torch.sparse_coo_tensor(indices=indices, values=values, size=(num_nodes, num_nodes))

def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask.cuda(), samples


def normalize(adj, mode, sparse=False):
    if not sparse:
        raise NotImplementedError
    else:
        adj = adj.coalesce()
        edge_indices, edge_values = to_undirected(adj.indices(), adj.values())
        deg = scatter_sum(edge_values, edge_indices[0], dim_size=adj.size(0))
        deg = torch.clamp(deg, 1e-10)
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(deg))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            raise NotImplementedError
        new_values = adj.values() * D_value

        return torch.sparse_coo_tensor(adj.indices(), new_values, adj.size()).to(adj.device)


def sampling_neighbor(data, n_seeds, sampling_size, training_all_nodes=True):
    edge_index, edge_weight, x, y = data.edge_index, data.edge_attr, data.x, data.y
    # edge_index, edge_weight = to_undirected(edge_index=edge_index, edge_attr=edge_weight, reduce='mean')
    # edge_index, edge_weight = remove_self_loops(edge_index=edge_index, edge_attr=edge_weight)
    n_edges = edge_index.shape[1]
    assert not contains_self_loops(edge_index)
    n_instance = data.num_nodes
    sampled_set_list = []
    subgraph_list = []
    visited = torch.zeros([n_instance], dtype=torch.bool, device=edge_index.device)
    while n_instance - torch.sum(visited) >= sampling_size + n_seeds:
        seeds = np.random.choice(torch.argwhere(torch.logical_not(visited)).flatten().cpu().numpy(), size=n_seeds,
                                 replace=False)
        frontier = torch.from_numpy(seeds).to(edge_index.device)
        visited[frontier] = True
        subgraph_nodes = frontier
        while subgraph_nodes.shape[0] < sampling_size:
            mask0 = torch.isin(edge_index[0], frontier)
            new_neighbors0 = edge_index[1][mask0]
            mask1 = torch.isin(edge_index[1], frontier)
            new_neighbors1 = edge_index[0][mask1]
            new_neighbors = torch.cat([new_neighbors0, new_neighbors1], dim=-1)
            mask = ~visited[new_neighbors]
            new_neighbors_unvisited = new_neighbors[mask]
            unique_new_nodes = torch.unique(new_neighbors_unvisited)
            if subgraph_nodes.shape[0] + unique_new_nodes.shape[0] > sampling_size:
                unique_new_nodes = unique_new_nodes[:sampling_size - subgraph_nodes.shape[0]]
            if unique_new_nodes.shape[0] == 0:
                seeds = np.random.choice(torch.argwhere(torch.logical_not(visited)).flatten().cpu().numpy(),
                                         size=min(1, sampling_size - subgraph_nodes.shape[0]), replace=False)
                unique_new_nodes = torch.from_numpy(seeds).to(edge_index.device)
            subgraph_nodes = torch.cat([subgraph_nodes, unique_new_nodes], dim=-1)
            frontier = unique_new_nodes
            visited[frontier] = True
        sampled_set_list.append(subgraph_nodes)
    if training_all_nodes:
        subgraph_nodes = torch.argwhere(torch.logical_not(visited)).flatten()
        sampled_set_list.append(subgraph_nodes)
        assert torch.unique(torch.cat(sampled_set_list)).shape[0] == n_instance
    for subgraph_nodes in sampled_set_list:
        sub_edge_index, sub_edge_weight, edge_mask = subgraph(edge_index=edge_index, edge_attr=edge_weight,
                                                              subset=subgraph_nodes,
                                                              relabel_nodes=True, num_nodes=n_instance,
                                                              return_edge_mask=True)
        sub_x = x[subgraph_nodes]
        sub_y = y[subgraph_nodes]
        sub_e_id = torch.arange(n_edges).to(edge_index.device)[edge_mask]
        subgraph_i = torch_geometric.data.Data(edge_index=sub_edge_index,
                                               edge_attr=sub_edge_weight,
                                               x=sub_x, y=sub_y,
                                               e_id=sub_e_id,
                                               n_id=subgraph_nodes,
                                               num_nodes=subgraph_nodes.shape[0])
        subgraph_list.append(subgraph_i)
    return subgraph_list


def anchor_sample_undirected_edges(edge_index, e_id, num_edges_to_sample):
    """
    Correctly samples undirected edges and returns the e_ids for both directions.

    Args:
        edge_index (torch.Tensor): Tensor of shape [2, num_edges] containing
                                   both (i, j) and (j, i) for each edge.
        e_id (torch.Tensor): Tensor of shape [num_edges] containing the ID for
                             each directed edge in edge_index.
        tau (float): The fraction of edges to DROP. The sampling rate is 1 - tau.

    Returns:
        torch.Tensor: A 1D tensor containing the e_ids of the sampled edges,
                      including both directions.
    """
    # Step 1: Isolate Unique Undirected Edges by creating a canonical representation
    # We choose the convention where the source node index is less than the target.
    # This gives us one unique representation for each undirected edge.
    mask_canonical = edge_index[0] < edge_index[1]
    # Get the indices of these canonical edges in the original full edge_index
    # These indices point to one direction of each unique edge.
    canonical_edge_original_indices = torch.where(mask_canonical)[0]

    num_unique_edges = canonical_edge_original_indices.numel()

    # # Step 2: Sample from the set of unique edges
    # sample_rate = 1.0 - tau
    # num_edges_to_sample = int(num_unique_edges * sample_rate)

    # Randomly select a subset of the unique edges
    # We shuffle the indices of the unique edges and take the first `num_edges_to_sample`.
    perm = torch.randperm(num_unique_edges)
    sampled_unique_indices_in_list = perm[:num_edges_to_sample]

    # Get the original indices (from the full edge_index) of the *sampled* canonical edges
    sampled_canonical_original_indices = canonical_edge_original_indices[sampled_unique_indices_in_list]

    # Step 3: Identify the partners of the sampled edges
    # These are the (i, j) pairs that we've just sampled.
    sampled_canonical_edges = edge_index[:, sampled_canonical_original_indices]

    # To efficiently find the partners (j, i), we can create a lookup map.
    # For performance on very large graphs, a numerical hash could be used.
    # For clarity and general use, a dictionary is very effective.
    # This map takes an edge tuple (u, v) and returns its index in the original edge_index.
    edge_to_original_idx_map = {tuple(edge.tolist()): i for i, edge in enumerate(edge_index.T)}

    partner_original_indices = []
    for edge in sampled_canonical_edges.T:
        # edge is a tensor [i, j]
        i, j = edge[0].item(), edge[1].item()
        # The partner edge is (j, i)
        if (j, i) in edge_to_original_idx_map:
            partner_idx = edge_to_original_idx_map[(j, i)]
            partner_original_indices.append(partner_idx)

    partner_original_indices = torch.tensor(partner_original_indices, dtype=torch.long)

    # Step 4: Retrieve the e_ids for both directions
    # Get the e_ids for the canonical edges we sampled
    sampled_canonical_eids = e_id[sampled_canonical_original_indices]

    # Get the e_ids for their partners
    sampled_partner_eids = e_id[partner_original_indices]

    # Concatenate them to get the final list of all sampled e_ids
    final_sampled_eids = torch.cat([sampled_canonical_eids, sampled_partner_eids])

    return final_sampled_eids


def learner_sample_undiredted_edges(edge_index, edge_values, num_edges_to_sample):
    mask_canonical = edge_index[0] < edge_index[1]
    canonical_edge_original_indices = torch.where(mask_canonical)[0]
    num_unique_edges = canonical_edge_original_indices.numel()
    perm = torch.randperm(num_unique_edges)
    sampled_unique_indices_in_list = perm[:num_edges_to_sample]
    sampled_canonical_original_indices = canonical_edge_original_indices[sampled_unique_indices_in_list]
    sampled_canonical_edges = edge_index[:, sampled_canonical_original_indices]
    sampled_canonical_values = edge_values[sampled_canonical_original_indices]
    learner_keep_edges, learner_keep_values = torch_geometric.utils.to_undirected(edge_index=sampled_canonical_edges,
                                                                                  edge_attr=sampled_canonical_values,
                                                                                  reduce='mean')
    return learner_keep_edges, learner_keep_values


def update_data_prob_batch(batch, learner_adj, tau):
    e_id = batch.e_id.to(torch.long).to(learner_adj.device)
    n_id = batch.n_id.to(learner_adj.device)
    num_unique_edges = int(e_id.shape[0] / 2)
    num_edges_to_sample = int(num_unique_edges * (1.0 - tau))
    final_sampled_eids = anchor_sample_undirected_edges(batch.edge_index, e_id, num_edges_to_sample)
    anchor_remove_e_id = final_sampled_eids

    # learner_adj
    learner_indices = learner_adj.indices()
    learner_values = learner_adj.values()
    assert not contains_self_loops(learner_indices)
    learner_num_edges = learner_values.shape[0]
    assert learner_num_edges > 0
    learner_keep_edges, learner_keep_values = learner_sample_undiredted_edges(learner_indices, learner_values,
                                                                              num_edges_to_sample)

    return anchor_remove_e_id, learner_keep_edges, learner_keep_values


def update_data_prob(data, anchor_remove_e_id_list, learner_keep_edges_list, learner_keep_values_list):
    edge_index = data.edge_index.detach().clone()
    edge_values = data.edge_attr.detach().clone()
    if len(anchor_remove_e_id_list) > 0 and len(learner_keep_edges_list) > 0:
        anchor_remove_e_id = torch.cat(anchor_remove_e_id_list, dim=-1)
        learner_keep_edges = torch.cat(learner_keep_edges_list, dim=-1)
        learner_keep_values = torch.cat(learner_keep_values_list, dim=-1)
        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        keep_mask[anchor_remove_e_id] = False
        edge_index = edge_index[:, keep_mask]
        edge_values = edge_values[keep_mask]
        edge_index = torch.cat([edge_index, learner_keep_edges], dim=-1)
        edge_values = torch.cat([edge_values, learner_keep_values], dim=-1)
    data_new = torch_geometric.data.Data(x=data.x, y=data.y, edge_index=edge_index, edge_attr=edge_values)
    return data_new


