import copy
import time

import networkx as nx
import torch
from sklearn.metrics import accuracy_score

from logger import create_logger
# from models.vec_dataset import load_data
from models.model import HYPCSE
from geoopt.optim import RiemannianAdam
import numpy as np
from utils.metrics import dasgupta_cost, se_cost, den_purity_recursive, den_purity, se_cost_iterative
import random
from gsl.gsl_utils import torch_sparse_eye, normalize, get_feat_mask, contrastive_loss_hyperbolic
from gsl.graph_learners import GNN_learner, MLP_learner
import traceback
from torch import nn
from models.encoders import LorentzDecoder
from manifold.lorentz import Lorentz
import torch.nn.functional as F
import torch.utils.data as data
import os
from models.vec_dataset import load_data_mat_simple
from torch_scatter import scatter_sum
from torch_geometric.utils import to_undirected, negative_sampling, remove_self_loops
from utils.utils import index2adjacency
from sklearn.decomposition import PCA


class DatasetNC(data.Dataset):
    def __init__(self, root_path, name, knn_k=10):
        path = os.path.join(root_path, f"{name}.mat")
        _, y, edge_index, weights, similarities, similarities_complete = load_data_mat_simple(path, "gaussian_static", knn_k=knn_k)
        self.num_nodes = y.shape[0]
        self.edge_index, self.weight = to_undirected(edge_index=edge_index, edge_attr=weights, reduce='mean')
        self.edge_index, self.weight = remove_self_loops(edge_index=self.edge_index, edge_attr=self.weight)
        self.degrees = scatter_sum(self.weight, self.edge_index[0])
        _, y = np.unique(np.array(y), return_inverse=True)
        self.labels = y.tolist()
        self.num_classes = np.max(np.unique(self.labels))+1
        # self.neg_edge_index = negative_sampling(self.edge_index)
        self.adj = index2adjacency(self.num_nodes, self.edge_index, self.weight, is_sparse=True)
        self.similarities = similarities
        self.similarities_complete = similarities_complete

def load_data_nc(configs):
    dataset = DatasetNC(root_path=configs.root_path, name=configs.dataset, knn_k=configs.knn_k)
    data = {}
    data['edge_index'] = dataset.edge_index
    data['degrees'] = dataset.degrees
    data['edge_weight'] = dataset.weight
    data['num_nodes'] = dataset.num_nodes
    data['labels'] = dataset.labels
    data['num_classes'] = dataset.num_classes
    data['adj'] = dataset.adj
    data['similarities'] = dataset.similarities
    data['similarities_complete'] = dataset.similarities_complete
    idx = torch.randperm(data['num_nodes'], dtype=torch.long)
    data['idx'] = idx
    data['idx_train'] = idx[:int(0.3*data['num_nodes'])]
    data['idx_test'] = idx[int(0.3*data['num_nodes']):int(0.9*data['num_nodes'])]
    data['idx_val'] = idx[int(0.9*data['num_nodes']):]
    data['num_features'] = 10
    pca = PCA(n_components=data['num_features'])
    sim_reduced = pca.fit_transform(data['similarities_complete'].detach().cpu().numpy())
    data['feature'] = torch.from_numpy(sim_reduced)
    return data




class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


def cal_acc(output, labels):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    return accuracy

def cls_loss(embeddings, data, split, margin, decoder_nc):
    idx = data[f'idx_{split}']
    output = decoder_nc.decode(embeddings, normalize(data['adj'], 'sym', True))
    output = output[idx]
    # print(data['labels'])
    # print(type(data['labels']))
    # print(idx)
    data_labels = torch.from_numpy(np.array(data['labels'])).to(output.device).to(torch.long)
    # print(data_labels, data_labels.shape)
    # print(data_labels[idx].unsqueeze(-1))
    # print(output.shape)
    correct = output.gather(1, data_labels[idx].unsqueeze(-1))
    loss = F.relu(margin - correct + output).mean()
    return loss

def cls_acc(embeddings, data, split, decoder_nc):
    idx = data[f'idx_{split}']
    output = decoder_nc.decode(embeddings, normalize(data['adj'], 'sym', True))
    output = output[idx]
    data_labels = torch.from_numpy(np.array(data['labels'])).to(output.device).to(torch.long)
    acc = cal_acc(output, data_labels[idx])
    return acc


class ExpNC:
    def __init__(self, configs):
        self.configs = configs
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

    def send_device(self, data):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        # data = load_data_nc(self.configs)
        # self.send_device(data)

        total_dp = []
        total_se = []
        total_da = []
        total_acc = []
        for exp_iter in range(self.configs.exp_iters):
            data = load_data_nc(self.configs)
            self.send_device(data)
            self.setup_seed(exp_iter)
            anchor_adj_raw = data['adj']

            anchor_adj = normalize(anchor_adj_raw, 'sym', self.configs.sparse)
            # anchor_adj = copy.deepcopy(anchor_adj_raw)

            if self.configs.type_learner == 'gnn':
                graph_learner = GNN_learner(2, data['num_features'], self.configs.gsl_k, self.configs.sim_function,6,
                                            self.configs.sparse, self.configs.activation_learner, anchor_adj).to(device)
                optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=self.configs.lr,weight_decay=self.configs.w_decay)
            elif self.configs.type_learner == 'mlp':
                graph_learner = MLP_learner(2, data['num_features'], self.configs.gsl_k, self.configs.sim_function,6,
                                            self.configs.sparse, self.configs.activation_learner).to(device)
                optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=self.configs.lr,weight_decay=self.configs.w_decay)
            else:
                raise NotImplementedError

            logger.info(f"\ntrain iters {exp_iter}")

            model = HYPCSE(embedder='LSENet',
                           in_features=data['num_features'],
                           hidden_dim_enc=self.configs.hidden_dim_enc,
                           # num_nodes=data['num_nodes'],
                           n_layers=self.configs.n_layers,
                           t2=self.configs.t2,
                           r2=self.configs.r2,
                           embed_dim=self.configs.embed_dim,
                           proj_dim=self.configs.proj_dim,
                           dropout=self.configs.dropout,
                           dropedge_rate=self.configs.dropedge_rate,
                           nonlin=self.configs.nonlin,
                           n_classes=data['num_classes']).to(device)
            optimizer = RiemannianAdam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
            if self.configs.task == 'HC':
                dp, se, da = self.train_hc(data, model, graph_learner, optimizer, optimizer_learner, anchor_adj, logger, device, exp_iter)
                total_dp.append(dp)
                total_se.append(se)
                total_da.append(da)
            elif self.configs.task == 'NC':
                acc = self.train_nc(data, model, graph_learner, optimizer, optimizer_learner, anchor_adj, logger)
                total_acc.append(acc)
            else:
                raise NotImplementedError

        if self.configs.task == 'HC':
            logger.info(f"DP: {np.mean(total_dp)}+-{np.std(total_dp)}, "
                        f"SE: {np.mean(total_se)}+-{np.std(total_se)}, "
                        f"DA: {np.mean(total_da)}+-{np.std(total_da)}")

            import json
            from datetime import datetime

            self.configs.dp = np.mean(total_dp)
            self.configs.se = np.mean(total_se)
            self.configs.da = np.mean(total_da)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"./results/{self.configs.dataset}/Time_{current_time}_DP_{self.configs.dp}.txt"
            with open(file_name, 'w') as file:
                json.dump(self.configs, file, indent=4)

            self.configs.total_dp = total_dp
            self.configs.total_se = total_se
            self.configs.total_da = total_da

            return np.mean(total_dp)
        elif self.configs.task == 'NC':
            logger.info(f"ACC: {np.mean(total_acc)}+-{np.std(total_acc)}")
            import json
            from datetime import datetime

            self.configs.acc = np.mean(total_acc)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"./results/{self.configs.dataset}/Time_{current_time}_ACC_{self.configs.acc}.txt"
            with open(file_name, 'w') as file:
                json.dump(self.configs, file, indent=4)

            self.configs.total_acc = total_acc

            return np.mean(total_acc)
        else:
            raise NotImplementedError

    def train_hc(self, data, model, graph_learner, optimizer, optimizer_learner, anchor_adj, logger, device, exp_iter):
        best_cluster_result = {}
        best_cluster = {'dp': 0, 'se': 1e12, 'da': 1e12}
        best_model = None
        for epoch in range(1, self.configs.epochs + 1):
            model.train()

            # view 1: anchor graph
            if self.configs.maskfeat_rate_anchor:
                mask_v1, _ = get_feat_mask(data['feature'], self.configs.maskfeat_rate_anchor)
                features_v1 = data['feature'] * (1 - mask_v1)
            else:
                features_v1 = copy.deepcopy(data['feature'])

            anchor_adj = anchor_adj.coalesce()
            z1, embeddings = model(features_v1, anchor_adj, 'anchor')

            # view 2: learned graph
            if self.configs.maskfeat_rate_learner:
                mask_v2, _ = get_feat_mask(data['feature'], self.configs.maskfeat_rate_learner)
                features_v2 = data['feature'] * (1 - mask_v2)
            else:
                features_v2 = copy.deepcopy(data['feature'])

            if self.configs.use_gl:
                learned_adj = graph_learner(data['feature'])
                learned_adj = learned_adj.coalesce()
            else:
                learned_adj = normalize(data['adj'], 'sym', self.configs.sparse)
                learned_adj = learned_adj.coalesce()
                self.configs.gsl_tau = 1.0

            z2, _ = model(features_v2, learned_adj, 'learner')

            # compute contrastive loss
            loss_contrastive = contrastive_loss_hyperbolic(z1, z2, self.configs.r3, self.configs.t3, model.manifold)


            loss_dist0 = model.dist0_loss(embeddings)
            loss_se = model.se_loss(embeddings, anchor_adj, se_sparse=self.configs.se_sparse, se_k=self.configs.se_k)

            loss = loss_se * self.configs.weight_loss_se + loss_contrastive * self.configs.weight_loss_contrastive \
                   + loss_dist0 * self.configs.weight_loss_dist0
            # loss = 0 * self.configs.weight_loss_se + loss_contrastive * self.configs.weight_loss_contrastive \
            #        + loss_dist0 * self.configs.weight_loss_dist0

            optimizer.zero_grad()
            optimizer_learner.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_learner.step()

            # Structure Bootstrapping
            if (1 - self.configs.gsl_tau):
                anchor_adj = anchor_adj * self.configs.gsl_tau + learned_adj * (1 - self.configs.gsl_tau)
                anchor_adj = anchor_adj.coalesce()
                anchor_adj = torch.sparse_coo_tensor(indices=anchor_adj.indices().detach(), values=anchor_adj.values().detach(),
                                                     size=anchor_adj.size(), is_coalesced=True).to(anchor_adj.device)

            if epoch % self.configs.eval_freq == 0:
                logger.info("-----------------------Evaluation Start---------------------")
                model.eval()
                decode_time = time.time()
                if self.configs.evaluate_adj == 'data_adj':
                    eval_adj = data['adj']
                    # embeddings = model.encode(data['feature'], data['adj'])
                elif self.configs.evaluate_adj == 'data_adj_normalized':
                    eval_adj = normalize(data['adj'], 'sym', self.configs.sparse)
                    # embeddings = model.encode(data['feature'], normalize(data['adj'], 'sym', self.configs.sparse))
                elif self.configs.evaluate_adj == 'anchor_adj':
                    eval_adj = anchor_adj
                    # embeddings = model.encode(data['feature'], anchor_adj)
                else:
                    raise NotImplementedError
                embeddings = model.encode(data['feature'], eval_adj)
                leaves_embeddings = model.manifold.to_poincare(embeddings)
                tree = model.decode_tree(leaves_embeddings, decoding_algo=self.configs.decoding_algo, n_cluster=data["num_classes"], fast_decoding=False)
                decode_time = time.time() - decode_time
                logger.info(f"Decoding cost time: {decode_time: .3f} s")
                trues = data['labels']

                # dp_recursive = den_purity_recursive(tree, trues)
                # print(dp, dp_recursive)
                # dp = dp_recursive
                if self.configs.eval_complete_graph:
                    similarities = data['similarities_complete'].to(torch.float64).detach().cpu().numpy()
                else:
                    similarities = data['similarities'].to(torch.float64).detach().cpu().numpy()
                try:
                    dp = den_purity(tree, trues)
                    se = se_cost(tree, similarities)
                    da = dasgupta_cost(tree, similarities)
                except nx.NetworkXError:
                    # print(traceback.format_exc())
                    dp = 0
                except Exception:
                    # print(traceback.format_exc())
                    dp = 0
                if dp > best_cluster['dp']:
                    best_cluster['dp'] = dp
                    best_cluster_result['dp'] = [dp, se, da]
                if se < best_cluster['se']:
                    best_cluster['se'] = se
                    best_cluster_result['se'] = [dp, se, da]
                    # best_model = model.state_dict()
                    best_adj = copy.deepcopy(eval_adj)
                    logger.info('------------------Saving best model----------------------')
                    torch.save(model.state_dict(), f"./checkpoints/{self.configs.dataset}/{self.configs.save_path}")
                if da < best_cluster['da']:
                    best_cluster['da'] = da
                    best_cluster_result['da'] = [dp, se, da]
                logger.info(f"Epoch {epoch}: DP: {dp}, SE: {se}, DA: {da}")
                logger.info(
                    "-------------------------------------------------------------------------")


        logger.info('------------------Loading best model-------------------')
        best_result_final = None
        model.load_state_dict(torch.load(f"./checkpoints/{self.configs.dataset}/{self.configs.save_path}"))
        model.eval()
        embeddings = model.encode(data['feature'], best_adj)
        leaves_embeddings = model.manifold.to_poincare(embeddings)

        tree = model.decode_tree(leaves_embeddings, decoding_algo=self.configs.decoding_algo,n_cluster=data["num_classes"], fast_decoding=False)

        similarities = data['similarities'].to(torch.float64).detach().cpu().numpy()
        trues = data['labels']
        try:
            dp = den_purity(tree, trues)
            se = se_cost(tree, similarities)
            da = dasgupta_cost(tree, similarities)
            best_result_final = [dp, se, da]
        except nx.NetworkXError:
            dp = 0
        except Exception:
            dp = 0

        for k, result in best_cluster_result.items():
            dp, se, da = result
            logger.info(f"Best Results according to {k}: DP: {dp}, SE: {se}, DA: {da}\n")
        if best_result_final is None:
            return best_cluster_result['se'][0], 0, 0
        else:
            dp, se, da = best_result_final
            logger.info(f"Best Results Final : DP: {dp}, SE: {se}, DA: {da}\n")
            return dp, se, da


    def train_nc(self, data, model, graph_learner, optimizer, optimizer_learner, anchor_adj, logger):
        best_acc = 0.0
        for epoch in range(1, self.configs.epochs + 1):
            model.train()

            # view 1: anchor graph
            if self.configs.maskfeat_rate_anchor:
                mask_v1, _ = get_feat_mask(data['feature'], self.configs.maskfeat_rate_anchor)
                features_v1 = data['feature'] * (1 - mask_v1)
            else:
                features_v1 = copy.deepcopy(data['feature'])

            anchor_adj = anchor_adj.coalesce()
            z1, embeddings = model(features_v1, anchor_adj, 'anchor')

            # view 2: learned graph
            if self.configs.maskfeat_rate_learner:
                mask_v2, _ = get_feat_mask(data['feature'], self.configs.maskfeat_rate_learner)
                features_v2 = data['feature'] * (1 - mask_v2)
            else:
                features_v2 = copy.deepcopy(data['feature'])

            learned_adj = graph_learner(data['feature'])
            learned_adj = learned_adj.coalesce()

            z2, _ = model(features_v2, learned_adj, 'learner')

            # compute contrastive loss
            loss_contrastive = contrastive_loss_hyperbolic(z1, z2, self.configs.r3, self.configs.t3, model.manifold)


            loss_dist0 = model.dist0_loss(embeddings)
            loss_se = model.se_loss(embeddings, anchor_adj, se_sparse=False, se_k=None)
            loss_cls = cls_loss(embeddings, data, 'train', self.configs.cls_margin, model.decoder_nc)

            loss = loss_se * self.configs.weight_loss_se + loss_contrastive * self.configs.weight_loss_contrastive \
                   + loss_dist0 * self.configs.weight_loss_dist0 + loss_cls * self.configs.weight_loss_cls

            optimizer.zero_grad()
            optimizer_learner.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_learner.step()

            if (1 - self.configs.gsl_tau):
                anchor_adj = anchor_adj * self.configs.gsl_tau + learned_adj * (1 - self.configs.gsl_tau)
                anchor_adj = anchor_adj.coalesce()
                anchor_adj = torch.sparse_coo_tensor(indices=anchor_adj.indices().detach(), values=anchor_adj.values().detach(),
                                                     size=anchor_adj.size(), is_coalesced=True).to(anchor_adj.device)

            if epoch % self.configs.eval_freq == 0:
                logger.info("-----------------------Evaluation Start---------------------")
                model.eval()
                decode_time = time.time()
                if self.configs.evaluate_adj == 'data_adj':
                    eval_adj = data['adj']
                    # embeddings = model.encode(data['feature'], data['adj'])
                elif self.configs.evaluate_adj == 'data_adj_normalized':
                    eval_adj = normalize(data['adj'], 'sym', self.configs.sparse)
                    # embeddings = model.encode(data['feature'], normalize(data['adj'], 'sym', self.configs.sparse))
                elif self.configs.evaluate_adj == 'anchor_adj':
                    eval_adj = anchor_adj
                    # embeddings = model.encode(data['feature'], anchor_adj)
                else:
                    raise NotImplementedError
                embeddings = model.encode(data['feature'], eval_adj)
                acc = cls_acc(embeddings, data, 'val', model.decoder_nc)
                if acc >= best_acc:
                    best_acc = acc
                    best_adj = copy.deepcopy(eval_adj)
                    logger.info('------------------Saving best model----------------------')
                    torch.save(model.state_dict(), f"./checkpoints/{self.configs.dataset}/{self.configs.save_path}")
                logger.info(f"Epoch {epoch}: ACC: {acc}")
                logger.info(
                    "-------------------------------------------------------------------------")

        logger.info('------------------Loading best model-------------------')
        model.load_state_dict(torch.load(f"./checkpoints/{self.configs.dataset}/{self.configs.save_path}"))
        model.eval()
        embeddings = model.encode(data['feature'], best_adj)
        acc = cls_acc(embeddings, data, 'test', model.decoder_nc)
        logger.info(f"Test ACC: ACC: {acc}\n")
        return acc

