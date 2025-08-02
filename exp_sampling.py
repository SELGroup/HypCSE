import copy
import time

import networkx as nx
import torch
from logger import create_logger
from models.vec_dataset import load_data
from models.model import HYPCSE
from geoopt.optim import RiemannianAdam
import numpy as np
from utils.metrics import dasgupta_cost, se_cost, den_purity_recursive, den_purity, se_cost_iterative
import random
from gsl.gsl_utils import torch_sparse_eye, normalize, get_feat_mask, contrastive_loss_hyperbolic
from gsl.graph_learners import GNN_learner, MLP_learner, GNN_learner_adj
import traceback
from torch_geometric.loader import NeighborLoader
import torch_geometric
from gsl.gsl_utils import update_data_prob_batch, update_data_prob
from torch_geometric.utils import to_undirected, remove_self_loops
from gsl.gsl_utils import sampling_neighbor
from torch_geometric.utils import subgraph

class ExpSampling:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def setup_seed(self, seed):
        if "seeds" in self.configs.keys():
            seed = self.configs['seeds'][seed]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

    def send_device(self, data):
        # data.to(self.device)
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        data, data_dict = load_data(self.configs)
        self.send_device(data_dict)
        data = data.to(device)

        total_dp = []
        total_se = []
        total_da = []
        for exp_iter in range(self.configs.exp_iters):
            self.setup_seed(exp_iter)

            if self.configs.type_learner == 'gnn':
                graph_learner = GNN_learner_adj(2, data_dict["num_features"], self.configs.gsl_k, self.configs.sim_function,6,
                                            self.configs.sparse, self.configs.activation_learner).to(device)
                optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=self.configs.lr,weight_decay=self.configs.w_decay)
            elif self.configs.type_learner == 'mlp':
                graph_learner = MLP_learner(2, data_dict["num_features"], self.configs.gsl_k, self.configs.sim_function,6,
                                            self.configs.sparse, self.configs.activation_learner).to(device)
                optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=self.configs.lr,weight_decay=self.configs.w_decay)
            else:
                raise NotImplementedError

            logger.info(f"\ntrain iters {exp_iter}")

            model = HYPCSE(embedder='LSENet',
                           in_features=data_dict["num_features"],
                           hidden_dim_enc=self.configs.hidden_dim_enc,
                           n_layers=self.configs.n_layers,
                           t2=self.configs.t2,
                           r2=self.configs.r2,
                           embed_dim=self.configs.embed_dim,
                           proj_dim=self.configs.proj_dim,
                           dropout=self.configs.dropout,
                           dropedge_rate=self.configs.dropedge_rate,
                           nonlin=self.configs.nonlin, ).to(device)
            optimizer = RiemannianAdam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
            if self.configs.task == 'HC':
                dp, se, da = self.train_hc(data, data_dict, model, graph_learner, optimizer, optimizer_learner,
                                           logger, device, exp_iter)
                total_dp.append(dp)
                total_se.append(se)
                total_da.append(da)
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
        else:
            raise NotImplementedError

    def train_hc(self, data, data_dict, model, graph_learner, optimizer, optimizer_learner, logger, device, exp_iter):
        best_cluster_result = {}
        best_cluster = {'dp': 0, 'se': 1e12, 'da': 1e12}
        best_model = None
        for epoch in range(1, self.configs.epochs + 1):
            model.train()
            assert torch.min(data_dict['degrees'])>0

            subgraph_list = sampling_neighbor(data, self.configs.n_seeds, self.configs.batch_size, training_all_nodes=True)

            anchor_remove_e_id_list = []
            learner_keep_edges_list = []
            learner_keep_values_list = []
            total_loss = 0
            for batch_index, batch in enumerate(subgraph_list):


                batch_edge_index, batch_edge_weight = batch.edge_index, batch.edge_attr
                anchor_adj = torch.sparse_coo_tensor(indices=batch_edge_index, values=batch_edge_weight,
                                                     size=[batch.y.shape[0], batch.y.shape[0]])
                anchor_adj = normalize(anchor_adj, 'sym', self.configs.sparse).coalesce()
                # exit(0)
                anchor_adj = torch.sparse_coo_tensor(indices=anchor_adj.indices().detach(),
                                                     values=anchor_adj.values().detach(),
                                                     size=anchor_adj.size(), is_coalesced=True).to(anchor_adj.device)

                # view 1: anchor graph
                if self.configs.maskfeat_rate_anchor:
                    mask_v1, _ = get_feat_mask(batch.x, self.configs.maskfeat_rate_anchor)
                    features_v1 = batch.x * (1 - mask_v1)
                else:
                    features_v1 = copy.deepcopy(batch.x)

                z1, embeddings = model(features_v1, anchor_adj, 'anchor')


                # view 2: learner graph
                if self.configs.maskfeat_rate_learner:
                    mask_v2, _ = get_feat_mask(batch.x, self.configs.maskfeat_rate_learner)
                    features_v2 = batch.x * (1 - mask_v2)
                else:
                    features_v2 = copy.deepcopy(batch.x)

                batch_n_id = batch.n_id
                original_edge_index, original_edge_weight = subgraph(edge_index=data_dict["edge_index"],
                                                                     edge_attr=data_dict["edge_weight"],
                                                                     subset=batch_n_id,
                                                                     relabel_nodes=True, return_edge_mask=False)
                original_adj = torch.sparse_coo_tensor(indices=original_edge_index, values=original_edge_weight,
                                                       size=(batch_n_id.shape[0], batch_n_id.shape[0])).to(anchor_adj.device)
                original_adj = normalize(original_adj, "sym", self.configs.sparse).coalesce()
                learner_adj = graph_learner(batch.x, original_adj).coalesce()
                learner_adj = normalize(learner_adj, 'sym', self.configs.sparse).coalesce()

                z2, _ = model(features_v2, learner_adj, 'learner')

                loss_contrastive = contrastive_loss_hyperbolic(z1, z2, self.configs.r3, self.configs.t3, model.manifold)
                loss_dist0 = model.dist0_loss(embeddings)
                loss_se = model.se_loss(embeddings, anchor_adj, se_sparse=False, se_k=None)
                loss = loss_se * self.configs.weight_loss_se + loss_contrastive * self.configs.weight_loss_contrastive \
                       + loss_dist0 * self.configs.weight_loss_dist0

                optimizer.zero_grad()
                optimizer_learner.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_learner.step()

                total_loss += float(loss) * self.configs.batch_size

                # structure updating
                if (1 - self.configs.gsl_tau) :
                    anchor_remove_e_id, learner_keep_edges, learner_keep_values = update_data_prob_batch(batch, learner_adj, self.configs.gsl_tau)
                    anchor_remove_e_id_list.append(anchor_remove_e_id)
                    learner_keep_edges_list.append(learner_keep_edges)
                    learner_keep_values_list.append(learner_keep_values)

            if epoch % self.configs.eval_freq == 0:
                if self.configs.eval_batch:
                    embeddings = self.encode_batch(data, data_dict, model, logger)
                else:
                    logger.info("-----------------------Evaluation Start---------------------")
                    model.eval()
                    if self.configs.evaluate_adj == 'data_adj':
                        eval_adj = data_dict['adj'].coalesce()
                    elif self.configs.evaluate_adj == 'data_adj_normalized':
                        eval_adj = normalize(data_dict['adj'], 'sym', self.configs.sparse).coalesce()
                    elif self.configs.evaluate_adj == 'anchor_adj':
                        eval_adj = torch.sparse_coo_tensor(indices=data.edge_index, values=data.edge_attr,
                                                           size=[data.y.shape[0], data.y.shape[0]]).coalesce()
                    else:
                        raise NotImplementedError
                    embeddings = model.encode(data_dict['feature'], eval_adj)
                leaves_embeddings = model.manifold.to_poincare(embeddings)
                trues = data_dict["labels"]

                decode_time = time.time()
                if self.configs.eval_complete_graph:
                    similarities = data_dict["similarities_complete"].to(torch.float64).detach().cpu().numpy()
                else:
                    similarities = data_dict["similarities"].to(torch.float64).detach().cpu().numpy()
                tree = model.decode_tree(leaves_embeddings, decoding_algo=self.configs.decoding_algo,
                                         n_cluster=data_dict["num_classes"], fast_decoding=False)
                decode_time = time.time() - decode_time
                logger.info(f"Decoding cost time: {decode_time: .3f} s")

                try:
                    dp = den_purity(tree, trues)
                    se = se_cost(tree, similarities)
                    da = dasgupta_cost(tree, similarities)
                except nx.NetworkXError:
                    print(traceback.format_exc())
                    dp = 0
                except Exception:
                    print(traceback.format_exc())
                    dp = 0
                if dp > best_cluster['dp']:
                    best_cluster['dp'] = dp
                    best_cluster_result['dp'] = [dp, se, da]
                if se < best_cluster['se']:
                    best_cluster['se'] = se
                    best_cluster_result['se'] = [dp, se, da]
                    self.leaves_embeddings = leaves_embeddings
                if da < best_cluster['da']:
                    best_cluster['da'] = da
                    best_cluster_result['da'] = [dp, se, da]
                logger.info(f"Epoch {epoch}: DP: {dp}, SE: {se}, DA: {da}")

            if (1 - self.configs.gsl_tau):
                data = update_data_prob(data, anchor_remove_e_id_list, learner_keep_edges_list, learner_keep_values_list)
                print("num_edges", data.edge_index.shape[1])

        model.eval()
        decode_time = time.time()
        similarities = data_dict["similarities"].to(torch.float64).detach().cpu().numpy()
        trues = data_dict["labels"]
        self.labels = trues
        tree = model.decode_tree(self.leaves_embeddings, decoding_algo=self.configs.decoding_algo,
                                 n_cluster=data_dict["num_classes"], fast_decoding=False)
        decode_time = time.time() - decode_time
        logger.info(f"Decoding cost time: {decode_time: .3f} s")

        self.tree = tree
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
