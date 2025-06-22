import torch
import numpy as np
import os
import random
import argparse
from exp_sampling import ExpSampling
from logger import create_logger
import json
from utils.train_utils import DotDict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# seed = 3047
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='Hyperbolic Hierarchical Clustering via Structural Entropy')

parser.add_argument('--dataset', type=str, default='Zoo')
parser.add_argument('--root_path', type=str, default='./datasets')
parser.add_argument('--eval_freq', type=int, default=1)

parser.add_argument('--pre_epochs', type=int, default=200, help='the training epochs for pretraining')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--w_decay', type=float, default=5e-4)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--embed_dim', type=int, default=64)
parser.add_argument('--hidden_dim_enc', type=int, default=256)
parser.add_argument('--n_cluster_trials', type=int, default=5)
parser.add_argument('--t', type=float, default=1., help='for Fermi-Dirac decoder')
parser.add_argument('--r', type=float, default=2., help='Fermi-Dirac decoder')
parser.add_argument('--t2', type=float, default=1., help='temperature for the scaled softmax function in se loss C^z')
parser.add_argument('--r2', type=float, default=2., help='for the scaled softmax function in se loss C^z')

parser.add_argument('--save_path', type=str, default='model.pt')

parser.add_argument('--sparse', type=bool, default=True)

parser.add_argument('--gsl_k', type=int, default=15)
parser.add_argument('--type_learner', type=str, default='gnn', choices='gnn, mlp')
parser.add_argument('--sim_function', type=str, default='cosine', choices=['cosine', 'minkowski', 'gaussian'])
parser.add_argument('--activation_learner', type=str, default='relu', choices=['relu', 'tanh'])

parser.add_argument('--maskfeat_rate_learner', type=float, default=0.05)
parser.add_argument('--maskfeat_rate_anchor', type=float, default=0.05)
parser.add_argument('--dropedge_rate', type=float, default=0.3)

parser.add_argument('--gsl_tau', type=float, default=0.9999)
parser.add_argument('--t3', type=float, default=100.0)
parser.add_argument('--r3', type=float, default=0.0)
parser.add_argument('--proj_dim', type=int, default=16)

# loss weights
parser.add_argument('--weight_loss_se', type=float, default=1.)
parser.add_argument('--weight_loss_contrastive', type=float, default=1.)
parser.add_argument('--weight_loss_dist0', type=float, default=1.)
parser.add_argument('--evaluate_adj', type=str, default='anchor_adj', choices=['data_adj', 'anchor_adj', 'data_adj_normalized'])

parser.add_argument('--scalable', type=bool, default=False)
parser.add_argument('--eval_complete_graph', type=bool, default=False)
parser.add_argument('--decoding_algo', type=str, default=None)

# for neighbor sampling
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--n_seeds', type=int, default=10)



configs = parser.parse_args()

configs_dict = vars(configs)
with open(f'./configs/{configs.dataset}.json', 'rt') as f:
    configs_dict.update(json.load(f))
configs = DotDict(configs_dict)
f.close()

log_path = f"./results/{configs.version}/{configs.dataset}.log"
configs.log_path = log_path
if not os.path.exists(f"./results"):
    os.mkdir("./results")
if not os.path.exists(f"./results/{configs.dataset}"):
    os.mkdir(f"./results/{configs.dataset}")
if not os.path.exists(f"./checkpoints/{configs.dataset}"):
    os.mkdir(f"./checkpoints/{configs.dataset}")
if not os.path.exists(f"./results/{configs.version}"):
    os.mkdir(f"./results/{configs.version}")
print(f"Log path: {configs.log_path}")
logger = create_logger(configs.log_path)
logger.info(configs)

exp = ExpSampling(configs)
exp.train()
torch.cuda.empty_cache()

DPs = configs.total_dp
SEs = configs.total_se
DAs = configs.total_da

