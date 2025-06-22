
import torch
import numpy as np
import os
import random
import argparse
# from exp import Exp
# from exp_gsl import Exp_Gsl
# from exp_scalable import ExpScalable
from node_classification.exp_nc import ExpNC
from logger import create_logger
import json
from utils.train_utils import DotDict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# seed = 3047
# seed = 1
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)


def sim_cls(configs):
    configs_dict = vars(configs)
    print(f'./configs/{configs.dataset}.json')
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

    configs.task = 'NC'
    configs.cls_margin = 1.
    configs.weight_loss_cls = 1.
    configs.save_path = "model_nc.pt"

    logger.info(configs)

    exp = ExpNC(configs)
    exp.train()
    exp_configs = exp.configs
    torch.cuda.empty_cache()
    return exp_configs



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Hyperbolic Hierarchical Clustering via Structural Entropy')

    parser.add_argument('--dataset', type=str, default='Zoo')
    parser.add_argument('--root_path', type=str, default='./datasets')
    parser.add_argument('--task', type=str, default='HC', choices=['LP', 'HC'])
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--exp_iters', type=int, default=5)
    parser.add_argument('--version', type=str, default="run")

    parser.add_argument('--pre_epochs', type=int, default=200, help='the training epochs for pretraining')
    parser.add_argument('--epochs', type=int, default=2000)
    # parser.add_argument('--lr_pre', type=float, default=5e-3)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--w_decay', type=float, default=3e-2)
    parser.add_argument('--decay_rate', type=float, default=None)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_dim_enc', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--nonlin', type=str, default=None)
    # parser.add_argument('--temperature', type=float, default=0.05) # now use t2 to replace this hyperparamter temperature
    parser.add_argument('--n_cluster_trials', type=int, default=5)
    parser.add_argument('--t', type=float, default=1., help='for Fermi-Dirac decoder')
    parser.add_argument('--r', type=float, default=2., help='Fermi-Dirac decoder')
    parser.add_argument('--t2', type=float, default=1.,
                        help='temperature for the scaled softmax function in se loss C^z')
    parser.add_argument('--r2', type=float, default=2., help='for the scaled softmax function in se loss C^z')

    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--save_path', type=str, default='model.pt')
    # GPU
    parser.add_argument('--use_gpu', action='store_false', help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0,1',
                        help='device ids of multiple gpus')

    parser.add_argument('--sparse', type=bool, default=True)

    # gsl
    parser.add_argument('--gsl_mode', type=str, default='structure_refinement')
    parser.add_argument('--gsl_k', type=int, default=5)
    parser.add_argument('--type_learner', type=str, default='gnn', choices='gnn, mlp')
    parser.add_argument('--sim_function', type=str, default='cosine', choices=['cosine', 'minkowski', 'gaussian'])
    parser.add_argument('--activation_learner', type=str, default='relu', choices=['relu', 'tanh'])

    parser.add_argument('--maskfeat_rate_learner', type=float, default=0.05)
    parser.add_argument('--maskfeat_rate_anchor', type=float, default=0.05)
    parser.add_argument('--dropedge_rate', type=float, default=0.1)

    parser.add_argument('--contrast_batch_size', type=int, default=0)

    parser.add_argument('--gsl_tau', type=float, default=0.999)
    parser.add_argument('--gsl_c', type=int, default=0)
    parser.add_argument('--t3', type=float, default=100.0)
    parser.add_argument('--r3', type=float, default=0.0)

    parser.add_argument('--proj_dim', type=int, default=64)

    # loss weights
    parser.add_argument('--weight_loss_se', type=float, default=1.)
    parser.add_argument('--weight_loss_contrastive', type=float, default=1.)
    parser.add_argument('--weight_loss_dist0', type=float, default=1.)
    parser.add_argument('--weight_loss_lp', type=float, default=0.)

    # try contrastive loss to pretrain, or no pretrain.
    parser.add_argument('--pretrain_method', type=str, default='contrastive',
                        choices=['link_prediction', 'contrastive'])
    # The projector may connect to the layer before the last embedding layer.
    # Add an additional embedding layer to achieve this. This additioanl embedding can be 2 dim (2+1 dim in Lorentz space), good for visualization.
    parser.add_argument('--additional_embed_dim', type=int, default=0)

    parser.add_argument('--model_choosing', type=str, default='static', choices=['adaptive', 'static'])

    parser.add_argument('--gc_type', type=str, default='gaussian_static',
                        choices=['gaussian_static', 'gaussian_adaptive'])

    parser.add_argument('--evaluate_adj', type=str, default='anchor_adj',
                        choices=['data_adj', 'anchor_adj', 'data_adj_normalized'])

    parser.add_argument('--scalable', type=bool, default=False)

    parser.add_argument('--se_sparse', type=bool, default=True)
    parser.add_argument('--se_k', type=int, default=None)
    parser.add_argument('--eval_complete_graph', type=bool, default=False)
    parser.add_argument('--decoding_algo', type=str, default=None)

    parser.add_argument('--cls_margin', type=float, default=1., required=False)
    parser.add_argument('--weight_loss_cls', type=float, default=1., required=False)

    # for dataset in ['Zoo', 'Iris', 'Glass', 'Segmentation', 'Spambase']:

    configs = parser.parse_args()

    # for dataset in ['Zoo', 'Iris', 'Glass', 'Segmentation']:
    for dataset in ['Zoo']:
        result_path = f"./{dataset}.txt"
        configs.dataset = dataset
        exp_configs = sim_cls(configs)
        ACCs = exp_configs.total_acc
        with open(result_path, 'w') as f:
            f.write("ACC:\t")
            for i in range(len(ACCs)):
                f.write("{}\t".format(ACCs[i]))
            f.write("average:\t{}\t".format(np.mean(ACCs)))
            f.write("std:\t{}\n".format(np.std(ACCs)))
        print(np.mean(ACCs))


