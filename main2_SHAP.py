import torch
import torch.nn as nn
import shap
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
import sys
from sklearn.model_selection import StratifiedKFold, ParameterGrid
import torch.autograd
from torch.utils.data import DataLoader, TensorDataset
import csv
from tqdm import tqdm
import time
import seaborn as sns

from dataset import Datasets_SZ, Datasets_ASD, Datasets_AnxDep, Datasets_MCI, Datasets_AD, Datasets_PD
from model import Ourmutimodel

import warnings
warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='../')
    parser.add_argument('--datname', type=str, default='PD', help='SZ, ASD')  # for datasets
    parser.add_argument('--csv_name', type=str, default='v14_PD',
                        help='(v14_SZ_lr5e05, v14_ASD_lr2e05_sim3, v14_ASD_lr3e05)')
    parser.add_argument('--project', type=str, default='Promptlearning', help='wandb project name')

    parser.add_argument('--cuda', type=int, default=0, help='')
    parser.add_argument('--seed', type=int, default=0, help='random seed setting')
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--n_fold', type=int, default=5)
    parser.add_argument('--n_epoch', type=int, default=30, help='Max number of epochs to train.')

    parser.add_argument('--lr', type=float, default=0.00003)  # 0.001
    parser.add_argument('--outer', action='store_true', default=False, help="train only outer")

    parser.add_argument('--batch_size', type=int, default=32)  # 32
    parser.add_argument('--dropout', type=float, default=0.5)  # 0.5

    # FCN
    parser.add_argument('--fcn_layer', type=int, default=4)  # 3
    parser.add_argument('--fcn_head', type=int, default=1)
    # MSN
    parser.add_argument('--msn_layer', type=int, default=2)  # 1
    parser.add_argument('--msn_head', type=int, default=4)

    argv = parser.parse_args()
    return argv


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(model, fold, results_path, args, params):
    path = os.path.join(results_path, f'model_{fold}.pth')
    model.load_state_dict(torch.load(path, map_location=args.device), strict=True)
    return model

def hotmap(shap_values, fcn_feature_names, msn_feature_names, fcn_row_idx, fcn_col_idx, msn_row_idx, msn_col_idx, fold_outer):
    if hasattr(shap_values[1], 'cpu'):
        shap_vals = shap_values[1].cpu().numpy()
    else:
        shap_vals = shap_values[1]
    num_samples = shap_vals.shape[0]

    # fcn
    num_fcn_features = len(fcn_feature_names)
    # msn
    num_msn_features = len(msn_feature_names)

    # shap
    fcn_shap_vals = shap_vals[:, :num_fcn_features]
    msn_shap_vals = shap_vals[:, num_fcn_features:]

    # (num_fcn_features,)
    fcn_edge_importance = np.mean(np.abs(fcn_shap_vals), axis=0)
    msn_edge_importance = np.mean(np.abs(msn_shap_vals), axis=0)

    def build_heatmap_matrix(node_num, row_idx, col_idx, edge_values):
        mat = np.zeros((node_num, node_num))
        mat[row_idx, col_idx] = edge_values
        # mat[col_idx, row_idx] = edge_values
        return mat

    fcn_heatmap = build_heatmap_matrix(args.fcn_node, fcn_row_idx.numpy(), fcn_col_idx.numpy(), fcn_edge_importance)
    msn_heatmap = build_heatmap_matrix(args.msn_node, msn_row_idx.numpy(), msn_col_idx.numpy(), msn_edge_importance)

    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # sns.heatmap(fcn_heatmap, cmap="Reds", square=True)
    # plt.title(f'fold-{fold_outer} FCN SHAP Importance Heatmap')
    # plt.xlabel('Node index')
    # plt.ylabel('Node index')
    # plt.subplot(1, 2, 2)
    # sns.heatmap(msn_heatmap, cmap="Blues", square=True)
    # plt.title(f'fold-{fold_outer} MSN SHAP Importance Heatmap')
    # plt.xlabel('Node index')
    # plt.ylabel('Node index')
    # plt.tight_layout()
    # plt.show()

    return fcn_heatmap, msn_heatmap



args = parse()
set_rng_seed(args.seed)
args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
# args.device = torch.device('cpu')
print(f'use {args.device}')
args.data_path = os.path.join(args.dir, f'process/{args.datname}_Process/')

if args.datname == 'SZ': dataset = Datasets_SZ(args)
elif args.datname == 'ASD': dataset = Datasets_ASD(args)
elif args.datname == 'Anx': dataset = Datasets_AnxDep(args)
elif args.datname == 'Dep': dataset = Datasets_AnxDep(args)
elif args.datname == 'MCI': dataset = Datasets_MCI(args)
elif args.datname == 'AD': dataset = Datasets_AD(args)
elif args.datname == 'PD': dataset = Datasets_PD(args)
else: assert False

priori = dataset.priori_dict
args.fcn_node = dataset.fcn[0].shape[0]
args.msn_node = dataset.msn[0].shape[0]

results_path = os.path.join(args.dir, 'models', args.csv_name)
if not os.path.exists(results_path): os.mkdir(results_path)

param_grid = {
    'batch_size': [args.batch_size],  # 32
    'dropout': [args.dropout],  # 0.5
    # FCN
    'fcn_layer': [args.fcn_layer],  # 3
    'fcn_head': [args.fcn_head],
    # MSN
    'msn_layer': [args.msn_layer],  # 1
    'msn_head': [args.msn_head],
}

skf_outer = StratifiedKFold(n_splits=args.n_fold, random_state=args.seed, shuffle=True)
fcn_heatmap_all = []
msn_heatmap_all = []
for fold_outer, (train_val_idx, test_idx) in enumerate(skf_outer.split(dataset, dataset.labels)):
    if os.path.exists(os.path.join(results_path, f'shap_heatmap.npz')):
        shap_heatmap = np.load(os.path.join(results_path, f'shap_heatmap.npz'), allow_pickle=True)
        fcn_heatmap_all = shap_heatmap['fcn_heatmap_all']
        msn_heatmap_all = shap_heatmap['msn_heatmap_all']
        break
    train_val_dataset = torch.utils.data.Subset(dataset, train_val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    best_inner_params = ParameterGrid(param_grid)[0]
    if os.path.exists(os.path.join(results_path, f'dict_all.npy')):
        dict_all = np.load(os.path.join(results_path, f'dict_all.npy'), allow_pickle=True).item()
        best_inner_params = dict_all['best_inner_params'][fold_outer]

    train_loader = DataLoader(train_val_dataset, batch_size=1000, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    for i, multidata in enumerate(tqdm(train_loader, ncols=80, desc=f'outer-train fold:{fold_outer}')):
        fcnmsn_train = multidata['fcn-msn']
    for i, multidata in enumerate(tqdm(test_loader, ncols=80, desc=f'outer-test fold:{fold_outer}')):
        fcnmsn_test = multidata['fcn-msn']

    model = Ourmutimodel(args, best_inner_params, priori).to(args.device)
    model = load_model(model, fold_outer, results_path, args, best_inner_params).eval()

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            output = self.model(x)
            return output['Y_pred']


    # background_summary = shap.kmeans(fcnmsn_train, 100)
    # background_tensor = torch.tensor(background_summary.data, dtype=torch.float32).to(args.device)
    background_tensor = torch.tensor(fcnmsn_train, dtype=torch.float32).to(args.device)
    test_tensor = torch.tensor(fcnmsn_test, dtype=torch.float32).to(args.device)

    start_time = time.time()
    wrapped_model = ModelWrapper(model)
    explainer = shap.GradientExplainer(wrapped_model, background_tensor)
    if os.path.exists(os.path.join(results_path, f'shap_values_{fold_outer}.npz')):
        print(f'shap_values_{fold_outer}.npz already exists, skip')
        shap_values = np.load(os.path.join(results_path, f'shap_values_{fold_outer}.npz'))['shap_values']
    else:
        shap_values = explainer.shap_values(test_tensor, rseed=args.seed)
    end_time = time.time()
    print(f"time: {end_time - start_time:.2f} 秒")


    fcn_row_idx, fcn_col_idx = torch.triu_indices(args.fcn_node, args.fcn_node, offset=1)
    msn_row_idx, msn_col_idx = torch.triu_indices(args.msn_node, args.msn_node, offset=1)
    fcn_feature_names = [f'FCN_{i+1}-{j+1}' for i, j in zip(fcn_row_idx, fcn_col_idx)]
    msn_feature_names = [f'MSN_{i+1}-{j+1}' for i, j in zip(msn_row_idx, msn_col_idx)]
    feature_names = fcn_feature_names + msn_feature_names

    fcn_heatmap, msn_heatmap = hotmap(shap_values,
                                      fcn_feature_names,
                                      msn_feature_names,
                                      fcn_row_idx,
                                      fcn_col_idx,
                                      msn_row_idx,
                                      msn_col_idx,
                                      fold_outer)
    fcn_heatmap_all.append(fcn_heatmap)
    msn_heatmap_all.append(msn_heatmap)

    if not os.path.exists(os.path.join(results_path, f'shap_values_{fold_outer}.npz')):
        np.savez(os.path.join(results_path, f'shap_values_{fold_outer}.npz'), shap_values=shap_values)

fcn_heatmap_all = np.array(fcn_heatmap_all)
msn_heatmap_all = np.array(msn_heatmap_all)
# save
np.savez(os.path.join(results_path, f'shap_heatmap.npz'), fcn_heatmap_all=fcn_heatmap_all, msn_heatmap_all=msn_heatmap_all)

# %% biobert_biomarkers_Gem25FP0520_sim

def extract_top_values(matrix, top_n):
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "error"
    n = matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    triu_values = matrix[triu_idx]
    threshold = np.partition(triu_values, -top_n)[-top_n]
    mask = np.zeros_like(matrix, dtype=int)
    mask_idx = (triu_values >= threshold)
    mask[triu_idx[0][mask_idx], triu_idx[1][mask_idx]] = 1
    return threshold, mask



top_n = 100
sim_fMRI_regions_norm = dataset.priori_dict['sim_fMRI_regions_norm'].numpy()
sim_fMRI_conn_norm = dataset.priori_dict['sim_fMRI_conn_norm'].numpy()  # 124
sim_sMRI_regions_norm = dataset.priori_dict['sim_sMRI_regions_norm'].numpy()
sim_sMRI_conn_norm = dataset.priori_dict['sim_sMRI_conn_norm'].numpy()  # 438
# threshold_fcn, mask_fcn_prior = extract_top_values(sim_fMRI_conn_norm, 53*(53-1)//2)
threshold_fcn, mask_fcn_prior = extract_top_values(sim_fMRI_conn_norm, top_n)
# threshold_msn, mask_msn_prior = extract_top_values(sim_sMRI_conn_norm, 116*(116-1)//2)
threshold_msn, mask_msn_prior = extract_top_values(sim_sMRI_conn_norm, top_n)
mask_fcn_prior = mask_fcn_prior * sim_fMRI_conn_norm
mask_msn_prior = mask_msn_prior * sim_sMRI_conn_norm


mask_fcn_all = []
mask_msn_all = []
for fold_outer in range(5):
    threshold_fcn, mask_fcn = extract_top_values(fcn_heatmap_all[fold_outer], top_n)
    threshold_msn, mask_msn = extract_top_values(msn_heatmap_all[fold_outer], top_n)
    mask_fcn_all.append(mask_fcn)
    mask_msn_all.append(mask_msn)


mask_fcn_all = np.stack(mask_fcn_all)
mask_msn_all = np.stack(mask_msn_all)
# mask_fcn = np.prod(mask_fcn_all, axis=0)
mask_fcn = np.sum(mask_fcn_all, axis=0)
# mask_msn = np.prod(mask_msn_all, axis=0)
mask_msn = np.sum(mask_msn_all, axis=0)

# %% plot

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# FCN
sns.heatmap(mask_fcn_prior, cmap="Reds", square=True, ax=axes[0, 0], cbar=True)
axes[0, 0].set_title('FCN prior')
axes[0, 0].set_xlabel('Node index')
axes[0, 0].set_ylabel('Node index')

sns.heatmap(mask_fcn, cmap="Reds", square=True, ax=axes[0, 1], cbar=True)
axes[0, 1].set_title('FCN SHAP Importance Heatmap')
axes[0, 1].set_xlabel('Node index')
axes[0, 1].set_ylabel('Node index')

sns.heatmap(mask_fcn * (mask_fcn_prior>0), cmap="Reds", square=True, ax=axes[0, 2], cbar=True)
axes[0, 2].set_title('FCN SHAP Importance * Prior')
axes[0, 2].set_xlabel('Node index')
axes[0, 2].set_ylabel('Node index')

# MSN
sns.heatmap(mask_msn_prior, cmap="Blues", square=True, ax=axes[1, 0], cbar=True)
axes[1, 0].set_title('MSN prior')
axes[1, 0].set_xlabel('Node index')
axes[1, 0].set_ylabel('Node index')

sns.heatmap(mask_msn, cmap="Blues", square=True, ax=axes[1, 1], cbar=True)
axes[1, 1].set_title('MSN SHAP Importance Heatmap')
axes[1, 1].set_xlabel('Node index')
axes[1, 1].set_ylabel('Node index')

sns.heatmap(mask_msn * (mask_msn_prior>0), cmap="Blues", square=True, ax=axes[1, 2], cbar=True)
axes[1, 2].set_title('MSN SHAP Importance * Prior')
axes[1, 2].set_xlabel('Node index')
axes[1, 2].set_ylabel('Node index')

plt.tight_layout()
plt.show()

print(f'mask_fcn_prior: {(mask_fcn_prior>0).sum()}, mask_msn_prior: {(mask_msn_prior>0).sum()}')
# %% save mat for matlab

from scipy.io import loadmat, savemat

mat_dict = {
    'fcn_heatmap_all': fcn_heatmap_all.astype('float64'),  # [5, 53, 53] shap resize triu
    'msn_heatmap_all': msn_heatmap_all.astype('float64'),

    'sim_fMRI_regions_norm': sim_fMRI_regions_norm.astype('float64'),  # [53] prior
    'sim_fMRI_conn_norm': sim_fMRI_conn_norm.astype('float64'),  # [53, 53] prior
    'sim_sMRI_regions_norm': sim_sMRI_regions_norm.astype('float64'),
    'sim_sMRI_conn_norm': sim_sMRI_conn_norm.astype('float64'),

    'mask_fcn_prior': mask_fcn_prior.astype('float64'),  # [53, 53] prior triu
    'mask_msn_prior': mask_msn_prior.astype('float64'),

    'mask_fcn_all': mask_fcn_all.astype('float64'),  # [5, 53, 53] top-n triu
    'mask_msn_all': mask_msn_all.astype('float64'),
    'mask_fcn': mask_fcn.astype('float64'),  # [53, 53] top-n triu
    'mask_msn': mask_msn.astype('float64'),
}

save_path = os.path.join(results_path, f'shap_analysis_results_bothbigtop{top_n}_{args.datname}.mat')
savemat(save_path, mat_dict)
print(f"Results saved to {save_path}")


