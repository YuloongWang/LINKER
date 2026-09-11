import os
import random
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from einops import rearrange, repeat




def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z, dim=1):
        w = self.project(z)
        beta = torch.softmax(w, dim=dim)
        return (beta * z).sum(1), beta


class OurMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(OurMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)  # 同时生成 Q, K, V
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.attn_readout = Attention(embed_dim, embed_dim//2)

    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: [batch_size, seq_len, embed_dim]
            key:   [batch_size, seq_len, embed_dim]
            value: [batch_size, seq_len, embed_dim]
            attn_mask: [batch_size, num_heads, seq_len, seq_len]
        """
        B, T, C = query.size()  # batch_size, seq_len, embed_dim

        # Linear projection and split into heads
        # q, k, v = self.qkv_proj(query).split(self.embed_dim, dim=-1)
        q = self.qkv_proj(query)[:, :, :self.embed_dim]  # Q from query
        k = self.qkv_proj(key)[:, :, self.embed_dim:2 * self.embed_dim]  # K from key
        v = self.qkv_proj(value)[:, :, 2 * self.embed_dim:]  # V from value

        # Reshape and transpose for multi-head attention: [B, T, embed_dim] -> [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
        q = q.view(B, query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        # if not self.training:
        #     print(attn_weights.max(), attn_weights.mean(), end='   ')

        if attn_mask is None:
            # softmax
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            # Apply attention to values
            attn_output = attn_weights @ v  # [B, num_heads, T, head_dim]
            # Concatenate heads and apply final linear projection
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
            # attn_output = self.out_dropout(self.out_proj(attn_output))
            attn_output = self.out_proj(attn_output)
            return attn_output, {'attn_weights': attn_weights, 'attn_weights_node': attn_weights, 'attn_weights_edge': attn_weights}
        else:
            attn_node = attn_mask['node'].expand(B, self.num_heads, T, 1)
            attn_edge = attn_mask['edge'].expand(B, self.num_heads, T, T)
            # add mask
            attn_weight1d = rearrange(attn_weights, 'b h t1 t2 -> (b h) t1 t2')
            attn_weight1d = F.max_pool1d(attn_weight1d, kernel_size=T, stride=1)
            attn_weight1d = rearrange(attn_weight1d, '(b h) t1 1 -> b h t1 1', h=self.num_heads)
            attn_weights_node = (attn_weight1d * attn_node).squeeze(-1)
            # attn_weights_node = attn_node.squeeze(-1)

            # attn_weights_node = attn_node * attn_weights
            attn_weights_edge = attn_edge * attn_weights
            # softmax
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_weights_node = F.softmax(attn_weights_node, dim=-1)
            attn_weights_node = self.attn_dropout(attn_weights_node)
            attn_weights_edge = F.softmax(attn_weights_edge, dim=-1)
            attn_weights_edge = self.attn_dropout(attn_weights_edge)

            # Apply attention to values
            attn_output = attn_weights @ v  # [B, num_heads, T, head_dim]
            # attn_output_node = attn_weights_node @ v
            attn_output_node = attn_weights_node.unsqueeze(-1) * v
            attn_output_edge = attn_weights_edge @ v
            # attn_output_quary = attn_weights_quary @ v
            # Concatenate heads and apply final linear projection
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
            attn_output_node = attn_output_node.transpose(1, 2).contiguous().view(B, T, C)
            attn_output_edge = attn_output_edge.transpose(1, 2).contiguous().view(B, T, C)
            # attn_output_quary = attn_output_quary.transpose(1, 2).contiguous().view(B, T, C)
            # attn_output = self.out_dropout(self.out_proj(attn_output))
            attn_output = self.out_proj(attn_output)
            attn_output_node = self.out_proj(attn_output_node)
            attn_output_edge = self.out_proj(attn_output_edge)
            # attn_output_quary = self.out_proj(attn_output_quary)

            # attn_output = attn_output + attn_output_node + attn_output_edge

            attn_cat = torch.stack([attn_output, attn_output_node, attn_output_edge], dim=1)
            attn_output, _ = self.attn_readout(attn_cat)

            return attn_output, {'attn_weights': attn_weights, 'attn_weights_node': attn_weights_node, 'attn_weights_edge': attn_weights_edge}


class PostNormTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.multihead_attn = OurMultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4, bias=False),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim*4, hidden_dim, bias=False))

    def forward(self, x_trans, mask=None):
        # Post-Norm for Multihead Attention
        x_attend, attn_matrix = self.multihead_attn(x_trans, x_trans, x_trans, attn_mask=mask)
        x_trans = x_trans + self.dropout1(x_attend)  # Residual connection
        x_trans = self.layer_norm1(x_trans)  # Layer Normalization after residual connection
        # Post-Norm for Feed-Forward Network
        x_ffn = self.mlp(x_trans)
        x_trans = x_trans + self.dropout2(x_ffn)  # Residual connection
        x_trans = self.layer_norm2(x_trans)  # Layer Normalization after residual connection
        return x_trans, attn_matrix

        # # Pre-Norm for Multihead Attention
        # x_norm1 = self.layer_norm1(x_trans)
        # x_attend, attn_matrix = self.multihead_attn(x_norm1, x_norm1, x_norm1, attn_mask=mask)
        # x_trans = x_trans + self.dropout1(x_attend)  # Residual connection
        # # Pre-Norm for Feed-Forward Network
        # x_norm2 = self.layer_norm2(x_trans)
        # x_ffn = self.mlp(x_norm2)
        # x_trans = x_trans + self.dropout2(x_ffn)  # Residual connection
        # return x_trans, attn_matrix


class Ourmodel(nn.Module):
    def __init__(self, num_classes, num_layers, num_nodes, num_heads, hidden_dim, dropout, priori_node, priori_edge):
        super(Ourmodel, self).__init__()
        self.dropout = dropout
        if (hidden_dim % num_heads) == 0: self.hidden_dim = hidden_dim
        else: self.hidden_dim = hidden_dim - (hidden_dim % num_heads) + num_heads
        self.mask = {'node': priori_node, 'edge': priori_edge}
        self.pos_embed = nn.Parameter(torch.zeros(1, num_nodes, self.hidden_dim))
        self.model_trans = nn.ModuleList()
        # self.cls_attn = nn.ModuleList()
        self.cls_mlp = nn.ModuleList()
        for i in range(num_layers):
            self.model_trans.append(PostNormTransformer(hidden_dim=self.hidden_dim, num_heads=num_heads, dropout=dropout))
            self.cls_mlp.append(nn.Sequential(nn.Linear(num_nodes * self.hidden_dim, (num_nodes * self.hidden_dim) // 2, bias=False),
                                                nn.ReLU(), nn.Dropout(dropout),
                                                nn.Linear((num_nodes * self.hidden_dim) // 2, 512, bias=False),
                                                nn.ReLU(), nn.Dropout(dropout),
                                                nn.Linear(512, num_classes, bias=False),))
            # self.cls_attn.append(Attention(in_size=hidden_dim))
            # self.cls_mlp.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            #                                     nn.ReLU(), nn.Dropout(dropout),
            #                                     nn.Linear(hidden_dim // 2, num_classes, bias=False),))

    def forward(self, x):
        batch_size, n, t = x.shape

        logit = 0.0
        attn_matrix_merge = []
        if t != self.hidden_dim: x = torch.cat([x, torch.zeros_like(x[..., :(self.hidden_dim-t)])], dim=-1)
        x = x + self.pos_embed
        for i in range(len(self.model_trans)):
            x, attn_matrix = self.model_trans[i](x, self.mask)  # self.mask
            logit += F.dropout(self.cls_mlp[i](rearrange(x, 'b n c -> b (n c)', b=batch_size, n=n)), self.dropout, training=self.training)
            attn_matrix_merge.append(attn_matrix)
            # x_attn, _ = self.cls_attn[i](x)
            # logit += F.dropout(self.cls_mlp[i](x_attn), self.dropout, training=self.training)
        merged_dict = {
            'attn_weights': torch.cat([d['attn_weights'].unsqueeze(1) for d in attn_matrix_merge], dim=1),
            'attn_weights_edge': torch.cat([d['attn_weights_edge'].unsqueeze(1) for d in attn_matrix_merge], dim=1),
            'attn_weights_node': torch.cat([d['attn_weights_node'].unsqueeze(1) for d in attn_matrix_merge], dim=1)
        }
        return logit, x, merged_dict


class Ourmutimodel(nn.Module):
    def __init__(self, argv, params, priori):
        super(Ourmutimodel, self).__init__()
        set_rng_seed(argv.seed)
        self.device = argv.device
        self.num_classes = argv.n_class  # 2
        self.fcn_node = argv.fcn_node  # 53
        self.msn_node = argv.msn_node  # 116
        self.dropout = params['dropout']

        self.fcn_layer = params['fcn_layer']
        self.fcn_head = params['fcn_head']
        self.msn_layer = params['msn_layer']
        self.msn_head = params['msn_head']

        # prior knowledge
        self.fp_node = priori['sim_fMRI_regions_norm'].to(self.device).detach()  # [116]
        self.fp_edge = priori['sim_fMRI_conn_norm'].to(self.device).detach()  # [116, 116]
        self.sp_node = priori['sim_sMRI_regions_norm'].to(self.device).detach()  # [116]
        self.sp_edge = priori['sim_sMRI_conn_norm'].to(self.device).detach()  # [116, 116]

        # FCN
        # self.linear_fcn = nn.Linear(53, 64, bias=False)
        self.model_fcn = Ourmodel(
            num_classes=self.num_classes,
            num_layers=self.fcn_layer,
            num_nodes=self.fcn_node,
            num_heads=self.fcn_head,
            hidden_dim=self.fcn_node,
            dropout=self.dropout,
            priori_node=self.fp_node,
            priori_edge=self.fp_edge,
        )

        # # MSN
        self.model_msn = Ourmodel(
            num_classes=self.num_classes,
            num_layers=self.msn_layer,
            num_nodes=self.msn_node,
            num_heads=self.msn_head,
            hidden_dim=self.msn_node,
            dropout=self.dropout,
            priori_node=self.sp_node,
            priori_edge=self.sp_edge,
        )

    def forward(self, multidata):
        # multidata
        fcn = multidata[:, :self.fcn_node*(self.fcn_node-1)//2].to(self.device)
        msn = multidata[:, self.fcn_node*(self.fcn_node-1)//2:].to(self.device)
        fcn = self.reshape_upper(fcn, self.fcn_node)
        msn = self.reshape_upper(msn, self.msn_node)
        # fcn = rearrange(fcn, 'b (n1 n2) -> b n1 n2', n1=self.fcn_node, n2=self.fcn_node)
        # msn = rearrange(msn, 'b (n1 n2) -> b n1 n2', n1=self.msn_node, n2=self.msn_node)

        batch_size, _, _ = fcn.shape

        # fcn = self.linear_fcn(fcn)
        logit_fcn, x_fcn, attn_matrix_fcn = self.model_fcn(fcn)
        logit_msn, x_msn, attn_matrix_msn = self.model_msn(msn)

        logit = logit_fcn + logit_msn

        output = {
            'Y_pred': logit,
            'attn_fcn': attn_matrix_fcn,
            'attn_msn': attn_matrix_msn,
        }
        return output

    def reshape_upper(self, mat_upper, n):
        """
        mat_upper: Tensor, shape = (batch_size, num_upper_elements)
        n: int, 

        return：
            mat_nn: Tensor, shape = (batch_size, n, n)
        """
        batch_size = mat_upper.shape[0]
        device = mat_upper.device

        row_idx, col_idx = torch.triu_indices(n, n, offset=1)
        mat_nn = torch.zeros((batch_size, n, n), device=device, dtype=mat_upper.dtype)

        # mat_upper is (batch_size, num_upper_elements)
        # row_idx, col_idx is (num_upper_elements)
        mat_nn[:, row_idx, col_idx] = mat_upper
        mat_nn[:, col_idx, row_idx] = mat_upper
        idx = torch.arange(n)
        mat_nn[:, idx, idx] = 1
        return mat_nn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('--datname', type=str, default='SZ', help='SZ, ASD')
    parser.add_argument('--cuda', type=int, default=3, help='')
    parser.add_argument('--seed', type=int, default=0, help='random seed setting')
    parser.add_argument('--n_class', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=32)  # 32
    parser.add_argument('--dropout', type=float, default=0.5)  # 0.5

    # FCN
    parser.add_argument('--fcn_layer', type=int, default=4)  # 3
    parser.add_argument('--fcn_head', type=int, default=2)
    # MSN
    parser.add_argument('--msn_layer', type=int, default=2)  # 1
    parser.add_argument('--msn_heads', type=int, default=4)

    args = parser.parse_args()

    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    args.data_path = f'../process/{args.datname}_Process/'
    if args.datname == 'SZ':
        from ..main_v14_SZ.dataset import Datasets
    elif args.datname == 'ASD':
        from ..main_v14_ASD.dataset import Datasets
    else:
        assert False
    dataset = Datasets(args)
    args.fcn_node = dataset.fcn[0].shape[0]
    args.msn_node = dataset.msn[0].shape[0]

    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)
    for i, multidata in enumerate(tqdm(dataloader, ncols=80)):
        msn = multidata['msn'].to(args.device)  # torch.Size([32, 116, 116])
        fcn = multidata['fcn'].to(args.device)  # torch.Size([32, 116, 116])

    priori = dataset.priori_dict
    param_grid = {
        'batch_size': [args.batch_size],  # 32
        'dropout': [args.dropout],  # 0.5
        # FCN
        'fcn_layer': [args.fcn_layer],  # 3
        'fcn_head': [args.fcn_head],
        # MSN
        'msn_layer': [args.msn_layer],  # 1
        'msn_head': [args.msn_heads],
    }
    params = ParameterGrid(param_grid)[0]

    model = Ourmutimodel(args, params, priori).to(args.device)
    output = model(multidata['fcn-msn'])

