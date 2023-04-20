import numpy as np
from einops import repeat, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


__author__ = "__Girish_Hegde__"


class Embedding(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.omegas = torch.FloatTensor(
            [(2**l)*np.pi for l in range(L)]
        )[None, None, :]  # [1, 1, L - 1]
    
    def forward(self, x):
        thetas = self.omegas.to(x.device)*x  # [b, c, L - 1]
        x = rearrange(
            [torch.sin(thetas), torch.cos(thetas)], 't b c l -> b (c t l)'
        )
        return x


class MLP(nn.Module):
    def __init__(self, pos_emb_dim=10, dir_emb_dim=4, n_layers=8, feat_dim=256):
        super().__init__()
        self.pos_emb_dim = pos_emb_dim
        self.dir_emb_dim = dir_emb_dim
        self.n_layers = n_layers
        self.feat_dim = feat_dim


class NeRF: pass