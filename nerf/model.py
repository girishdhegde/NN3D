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
        thetas = self.omegas.to(x.device)*x  # [b, c, L]
        x = rearrange(
            [torch.sin(thetas), torch.cos(thetas)], 't b c l -> b (c t l)'
        )  # [b, c*t*L]
        return x


class MLP(nn.Module):
    def __init__(
            self, pos_emb_dim=10, dir_emb_dim=4, 
            n_layers=8, feat_dim=256, skips=5, 
            rgb_layers=1,
        ):
        super().__init__()
        self.pos_emb_dim = pos_emb_dim
        self.dir_emb_dim = dir_emb_dim
        self.n_layers = n_layers
        self.feat_dim = feat_dim
        self.skip_conn_layer = skips
        self.rgb_layers = rgb_layers

        self.pos_emb = Embedding(pos_emb_dim)
        self.dir_emb = Embedding(dir_emb_dim)

        pos_dim, dir_dim = pos_emb_dim*3*2, dir_emb_dim*3*2

        self.layers = nn.ModuleList(
            [nn.Linear(pos_dim, feat_dim), nn.ReLU()] + 
            [layer
                for i in range(n_layers - 1)
                    for layer in (
                                    nn.Linear(feat_dim + (pos_dim if (i + 1) in skips else 0), feat_dim), 
                                    nn.ReLU()
                                )
            ] + 
            [nn.Linear(feat_dim, feat_dim)]
        )

        self.to_density = nn.Sequential(nn.Linear(feat_dim, 1), nn.ReLU())

        self.to_rgb = nn.Sequential(
            nn.Linear(feat_dim + dir_dim, feat_dim//2), nn.ReLU(), 
            *(layer for i in range(rgb_layers - 1) 
                        for layer in (nn.Linear(feat_dim//2, feat_dim//2), nn.ReLU())
            ), nn.Linear(feat_dim//2, 3), nn.Sigmoid(),
        )


class NeRF: pass