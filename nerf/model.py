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
        thetas = self.omegas.to(x.device)*x[..., None]  # [b, c, L]
        x = rearrange(
            [torch.sin(thetas), torch.cos(thetas)], 't b c l -> b (c t l)'
        )  # [b, c*t*L]
        return x


class Field(nn.Module):
    def __init__(
            self, pos_emb_dim=10, dir_emb_dim=4, 
            n_layers=8, feat_dim=256, skips=[5, ],  
            rgb_layers=1,
        ):
        super().__init__()
        self.pos_emb_dim = pos_emb_dim
        self.dir_emb_dim = dir_emb_dim
        self.n_layers = n_layers
        self.feat_dim = feat_dim
        self.skips = skips
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

    def forward(self, x, d):
        x = self.pos_emb(x)
        d = self.dir_emb(d)
        h = x

        for i, layer in enumerate(self.layers):
            if (not i%2) and (i//2 in self.skips):
                x = torch.cat([x, h], dim=-1)
            x = layer(x)

        density = self.to_density(x)
        rgb = self.to_rgb(torch.cat([x, d], dim=-1))

        return density, rgb

    def get_config(self):
        config = {
            'pos_emb_dim': self.pos_emb_dim, 'dir_emb_dim': self.dir_emb_dim, 
            'n_layers': self.n_layers, 'feat_dim': self.feat_dim, 
            'skips': self.skips, 'rgb_layers': self.rgb_layers,
        }
        return config

    def save_ckpt(self, filename):
        ckpt = {
            'config':self.get_config(),
            'state_dict':self.state_dict(),
        }
        torch.save(ckpt, filename)
        return ckpt

    @property
    def n_params(self):
        total = sum(p.numel() for p in self.parameters())
        return total

    @classmethod
    def create_from_ckpt(cls, filename):
        ckpt = torch.load(filename)
        net = cls(**ckpt['config'])
        net.load_state_dict(ckpt['state_dict'])
        return net


if __name__ == '__main__':
    net = Field(
        pos_emb_dim=10, dir_emb_dim=4, 
        n_layers=8, feat_dim=256, skips=[5, ],
        rgb_layers=2,
    )
    print(net)

    net.zero_grad()
    density, rgb = net(torch.randn(100, 3), torch.randn(100, 3))
    print(f'{density.shape, rgb.shape = }')
    rgb.sum().backward()
    print(f'{torch.abs(net.layers[0].weight.grad).sum() = }')

    net.zero_grad()
    density, rgb = net(torch.randn(100, 3), torch.randn(100, 3))
    print(f'{density.shape, rgb.shape = }')
    density.sum().backward()
    print(f'{torch.abs(net.layers[0].weight.grad).sum() = }')