import numpy as np
from einops import repeat, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from graphics import (
    generate_coarse_samples, generate_fine_samples, 
    volume_render, hierarchical_volume_render,
    rays2image, intersect_aabb,
)


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
    def create_from_ckpt(cls, ckpt):
        if not isinstance(ckpt, dict): ckpt = torch.load(ckpt)
        net = cls(**ckpt['config'])
        net.load_state_dict(ckpt['state_dict'])
        return net


class NeRF:
    def __init__(
        self,
        device = 'cuda',
        # model params
        pos_emb_dim = 10, 
        dir_emb_dim = 4, 
        n_layers = 8, 
        feat_dim = 256, 
        skips = [5, ],  
        rgb_layers = 1,
        # optim params
        lr = 5e-4,
        # volume rendering
        coarse_samples = 64,
        fine_samples = 128,
        # scene
        scene_params = None,
        # checkpoint
        ckpt = None,
        inference = False,
    ):  
        self.device = device
        self.coarse_samples = coarse_samples
        self.fine_samples = fine_samples
        self.aabb = torch.tensor([-1, -1, -1, 1, 1, 1.]).to(self.device)
        self.scene_params = scene_params

        if ckpt is None:
            self.coarse_net = Field(
                pos_emb_dim, dir_emb_dim, 
                n_layers, feat_dim, 
                skips, rgb_layers,
            ).to(device)

            self.fine_net = Field(
                pos_emb_dim, dir_emb_dim, 
                n_layers, feat_dim, 
                skips, rgb_layers,
            ).to(device)

            self.coarse_opt = torch.optim.Adam(self.coarse_net.parameters(), lr=lr)
            self.fine_opt = torch.optim.Adam(self.fine_net.parameters(), lr=lr)

        else:
            self.load_ckpt(ckpt, inference)

        self.criterion = nn.MSELoss()

    def get_ckpt(self):
        ckpt = {
            'coarse_net': {
                'config': self.coarse_net.get_config(),
                'state_dict': self.coarse_net.state_dict(),
            },
            'fine_net': {
                'config': self.fine_net.get_config(),
                'state_dict': self.fine_net.state_dict(),
            },
            'coarse_opt': {
                'state_dict': self.coarse_opt.state_dict(),
            },
            'fine_opt': {
                'state_dict': self.coarse_opt.state_dict(),
            },
            'volume_rendering': {
                'coarse_samples': self.coarse_samples,
                'fine_samples': self.fine_samples,
            },
            'scene': self.scene_params,
        }
        return ckpt

    def load_ckpt(self, ckpt, inference=False):
        if not isinstance(ckpt, dict): ckpt = torch.load(ckpt)
        self.coarse_net = Field.create_from_ckpt(ckpt['coarse_net'])
        self.fine_net = Field.create_from_ckpt(ckpt['fine_net'])
        self.coarse_net.to(self.device)
        self.fine_net.to(self.device)
        print(f'Models loaded successfully ...')

        if (not inference) and ('coarse_opt' in ckpt):
            self.coarse_opt = torch.optim.Adam(self.coarse_net.parameters(), lr=5e-4)
            self.fine_opt = torch.optim.Adam(self.fine_net.parameters(), lr=5e-4)
            self.coarse_opt.load_state_dict(ckpt['coarse_opt']['state_dict'])
            self.fine_opt.load_state_dict(ckpt['fine_opt']['state_dict'])
            print(f'Optimizers loaded successfully ...')
        
        if 'volume_rendering' in ckpt:
            self.coarse_samples = ckpt['volume_rendering']['coarse_samples']
            self.fine_samples = ckpt['volume_rendering']['fine_samples']

        if 'scene' in ckpt:
            self.scene_params = ckpt['scene']
        
    def save_ckpt(self, filename):
        ckpt = self.get_ckpt()
        torch.save(ckpt, filename)

    def train(self):
        self.coarse_net.train()
        self.fine_net.train()

    def eval(self):
        self.coarse_net.eval()
        self.fine_net.eval()
    
    def zero_grad(self, *args, **kwargs):
        self.coarse_opt.zero_grad(*args, **kwargs)
        self.fine_opt.zero_grad(*args, **kwargs)

    def render(self, origins, directions, tmins, tmaxs):
        n_rays = origins.shape[0]
        samples_c, distances_c, starts, binsizes = generate_coarse_samples(
            n_rays, self.coarse_samples, tmins, tmaxs,
        )
        directions_c = repeat(directions, 'n c -> n r c', r=self.coarse_samples)
        positions_c = origins[:, None, :] + directions_c*samples_c[:, :, None]

        densities_c, colors_c = self.coarse_net(
            positions_c.reshape(-1, 3), directions_c.reshape(-1, 3)
        )

        ray_color_c, pdf = volume_render(
            samples_c, distances_c, 
            densities_c.reshape(n_rays, -1), colors_c.reshape(n_rays, -1, 3), 
            tmaxs,
        )

        samples_f = generate_fine_samples(
            n_rays, self.fine_samples, binsizes, starts, pdf.detach(),
        )
        directions_f = repeat(directions, 'n c -> n r c', r=self.fine_samples)
        positions_f = origins[:, None, :] + directions_f*samples_f[:, :, None]

        densities_f, colors_f = self.fine_net(
            positions_f.reshape(-1, 3), directions_f.reshape(-1, 3)
        )

        ray_color_f, pdf, (samples, distances, densities, colors) = hierarchical_volume_render(
            samples_c, 
            densities_c.detach().reshape(n_rays, -1), colors_c.detach().reshape(n_rays, -1, 3),
            samples_f, densities_f.reshape(n_rays, -1), colors_f.reshape(n_rays, -1, 3),
            tmaxs,  
        )

        return ray_color_c, ray_color_f, (pdf, samples, distances, densities, colors)
     
    def forward(self, data):
        origins, directions, density, rgb = (d.to(self.device) for d in data)
        tmins, tmaxs, valids = intersect_aabb(origins, directions, self.aabb)
        ray_color_c, ray_color_f, volume_data = self.render(origins, directions, tmins, tmaxs)
        loss = self.criterion(ray_color_c[valids], rgb[valids]) \
               + self.criterion(ray_color_f[valids], rgb[valids])
        return (ray_color_c, ray_color_f, valids), loss
       
    def optimize(self, gradient_clip=None, new_lr=None, *args, **kwargs):
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.coarse_net.parameters(), gradient_clip)
            nn.utils.clip_grad_norm_(self.fine_net.parameters(), gradient_clip)

        if new_lr is not None:
            for param_group in self.coarse_opt.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.fine_opt.param_groups:
                param_group['lr'] = new_lr

        self.coarse_opt.step()
        self.fine_opt.step()
        self.zero_grad(*args, **kwargs)

    @torch.no_grad()
    def render_image(self, origins, directions, n_rays=1024):
        origins, directions = origins.to(self.device), directions.to(self.device)
        remainder =  origins.shape[0]%n_rays
        colors_c, colors_f, valids = [], [], []

        for o, d in zip(
            rearrange(origins[:-remainder], '(b n) c -> b n c', n=n_rays),
            rearrange(directions[:-remainder], '(b n) c -> b n c', n=n_rays)
        ):
            torch.save([o, d], './data/bug/od.pt')
            tmins, tmaxs, vs = intersect_aabb(o, d, self.aabb)
            ray_color_c, ray_color_f, volume_data = self.render(o, d, tmins, tmaxs)
            colors_c.append(ray_color_c)
            colors_f.append(ray_color_f)
            valids.append(vs)
        colors_c = rearrange(colors_c, 'b n c -> (b n) c')
        colors_f = rearrange(colors_f, 'b n c -> (b n) c')
        valids = rearrange(valids, 'b n -> (b n)')

        tmins, tmaxs, vs = intersect_aabb(origins[-remainder:], directions[-remainder:], self.aabb)
        ray_color_c, ray_color_f, volume_data = self.render(
            origins[-remainder:], directions[-remainder:], tmins, tmaxs
        )
        colors_c = torch.vstack((colors_c, ray_color_c))
        colors_f = torch.vstack((colors_f, ray_color_f))
        valids = torch.hstack((valids, vs))

        return colors_c, colors_f, valids


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