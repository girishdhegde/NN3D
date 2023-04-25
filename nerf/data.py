from pathlib import Path
import random
import json

import numpy as np
import cv2
import torch

from camera import fovx2intrinsic, get_rays


__author__ = "__Girish_Hegde__"


def load_blender_data(basedir, split='train', res_scale=0.5, skip=1, return_torch=True):
    basedir = Path(basedir)
    with open(basedir/f'transforms_{split}.json', 'r') as fp:
        fovx, frames = json.load(fp).values()

    poses, images = [], []
    for frame in frames[::skip]:
        c2w = frame['transform_matrix']
        poses.append(c2w)
        filepath = frame['file_path']
        imgpath = basedir/f'{filepath[2:]}.png'
        img = cv2.imread(str(imgpath), cv2.IMREAD_UNCHANGED)
        if res_scale != 1:
            h, w = img.shape[:2]
            h, w = int(h*res_scale), int(w*res_scale)
            img = cv2.resize(img, (h, w), interpolation=cv2.INTER_CUBIC)
        images.append(img)

    poses, images = np.array(poses), np.array(images)/255
    if return_torch:
        poses, images = torch.FloatTensor(poses), torch.FloatTensor(images)
    return [h, w, fovx], poses, images


class BlenderSet:
    def __init__(self, basedir, split='train', res_scale=0.5, skip=1, n_rays=1024, max_iters=None):
        (self.h, self.w, self.fovx), poses, images = load_blender_data(
            basedir, split, res_scale, skip, return_torch=True,
        )
        self._get_bounds(poses)
        self.nframes = len(images)
        self.rgbs = images[..., :3].reshape(self.nframes, -1, 3)
        self.densities = images[..., 3].reshape(self.nframes, -1)
        self.n_rays = n_rays
        self.npts = self.h*self.w

        self.K = fovx2intrinsic(self.fovx, self.w, self.h, res_scale)

        self.origins, self.directions = [], []
        for c2w in poses:
            o, d = get_rays(self.h, self.w, self.K, c2w)
            self.origins.append(o)
            self.directions.append(d)
        self.origins = torch.stack(self.origins).reshape(self.nframes, -1, 3)
        self.directions = torch.stack(self.directions).reshape(self.nframes, -1, 3)

        self.max_iters = max_iters or 1e9

    def __len__(self):
        return self.max_iters
    
    def __getitem__(self, idx):
        if idx >= self.max_iters: raise StopIteration
        
        img_idx = torch.randint(
            low=0, high=self.nframes, size=(self.n_rays,), dtype=torch.int64
        )
        pixel_idx = torch.randperm(self.npts)[:self.n_rays]

        rgb = self.rgbs[img_idx, pixel_idx]
        density = self.densities[img_idx, pixel_idx]

        origins = self.origins[img_idx, pixel_idx]
        directions = self.directions[img_idx, pixel_idx]

        return origins, directions, density, rgb
    
    def _get_bounds(self, poses):
        self.maxes, self.mins = poses[:, :3, 3].max(0).values, poses[:, :3, 3].min(0).values
        self.traj_height = self.maxes[2] - self.mins[2]
        self.traj_center = torch.tensor([0., 0, 0])
        self.obj_center = torch.tensor([
            (self.maxes[0] - self.mins[0])/2, 
            (self.maxes[1] - self.mins[1])/2,
            0.5,
        ])
        self.traj_radius = max(self.obj_center[:2].numpy().tolist())
        self.tmin, self.tmax = 0, self.traj_radius*2
        # https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/blender_dataparser.py
        self.aabb_box = torch.FloatTensor(
            [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]  # [x_min, y_min, z_min, x_max, y_max, z_max]
        )

    def get_params(self):
        params = {
            'h': self.h, 'w': self.w, 'fovx': self.fovx, 'K': self.K,
            'nframes': self.nframes, 'maxes': self.maxes, 'mins': self.mins,
            'traj_height': self.traj_height, 'traj_center': self.traj_center,
            'obj_center': self.obj_center, 'traj_radius': self.traj_radius,
        }
        return params

    def get_image(self, idx=None):
        idx = idx or random.randint(0, self.nframes - 1)
        return idx, self.origins[idx], self.directions[idx], self.densities[idx], self.rgbs[idx]