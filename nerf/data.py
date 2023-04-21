from pathlib import Path
import random
import json

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from camera import fovx2intrinsic, get_rays_from_extrinsics


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


class SyntheticSet(Dataset):
    def __init__(self, basedir, split='train', res_scale=0.5, skip=1, n_rays=1024):
        super().__init__()
        (self.h, self.w, self.fovx), poses, images = load_blender_data(
            basedir, split, res_scale, skip, return_torch=True
        )
        self.nframes = len(images)
        self.rgbs = images[..., :3].reshape(self.nframes, -1, 3)
        self.densities = images[..., 3].reshape(self.nframes, -1)
        self.n_rays = n_rays
        self.npts = self.h*self.w

        self.K = fovx2intrinsic(self.fovx, self.w, self.h)
        
        self.origins, self.directions = [], []
        for c2w in poses:
            o, d = get_rays_from_extrinsics(self.h, self.w, self.K, c2w)
            self.origins.append(o)
            self.directions.append(d)
        self.origins = torch.stack(self.origins).reshape(self.nframes, -1, 3)
        self.directions = torch.stack(self.directions).reshape(self.nframes, -1, 3)

    def __len__(self):
        return 1e9
    
    def __getitem__(self, idx):
        idx = random.randint(0, self.nframes - 1)
        mask = torch.randperm(self.npts)[:self.n_rays]

        rgb = self.rgbs[idx, mask]
        density = self.densities[idx, mask]

        origins = self.origins[idx, mask]
        directions = self.directions[idx, mask]

        return origins, directions, rgb, density
    
    def get_image(self, idx=None):
        idx = idx or random.randint(0, self.nframes - 1)
        return self.origins[idx], self.directions[idx], self.rgbs[idx], self.densities[idx]


