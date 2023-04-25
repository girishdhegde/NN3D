from pathlib import Path
import random

import numpy as np
import cv2
from einops import rearrange, repeat
import torch
import torch.nn.functional as F


__author__ = "__Girish_Hegde__"


def set_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
        nerf, itr, val_loss, train_loss, best, filename, **kwargs,
    ):
    ckpt = {
        'nerf': nerf,
        'training':{
            'iteration':itr, 'val_loss':val_loss, 'train_loss':train_loss, 'best':best,
        },
        'kwargs':kwargs,
    }
    torch.save(ckpt, filename)
    return ckpt


def load_checkpoint(filename):
    itr, best = 1, float('inf')
    nerf_ckpt, kwargs = None, None
    if filename is not None:
        if Path(filename).is_file():
            ckpt = torch.load(filename, map_location='cpu')
            nerf_ckpt = ckpt['nerf']
            if 'training' in ckpt:
                itr, val_loss, train_loss, best = ckpt['training'].values()
                print('Training parameters loaded successfully ...')
            if 'kwargs' in ckpt:
                kwargs = ckpt['kwargs']
                print('Additional kwargs loaded successfully ...')
    return nerf_ckpt, itr, best, kwargs


@torch.no_grad()
def rays2image(ray_colors, valids, height, width, stride=1, scale=1, bgr=True, show=False, filename=None):
    if isinstance(ray_colors, torch.Tensor): 
        ray_colors = ray_colors.cpu().numpy()
        valids = valids.cpu().numpy()
    ray_colors[np.logical_not(valids)] = 0.

    img = np.zeros((height, width, 3))
    rendering = rearrange(ray_colors, '(h w) c -> h w c', w=width//stride)[:, :, ::-1 if bgr else 1]
    img[::stride, ::stride] = rendering
    img = (np.clip(img, 0, 1)*255).astype(np.uint8)
    if scale > 1: img = cv2.resize(img, (width*scale, height*scale), interpolation=cv2.INTER_NEAREST)

    if show:
        cv2.imshow('rendering', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if filename is not None:
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(filename), img)

    return img