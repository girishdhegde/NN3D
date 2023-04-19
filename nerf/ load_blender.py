from pathlib import Path
import json

import numpy as np
import cv2
import einops
import torch


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