import time
from pathlib import Path
import os

from tqdm import tqdm
import imageio
import cv2
import torch

from model import NeRF
from camera import get_rays, get_spiral_poses
from utils import rays2image


__author__ = "__Girish_Hegde__"


# =============================================================
# Parameters
# =============================================================
CKPT = './data/runs/best.pt'
OUTDIR = Path('./data/runs/ship_nerf')
VIZ_SCALE = 1  # scale output rendered image by this factor
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = torch.device('cpu')


# =============================================================
# Load Checkpoint
# =============================================================
ckpt = torch.load(CKPT, map_location='cpu')
ckpt = ckpt['nerf']

# =============================================================
# NeRF Init and Checkpoint Load
# =============================================================
nerf = NeRF(ckpt=ckpt, device=DEVICE, inference=True)
nerf.eval()
h = nerf.scene_params['h']
w = nerf.scene_params['w']
K = nerf.scene_params['K']
space_scale = nerf.scene_params['space_scale']
radius = nerf.scene_params['traj_radius']

# =============================================================
# Get Poses
# =============================================================
view_mats = get_spiral_poses(
    center = torch.tensor([0, 0, 0]),
    nviews = 100,
    radius = radius,
    vertical_range = (0, space_scale),
    rotations = 5,
)

# =============================================================
# Render Images
# =============================================================
print('Rendering ...')
with torch.no_grad():
    for i, c2w in tqdm(enumerate(view_mats), total=len(view_mats)):
        ray_o, ray_d = get_rays(h, w, K, c2w)
        ray_o, ray_d = ray_o.to(DEVICE), ray_d.to(DEVICE)
        rgb_c, rgb_f, vs = nerf.render_image(
            ray_o.reshape(-1, 3), ray_d.reshape(-1, 3), 1024
        )
        rays2image(
            rgb_f, vs, h, w, 
            stride=1, scale=VIZ_SCALE, bgr=False, 
            show=False, filename=OUTDIR/f'{i}.png'
        )

def create_gif_from_pngs(png_dir, gif_path, fps):
    png_files = [f for f in os.listdir(png_dir) if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('.')[0]))
    images = []
    for png_file in png_files:
        png_path = os.path.join(png_dir, png_file)
        images.append(cv2.imread(png_path)[..., ::-1])
    imageio.mimsave(gif_path, images, fps=fps)

create_gif_from_pngs(str(OUTDIR), OUTDIR/'nerf.gif', fps=10)

# =============================================================
# END
# =============================================================