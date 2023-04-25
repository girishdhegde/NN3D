import math
import time
from pathlib import Path

from tqdm import tqdm
import torch

from model import NeRF
from data import BlenderSet
from utils import set_seed, save_checkpoint, load_checkpoint, rays2image


__author__ = "__Girish_Hegde__"


# config file - (overrides the parameters given here)
CFG = './config/ship.py'  # 'path/to/config/file.py'
# =============================================================
# Parameters
# =============================================================
# model
POS_EMB_DIM = 10 
DIR_EMB_DIM = 4
N_LAYERS = 8
FEAT_DIM = 256 
SKIPS = [5, ]  
RGB_LAYERS = 1
# logging
LOGDIR = Path('./data/runs')
LOAD = LOGDIR/'ckpt.pt'  # or None
PRINT_INTERVAL = 10
# dataset
BASEDIR = './data/nerf_synthetic/ship'
RES_SCALE = 0.125
# training
N_RAYS = 1024  # num. of rays per batch
NC = 64  # num. of coarse samples per ray
NF = 128  # num. of fine/importance samples per ray
GRAD_ACC_STEPS = 1  # used to simulate larger batch sizes
MAX_ITERS = 100_000

EVAL_INTERVAL = 500
EVAL_ITERS = 100
EVAL_ONLY = False  # if True, script exits right after the first eval
SAVE_EVERY = False  # save unique checkpoint at every eval interval.
GRADIENT_CLIP = None  # 5
# adam optimizer
LR = 5e-4  # max learning rate
# learning rate decay settings
DECAY_LR = True  # whether to decay the learning rate
MIN_LR = LR/10
# object bbox
BOX = [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]  # [x_min, y_min, z_min, x_max, y_max, z_max]
# system
# dtype = 'bfloat16' # 'float32' or 'bfloat16'
# compile = True # use PyTorch 2.0 to compile the model to be faster
# =============================================================


# warning!!! executes codes in config file directly with no safety!
with open(CFG, 'r') as fp: exec(fp.read())  # import cfg settings
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOGDIR.mkdir(parents=True, exist_ok=True)
set_seed(108)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cudnn.benchmark = True  # optimize backend algorithms

# =============================================================
# Tokenizer, Dataset, Dataloader init
# =============================================================
trainset = BlenderSet(BASEDIR, 'train', res_scale=RES_SCALE, 
                      n_rays=N_RAYS, max_iters=MAX_ITERS*GRAD_ACC_STEPS, 
                      aabb_bbox=BOX, )
evalset = BlenderSet(BASEDIR, 'val', res_scale=RES_SCALE, 
                      n_rays=N_RAYS, max_iters=MAX_ITERS*GRAD_ACC_STEPS,
                      aabb_bbox=BOX, )

# =============================================================
# Load Checkpoint
# =============================================================
nerf_ckpt, itr, best, kwargs = load_checkpoint(LOAD)

# =============================================================
# NeRF(Model, Optimizer, Criterion) init and checkpoint load
# =============================================================
nerf = NeRF(
    DEVICE,
    POS_EMB_DIM, DIR_EMB_DIM, 
    N_LAYERS, FEAT_DIM, SKIPS,  
    RGB_LAYERS,
    LR,
    NC, NF,
    trainset.tmin, trainset.tmax,
    trainset.get_params(),
    nerf_ckpt,
)

# =============================================================
# Learning Rate Decay Scheduler (exponential decay)
# =============================================================
decay_rate = math.log(LR/MIN_LR)/MAX_ITERS
def get_lr(itr):
    lr = LR*math.exp(-decay_rate*itr)
    return lr

# =============================================================
# Training loop - forward, backward, loss, optimize
# =============================================================
trainloss, valloss, log_trainloss = 0, 0, 0
nerf.train()
nerf.zero_grad(set_to_none=True)
trainloader, evalloader = iter(trainset), iter(evalset)
print('Training ...')
start_time = time.perf_counter()
for itr in range(itr, MAX_ITERS + 1):
    # =============================================================
    # Validation
    # =============================================================
    if (itr%EVAL_INTERVAL == 0) or EVAL_ONLY:
        print('Evaluating ...')
        nerf.eval()
        valloss = 0
        with torch.no_grad():
            for data in tqdm(evalloader, total=len(evalloader)):
                data = next(trainloader)
                (ray_colors_c, ray_colors_f), loss = nerf.forward(data)
                valloss += loss.item()
        nerf.train()

        valloss = valloss/EVAL_ITERS
        trainloss = trainloss/EVAL_INTERVAL

        # =============================================================
        # Saving and Logging
        # =============================================================
        if EVAL_ONLY:
            log_data = f'val loss: {valloss}, \t time: {(end_time - start_time)/60}M'
            print(f'{"-"*150}\n{log_data}\n{"-"*150}')
            break

        print('Saving checkpoint ...')
        ckpt_name = LOGDIR/'ckpt.pt' if not SAVE_EVERY else LOGDIR/f'ckpt_{itr}.pt'
        save_checkpoint(
            nerf.get_ckpt(), itr, valloss, trainloss, best, ckpt_name,
        )

        if valloss < best:
            best = valloss
            save_checkpoint(
                nerf.get_ckpt(), itr, valloss, trainloss, best, LOGDIR/'best.pt',
            )
        
        idx, ray_o, ray_d, d, rgb = evalset.get_image()
        rgb_c, rgb_f = nerf.render_image(ray_o, ray_d, N_RAYS)
        rays2image(
            rgb_f, evalset.h, evalset.w, 
            stride=1, scale=1, bgr=True, 
            show=False, filename=LOGDIR/'renders'/f'{itr}_{idx}.png'
        )

        logfile = LOGDIR/'log.txt'
        log_data = f"iteration: {itr}/{MAX_ITERS}, \tval loss: {valloss}, \ttrain loss: {trainloss}, \tbest loss: {best}"
        with open(logfile, 'a' if logfile.is_file() else 'w') as fp:
            fp.write(log_data + '\n')
        end_time = time.perf_counter()
        log_data = f'{log_data}, \t time: {(end_time - start_time)/60}M'
        print(f'{"-"*150}\n{log_data}\n{"-"*150}')

        trainloss = 0
        start_time = time.perf_counter()
        print('Training ...')

    # =============================================================
    # Training
    # =============================================================
    # forward, loss, backward with grad. accumulation
    loss_ = 0
    for step in range(GRAD_ACC_STEPS):
        data = next(trainloader)
        (ray_colors_c, ray_colors_f), loss = nerf.forward(data)
        loss.backward()
        loss_ += loss.item()

    # optimize params
    loss_ = loss_/GRAD_ACC_STEPS
    trainloss += loss_
    log_trainloss += loss_

    if DECAY_LR: lr = get_lr(itr)
    nerf.optimize(GRADIENT_CLIP, new_lr=None if not DECAY_LR else lr, set_to_none=True)

    # print info.
    if itr%PRINT_INTERVAL == 0:
        log_data = f"iteration: {itr}/{MAX_ITERS}, \ttrain loss: {log_trainloss/PRINT_INTERVAL}"
        print(log_data)
        log_trainloss = 0

# =============================================================
# END
# =============================================================