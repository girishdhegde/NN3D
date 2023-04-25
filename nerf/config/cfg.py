from pathlib import Path

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
# system
# dtype = 'bfloat16' # 'float32' or 'bfloat16'
# compile = True # use PyTorch 2.0 to compile the model to be faster