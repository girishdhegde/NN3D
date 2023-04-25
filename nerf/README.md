# NeRF - Neural Radiance Fields
This repository contains a Pytorch implementation of the Neural Radiance Fields (NeRF) algorithm from scratch as described in the paper ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"](https://arxiv.org/pdf/2003.08934.pdf).

# Dependencies
* Python 3
* Pytorch 1.9 or higher
* Open3d (Optional)

# Requirements
```pip install -r requirements.txt```

# Dataset
To train and test the NeRF model, you will need a dataset of **images** and corresponding **camera poses**. So download the **nerf-sythetic** dataset from the links provided here https://www.matthewtancik.com/nerf.

# Usage

* Create config.py file with proper parameters. Refer example config - [config/ship.py](./config/ship.py)
* To train the NeRF model, run the following command:
    ```python train.py path/to/config.py```
* To sample/visualize results, run the following command:
    ```python demo.py path/to/config.py```
* To visualize/understand volume rendering, run the notebook [voxel_volume_rendeing.ipynb](./voxel_volume_rendering.ipynb)

# Results

# License: MIT

# References
* https://arxiv.org/pdf/2003.08934.pdf
* https://www.matthewtancik.com/nerf
* https://github.com/nerfstudio-project/nerfstudio
* https://github.com/yenchenlin/nerf-pytorch
* https://github.com/bmild/nerf