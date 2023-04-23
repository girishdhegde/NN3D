from pathlib import Path

import numpy as np
import cv2
from einops import rearrange, repeat
import torch


__author__ = "__Girish_Hegde__"


def generate_coarse_samples(n_rays, samples_per_ray, near_planes, far_planes):
    """ Function to generate one sample uniformly from within 'n' evenly-spaced bins 
        b/n near_plane and far_plane.

    Args:
        n_rays (int): number of rays.
        samples_per_ray (int): The number of samples per ray.
        near_planes (torch.Tensor[float]/float): [n_rays, ] - The minimum depths per ray
        far_planes (torch.Tensor[float]/float): [n_rays, ] - The maximum depths per ray.

    Returns:
        tuple:
            torch.tensor[float] - samples: [b, n, ] - each representing the depth of a ray.
            torch.tensor[float] - distances: [b, n - 1, ] - each representing the distance between adjacent samples.
            torch.tensor[float] - starts [b, n, ]-  binwise starting points.
            torch.tensor[float] - binsizes [b, ]-  binsizes.
    """
    if isinstance(near_planes, (int, float)): near_planes = torch.FloatTensor([near_planes]).repeat(n_rays)
    if isinstance(far_planes, (int, float)): far_planes = torch.FloatTensor([far_planes]).repeat(n_rays)

    binsizes = (far_planes - near_planes) / samples_per_ray

    # draw samples from 'n' evenly-spaced bins. 
    t = torch.linspace(0, 1, samples_per_ray)
    starts = near_planes[:, None] + t[None, :]*(far_planes[:, None] -
                                                near_planes[:, None] - 
                                                binsizes[:, None])

    samples = starts + binsizes[:, None] * torch.rand((n_rays, samples_per_ray))

    # calculate the distances between adjacent samples.
    distances = samples[..., 1:] - samples[..., :-1]  # [batchsize, nsamples - 1]

    return samples, distances, starts, binsizes


def generate_fine_samples(n_rays, samples_per_ray, binsizes, starts, prob):
    """ Function to generate samples acc. to given probability.

    Args:
        n_rays (int): number of rays.
        samples_per_ray (int): The number of samples per ray.
        binsizes (float): [n_rays, ] - bin size.
        starts (torch.tensor[float]): [n_rays, samples_per_ray, ]-  binwise starting points.
        prob (torch.tensor[float]): [n_rays, samples_per_ray, ] - binwise probability

    Returns:
        torch.tensor[float] - samples: [n_rays, samples_per_ray, ] - each representing the depth of a ray.
    """
    indices = torch.multinomial(prob, num_samples=samples_per_ray, replacement=True)
    starts = starts[torch.arange(n_rays)[:, None], indices]
    samples = starts + binsizes[:, None]*torch.rand((n_rays, samples_per_ray)) 
    return samples


def volume_render(samples, distances, densities, colors, far_planes):
    """ Render a volume with the given densities, colors, and sample distances, using a ray casting algorithm.

    Args:
        samples (torch.tensor[float]): [B, N, ] - each representing the depth of a ray.
        distances (torch.tensor): [B, N - 1, ] -where each distance represents the distance between adjacent samples.
        densities (torch.tensor): [B, N, ] - where each density represents the likelihood of the corresponding ray intersecting an object.
        colors (torch.tensor): [B, N, 3] - where each color represents the color of the corresponding ray if it intersects an object.
        far_planes (torch.Tensor[float]/float): [B, ] - The maximum depths per ray.

    Returns:
        Tuple[torch.tensor, torch.tensor]: [B, 3, ] - color of ray, and [B, N, ] - probability density function (PDF) of the weights.
    """
    bs, n = samples.shape
    opacity = densities[..., :-1]*distances
    transmittances = torch.hstack([
        torch.ones(bs, 1, device=samples.device), 
        torch.exp(-torch.cumsum(opacity, dim=-1))
    ])

    if isinstance(far_planes, (int, float)): far_planes = torch.FloatTensor([far_planes]).repeat(bs)
    alphas = 1 - torch.hstack([
        torch.exp(-opacity), 
        torch.exp(-densities[..., -1]*(far_planes - samples[:, -1]))[:, None]
    ])

    weights = transmittances*alphas

    ray_color = torch.einsum('bn, bnc -> bc', weights, colors)
    pdf = weights/(torch.sum(weights, dim=-1, keepdim=True) + 1e-6)
    # cdf = np.cumsum(pdf)
    
    return ray_color, pdf


def hierarchical_volume_render(
        coarse_samples, coarse_densities, coarse_colors,
        fine_samples, fine_densities, fine_colors, far_planes
    ):
    """ Hierarchical Volume Render.
    """
    samples = torch.cat([coarse_samples, fine_samples], dim=1)
    sort_ids = samples.argsort(dim=-1)

    samples = torch.gather(samples, dim=1, index=sort_ids)
    distances = samples[..., 1:] - samples[..., :-1]
    densities = torch.cat([coarse_densities, fine_densities], dim=1)
    densities = torch.gather(densities, dim=1, index=sort_ids)
    colors = torch.cat([coarse_colors, fine_colors], dim=1)
    colors = rearrange(
        [
            torch.gather(colors[:, :, 0], dim=1, index=sort_ids),
            torch.gather(colors[:, :, 1], dim=1, index=sort_ids),
            torch.gather(colors[:, :, 2], dim=1, index=sort_ids),
        ], 'c b n -> b n c'
    )

    ray_color, pdf = volume_render(samples, distances, densities, colors, far_planes)
    return ray_color, pdf, (samples, distances, densities, colors)


def rays2image(ray_colors, height, width, stride=1, scale=1, bgr=True, show=False, filename=None):
    if isinstance(ray_colors, torch.Tensor): ray_colors = ray_colors.numpy()
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
        cv2.imwrite(filename, img)

    return img


# https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/utils/math.py
def intersect_aabb(
    origins: torch.Tensor,
    directions: torch.Tensor,
    aabb: torch.Tensor,
    max_bound: float = 1e10,
    invalid_value: float = 1e10,
):
    """
    Implementation of ray intersection with AABB box

    Args:
        origins: [N,3] tensor of 3d positions
        directions: [N,3] tensor of normalized directions
        aabb: [6] array of aabb box in the form of [x_min, y_min, z_min, x_max, y_max, z_max]
        max_bound: Maximum value of t_max
        invalid_value: Value to return in case of no intersection

    Returns:
        t_min, t_max - two tensors of shapes N representing distance of intersection from the origin.
    """

    tx_min = (aabb[:3] - origins) / directions
    tx_max = (aabb[3:] - origins) / directions

    t_min = torch.min(tx_min, tx_max)
    t_max = torch.max(tx_min, tx_max)

    t_min = torch.max(t_min, dim=-1).values
    t_max = torch.min(t_max, dim=-1).values

    t_min = torch.clamp(t_min, min=0, max=max_bound)
    t_max = torch.clamp(t_max, min=0, max=max_bound)

    cond = t_max <= t_min
    t_min = torch.where(cond, invalid_value, t_min)
    t_max = torch.where(cond, invalid_value, t_max)

    return t_min, t_max


if __name__ == '__main__':
    Nc = 64  # No. of coarse samples
    Nf = 128  # No. of fine samples
    near_plane, far_plane = 0, 4
    bs = 2

    def get_random_data(b, n):
        densities = torch.rand((b, n))
        colors = torch.rand((b, n, 3))
        return densities, colors

    coarse_samples, coarse_distances, bin_starts, bin_size = generate_coarse_samples(bs, Nc, near_plane, far_plane)
    densities_c, colors_c = get_random_data(bs, Nc)
    coarse_color, pdf = volume_render(coarse_samples, coarse_distances, densities_c, colors_c, far_plane)
    print(f'{coarse_color=}')

    fine_samples = generate_fine_samples(bs, Nf, bin_size, bin_starts, pdf)
    densities_f, colors_f = get_random_data(bs, Nf)
    ray_color, ray_pdf, (samples, distances, densities, colors) = hierarchical_volume_render(
        coarse_samples, densities_c, colors_c,
        fine_samples, densities_f, colors_f,
        far_plane,
    )
    print(f'{ray_color=}')
    print(f'{ray_pdf.shape=}')

# TODO: verify updated functions