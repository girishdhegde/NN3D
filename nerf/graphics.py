from pathlib import Path

import numpy as np
import cv2
from einops import rearrange, repeat
import torch


__author__ = "__Girish_Hegde__"


def generate_coarse_samples(batch_size, nsamples, min_depth, max_depth):
    """ Function to generate one sample uniformly from within 'n' evenly-spaced bins 
        b/n min_depth and max_depth.

    Args:
        batch_size (int): batch size.
        nsamples (int): The number of samples to generate.
        min_depth (float): The minimum depth of the scene.
        max_depth (float): The maximum depth of the scene.

    Returns:
        tuple:
            torch.tensor[float] - samples: [b, n, ] - each representing the depth of a ray.
            torch.tensor[float] - distances: [b, n - 1, ] - each representing the distance between adjacent samples.
            torch.tensor[float] - starts [n, ]-  binwise starting points.
            float - binsize
    """
    binsize = (max_depth - min_depth) / nsamples

    #  draw samples from 'n' evenly-spaced bins. 
    starts = torch.linspace(min_depth, max_depth - binsize, nsamples)
    samples = starts[None, :] + binsize * torch.rand((batch_size, nsamples))

    # calculate the distances between adjacent samples.
    distances = samples[..., 1:] - samples[..., :-1]  # [batchsize, nsamples - 1]

    return samples, distances, starts, binsize


def generate_fine_samples(batch_size, nsamples, binsize, starts, prob):
    """ Function to generate samples acc. to given probability.

    Args:
        batch_size (int): batch size.
        nsamples (int): The number of samples to generate.
        binsize (float): bin size.
        starts (torch.tensor[float]): [nsamples, ]-  binwise starting points.
        prob (torch.tensor[float]): [batch_size, nsamples,] - binwise probability


    Returns:
        torch.tensor[float] - samples: [b, n, ] - each representing the depth of a ray.
    """
    indices = torch.multinomial(prob, num_samples=nsamples, replacement=True)
    samples = starts[indices] + binsize*torch.rand((batch_size, nsamples)) 
    return samples


def volume_render(samples, distances, densities, colors, max_depth):
    """ Render a volume with the given densities, colors, and sample distances, using a ray casting algorithm.

    Args:
        samples (torch.tensor[float]): [B, N, ] - each representing the depth of a ray.
        distances (torch.tensor): [B, N - 1, ] -where each distance represents the distance between adjacent samples.
        densities (torch.tensor): [B, N, ] - where each density represents the likelihood of the corresponding ray intersecting an object.
        colors (torch.tensor): [B, N, 3] - where each color represents the color of the corresponding ray if it intersects an object.
        max_depth (float): The maximum depth of the scene.

    Returns:
        Tuple[torch.tensor, torch.tensor]: [B, 3, ] - color of ray, and [B, N, ] - probability density function (PDF) of the weights.
    """
    bs, n = samples.shape
    opacity = densities[..., :-1]*distances
    transmittances = torch.hstack([
        torch.ones(bs, 1), 
        torch.exp(-torch.cumsum(opacity, dim=-1))
    ])

    alphas = 1 - torch.hstack([
        torch.exp(-opacity), 
        torch.exp(-densities[..., -1]*(max_depth - samples[:, -1]))[:, None]
    ])

    weights = transmittances*alphas

    ray_color = torch.einsum('bn, bnc -> bc', weights, colors)
    pdf = weights/(torch.sum(weights, dim=-1, keepdim=True) + 1e-6)
    # cdf = np.cumsum(pdf)
    
    return ray_color, pdf


def hierarchical_volume_render(
        coarse_samples, coarse_densities, coarse_colors,
        fine_samples, fine_densities, fine_colors, max_depth
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

    ray_color, pdf = volume_render(samples, distances, densities, colors, max_depth)
    return ray_color, pdf, (samples, distances, densities, colors)


def rays2image(ray_colors, height, width, stride=1, scale=1, bgr=True, show=False, filename=None):
    if isinstance(ray_colors, torch.Tensor): ray_colors = ray_colors.numpy()
    img = np.zeros((height, width, 3))
    rendering = rearrange(ray_colors, '(w h) c -> h w c', w=width//stride)[::-1, :, ::-1 if bgr else 1]
    img[::stride, ::stride] = rendering
    img = (np.clip(img, 0, 1)*255).astype(np.uint8)
    if scale > 1: img = cv2.resize(img, (height*scale, width*scale), interpolation=cv2.INTER_NEAREST)

    if show:
        cv2.imshow('rendering', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if filename is not None:
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(filename, img)

    return img


if __name__ == '__main__':
    Nc = 64  # No. of coarse samples
    Nf = 128  # No. of fine samples
    min_depth, max_depth = 0, 4
    bs = 2

    def get_random_data(b, n):
        densities = torch.rand((b, n))
        colors = torch.rand((b, n, 3))
        return densities, colors

    coarse_samples, coarse_distances, bin_starts, bin_size = generate_coarse_samples(bs, Nc, min_depth, max_depth)
    densities_c, colors_c = get_random_data(bs, Nc)
    coarse_color, pdf = volume_render(coarse_samples, coarse_distances, densities_c, colors_c, max_depth)
    print(f'{coarse_color=}')

    fine_samples = generate_fine_samples(bs, Nf, bin_size, bin_starts, pdf)
    densities_f, colors_f = get_random_data(bs, Nf)
    ray_color, ray_pdf, (samples, distances, densities, colors) = hierarchical_volume_render(
        coarse_samples, densities_c, colors_c,
        fine_samples, densities_f, colors_f,
        max_depth,
    )
    print(f'{ray_color=}')
    print(f'{ray_pdf.shape=}')