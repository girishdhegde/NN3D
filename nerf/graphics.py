import numpy as np
import cv2
import einops
import torch


__author__ = "__Girish_Hegde__"


def generate_coarse_samples(n, min_depth, max_depth):
    """ Function to generate one sample uniformly from within 'n' evenly-spaced bins 
        b/n min_depth and max_depth.

    Args:
        n (int): The number of samples to generate.
        min_depth (float): The minimum depth of the scene.
        max_depth (float): The maximum depth of the scene.

    Returns:
        tuple:
            np.ndarray[float] - samples: [n, ] - each representing the depth of a ray.
            np.ndarray[float] - distances: [n - 1, ] - each representing the distance between adjacent samples.
            np.ndarray[float] - starts [n, ]-  binwise starting points.
            float - binsize
    """
    binsize = (max_depth - min_depth) / n

    #  draw samples from 'n' evenly-spaced bins. 
    starts = np.linspace(min_depth, max_depth - binsize, n)
    samples = starts + binsize * np.random.uniform(0, 1, n)

    # calculate the distances between adjacent samples.
    distances = samples[1:] - samples[:-1]  # [N - 1]

    return samples, distances, starts, binsize


def generate_fine_samples(n, binsize, starts, prob):
    """ Function to generate samples acc. to given probability.

    Args:
        n (int): The number of samples to generate.
        binsize (float): bin size.
        starts (np.ndarray[float]): [N, ]-  binwise starting points.
        prob (np.ndarray[float]): [N, ] - binwise probability


    Returns:
        tuple:
            np.ndarray[float] - samples: [n, ] - each representing the depth of a ray.
            np.ndarray[float] - distances: [n - 1, ] - each representing the distance between adjacent samples.
    """
    samples = np.random.choice(starts, n, p=prob) + binsize*np.random.uniform(0, 1, n) 
    distances = samples[1:] - samples[:-1]
    return samples, distances


def volume_render(samples, distances, densities, colors):
    """Render a volume with the given densities, colors, and sample distances, using a ray casting algorithm.

    Args:
        samples (np.ndarray[float]): [N, ] - each representing the depth of a ray.
        densities (np.ndarray): [N, ] - where each density represents the likelihood of the corresponding ray intersecting an object.
        colors (np.ndarray): [N, 3] - where each color represents the color of the corresponding ray if it intersects an object.
        distances (np.ndarray): [N - 1, ] -where each distance represents the distance between adjacent samples.

    Returns:
        Tuple[np.ndarray, np.ndarray]: [3, ] - color of ray, and [N, ] - probability density function (PDF) of the weights.
    """
    transmittances = np.append(1, np.exp(-np.cumsum(densities[:-1]*distances)))
    alphas = 1 - np.append(np.exp(-densities[:-1]*distances), np.exp(-densities[-1]*(max_depth - samples[-1])))
    weights = transmittances*alphas

    ray_color = np.einsum('n, nc -> c', weights, colors)
    pdf = weights/np.sum(weights)
    # cdf = np.cumsum(pdf)

    return ray_color, pdf


if __name__ == '__main__':
    Nc = 32  # No. of coarse samples
    Nf = 32  # No. of fine samples
    min_depth, max_depth = 0, 4

    def get_random_data(n):
        densities = np.random.uniform(0, 1, n)
        colors = np.random.uniform(0, 1, size=(n, 3))
        return densities, colors

    coarse_samples, coarse_distances, bin_starts, bin_size = generate_coarse_samples(Nc, min_depth, max_depth)
    densities_c, colors_c = get_random_data(Nc)
    coarse_color, pdf = volume_render(coarse_samples, coarse_distances, densities_c, colors_c)
    print(f'{coarse_color=}')

    fine_samples, fine_distances = generate_fine_samples(Nf, bin_size, bin_starts, pdf)
    densities_f, colors_f = get_random_data(Nf)
    fine_color, _ = volume_render(fine_samples, fine_distances, densities_f, colors_f)
    print(f'{fine_color=}')

    inbins = np.digitize(fine_samples, bin_starts) - 1
    _, cnts = np.unique(inbins, return_counts=True)
    print(f'{pdf=}')
    print(f'{cnts/Nf=}')
