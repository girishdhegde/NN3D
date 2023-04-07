import numpy as np
import cv2
from einops import repeat, rearrange
import torch


__author__ = '__Girish_Hegde__'


def fovx2intrinsic(fovx, width, height=None):
    """ Function get intrinsic matrix given FOV in x dimension and width

    Args:
        fovx (float): FOV in x dimension in radians.
        width (int): width of image plane in pixels. 
        height (int): height of image plane in pixels.

    Retunrs:
        np.ndarray: [3, 3] - Pinhole intrinsic matrix.
    """
    cx, cy = width/2, height/2 if height is not None else width/2
    fx = cx/np.tan(fovx/2)

    K = torch.tensor([
        [fx,  0, cx],
        [ 0, fx, cy],
        [ 0,  0,  1]
    ])

    return K



def get_spherical_poses(
        centroid = np.array([0, 0, 0]),
        nviews = 20,
        radius = 1,
        vertical_offset = 0,
        up = np.array([0, 0, 1]),
        left_handed=True,
    ):
    """ Function to get spherical camera poses vectors around a object.

    Args:
        centroid (np.ndarray): [3, ] - centroid of scene.
        nviews (int): number of views.
        radius (float): radius of pose trajectory i.e. distance of cam from centroid.
        vertical_offset (float): vertical offset of cameras in up direction wrt centroid.
        up (np.ndarray): [3, ] - up vector, must be one of x, y, z axis.
        left_handed (bool): left or right handed coordinate s/m.

    Returns:
        tuple[np.ndarray]:
            eyes - [nviews, 3] camera eyes.
            fronts - [nviews, 3] camera front/lookat unit vectors.
            ups - [nviews, 3] camera up unit vectors.
            rights - [nviews, 3] camera right unit vectos.
            i - [nviews, 3] - x unit vector 
            j - [nviews, 3] - y unit vector 
            k - [nviews, 3] - z unit vector 
    """
    ups = repeat(up, 'c -> n c', n=nviews)
    vertical_axis = int(np.where(np.array(up))[0])
    planar_axes = list({0, 1, 2} - {vertical_axis})

    thetas = np.linspace(0, 2*np.pi, nviews)
    
    eyes = np.empty((nviews, 3))
    eyes[:, vertical_axis] = np.zeros_like(thetas)
    eyes[:, planar_axes[0]] = radius*np.cos(thetas)
    eyes[:, planar_axes[1]] = radius*np.sin(thetas)
    eyes = eyes + centroid[None, :]
    eyes[:, vertical_axis] += vertical_offset

    fronts = centroid[None, :] - eyes
    fronts = fronts/np.linalg.norm(fronts, axis=-1)[:, None]
    
    rights = np.cross(up[None, :], fronts)
    rights = rights/np.linalg.norm(rights, axis=-1)[:, None]
    if left_handed: rights = -rights

    i = rights.copy()
    if not vertical_offset:
        j = np.cross(i, k)
        j = j/np.linalg.norm(j, axis=-1)[:, None]
    else:
        j = ups.copy()
    k = fronts.copy()

    return eyes, fronts, ups, rights, i, j, k


def vecs2extrinsic(eyes, fronts, ups, rights):
    """ Function to convert camera vetors to Extrinsics.

    Args:
        eyes (np.ndarray): [nviews, 3] camera eyes.
        fronts (np.ndarray): [nviews, 3] camera front/lookat unit vectors.
        ups (np.ndarray): [nviews, 3] camera up unit vectors.
        rights (np.ndarray): [nviews, 3] camera right unit vectos.
    
    Returns:
        tuple[np.ndarray]:
            R - [nviews, 3, 3] rotations
            t - [nviews, 3] translations
            E - [nviews, 4, 4] extrinsics
    """
    R = rearrange([rights, -ups, -fronts], 'b n c -> n b c')
    t = -np.einsum('nij, nj -> ni', R, eyes)

    E = np.concatenate([R, t[:, :, None]], axis=-1)
    unit = np.array([0, 0, 0, 1])
    unit = repeat(unit, 'b -> n a b', n = E.shape[0], a=1)
    E = np.concatenate((E, unit), axis=1)

    return R, t, E
