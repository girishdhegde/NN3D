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
    k = fronts.copy()
    if vertical_offset:
        j = np.cross(i, k)
        j = j/np.linalg.norm(j, axis=-1)[:, None]
        if j[0].dot(up) < 0: j = -j
    else:
        j = ups.copy()

    return eyes, fronts, ups, rights, i, j, k


def get_rays_from_vecs(K, eyes, i, j, k, width, height, stride=1):
    """ Funciton to generate rays from camera eyes towards image plane grid points.

    Args:
        K (np.ndarray): [3, 3] - camera intrinsic matrix.
        eyes (np.ndarray): [N, 3] - camera eyes.
        i (np.ndarray): [N, 3] - camera x-unit vector. 
        j (np.ndarray): [N, 3] - camera y-unit vector. 
        k (np.ndarray): [N, 3] - camera z-unit vector.
        width (int): image width. 
        height (int): image height.
        stride (int): skip pixels. 

    Returns:
        Tuple:
            np.ndarray: origins - [N, 3] ray origins.
            np.ndarray: directions - [N, 3] ray directions.
            np.ndarray: grid_points - [N, 3] ray endings.

    Note:
        0th ray -> bottom left
        w//stride ray -> bottom right
        (w*h)//(stride*stride) ray -> top left
    """
    # get grid pixel coordinates
    u = np.arange(0, width, stride)
    v = np.arange(0, height, stride)
    uu, vv = np.meshgrid(u, v)
    uv = rearrange([uu, vv], 't h w -> (w h) t')
    uv = np.concatenate((uv, np.ones((len(uv), 1), )), axis=1)

    # get 3D local camera coord s/m grid points
    grid_points = (np.linalg.inv(K)@uv.T).T

    # get 3D world grid points
    grid_points = np.einsum('ic, cmn -> min', grid_points, rearrange([i, j, k], 'm n c -> m n c'))
    grid_points  = grid_points + eyes[:, None, :]

    # get rays
    origins = repeat(eyes, 'n c -> n m c', m=grid_points.shape[1])
    directions = grid_points - origins
    directions = directions/np.linalg.norm(directions, axis=-1)[..., None]

    return origins, directions, grid_points


# https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
def get_rays_from_extrinsics(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


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


def camera2world(points, R, t):
    R = R.T
    t = -R@t
    modpts = np.einsum('ij, nj -> ni', R, points) + t[None, :]
    return modpts
