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
        torch.tensor: [3, 3] - Pinhole intrinsic matrix.
    """
    cx, cy = width/2, height/2 if height is not None else width/2
    fx = cx/np.tan(fovx/2)

    K = torch.tensor([
        [fx,  0, cx],
        [ 0, fx, cy],
        [ 0,  0,  1]
    ], dtype=torch.float32)

    return K


def get_spherical_poses(
        centroid = torch.tensor([0, 0, 0]),
        nviews = 20,
        radius = 1,
        vertical_offset = 0,
        up = torch.tensor([0, 0, 1]),
        left_handed=True,
    ):
    """ Function to get spherical camera poses vectors around a object.

    Args:
        centroid (torch.tensor): [3, ] - centroid of scene.
        nviews (int): number of views.
        radius (float): radius of pose trajectory i.e. distance of cam from centroid.
        vertical_offset (float): vertical offset of cameras in up direction wrt centroid.
        up (torch.tensor): [3, ] - up vector, must be one of x, y, z axis.
        left_handed (bool): left or right handed coordinate s/m.

    Returns:
        tuple[torch.tensor]:
            eyes - [nviews, 3] camera eyes.
            fronts - [nviews, 3] camera front/lookat unit vectors.
            ups - [nviews, 3] camera up unit vectors.
            rights - [nviews, 3] camera right unit vectos.
            i - [nviews, 3] - x unit vector 
            j - [nviews, 3] - y unit vector 
            k - [nviews, 3] - z unit vector 
    """
    ups = repeat(up, 'c -> n c', n=nviews)
    vertical_axis = int(torch.where(torch.tensor(up, dtype=torch.float32))[0])
    planar_axes = list({0, 1, 2} - {vertical_axis})

    thetas = torch.linspace(0, 2*np.pi, nviews)
    
    eyes = torch.empty((nviews, 3))
    eyes[:, vertical_axis] = torch.zeros_like(thetas)
    eyes[:, planar_axes[0]] = radius*torch.cos(thetas)
    eyes[:, planar_axes[1]] = radius*torch.sin(thetas)
    eyes = eyes + centroid[None, :]
    eyes[:, vertical_axis] += vertical_offset

    fronts = centroid[None, :] - eyes
    fronts = fronts/torch.linalg.norm(fronts, dim=-1, keepdim=True)
    
    rights = torch.cross(up[None, :], fronts)
    rights = rights/torch.linalg.norm(rights, dim=-1, keepdim=True)
    if left_handed: rights = -rights

    i = rights.clone()
    k = fronts.clone()
    if vertical_offset:
        j = torch.cross(i, k)
        j = j/torch.linalg.norm(j, dim=-1, keepdim=True)
        if j[0].dot(up) < 0: j = -j
    else:
        j = ups.clone()

    return eyes, fronts, ups, rights, i, j, k


def get_rays_from_vecs(K, eyes, i, j, k, width, height, stride=1):
    """ Funciton to generate rays from camera eyes towards image plane grid points.

    Args:
        K (torch.tensor): [3, 3] - camera intrinsic matrix.
        eyes (torch.tensor): [N, 3] - camera eyes.
        i (torch.tensor): [N, 3] - camera x-unit vector. 
        j (torch.tensor): [N, 3] - camera y-unit vector. 
        k (torch.tensor): [N, 3] - camera z-unit vector.
        width (int): image width. 
        height (int): image height.
        stride (int): skip pixels. 

    Returns:
        Tuple:
            torch.tensor: origins - [N, 3] ray origins.
            torch.tensor: directions - [N, 3] ray directions.

    Note:
        0th ray -> bottom left
        w//stride ray -> bottom right
        (w*h)//(stride*stride) ray -> top left
    """
    # get grid pixel coordinates
    u = torch.arange(0, width, stride)
    v = torch.arange(0, height, stride)
    uu, vv = torch.meshgrid(u, v)
    uv = rearrange([uu, vv], 't h w -> (h w) t')
    uv = torch.cat((uv, torch.ones((len(uv), 1), )), dim=1)

    # get 3D local camera coord s/m direction vectors
    directions = (torch.linalg.inv(K)@uv.T).T
    # get 3D world direction vectors
    directions = torch.einsum('ic, cmn -> min', directions, rearrange([i, j, k], 'm n c -> m n c'))
    # grid_points  = directions + eyes[:, None, :]

    origins = repeat(eyes, 'n c -> n m c', m=directions.shape[1])
    directions = directions/torch.linalg.norm(directions, dim=-1, keepdim=True)

    return origins, directions


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
        eyes (torch.tensor): [nviews, 3] camera eyes.
        fronts (torch.tensor): [nviews, 3] camera front/lookat unit vectors.
        ups (torch.tensor): [nviews, 3] camera up unit vectors.
        rights (torch.tensor): [nviews, 3] camera right unit vectos.
    
    Returns:
        tuple[torch.tensor]:
            R - [nviews, 3, 3] rotations
            t - [nviews, 3] translations
            E - [nviews, 4, 4] extrinsics
    """
    R = rearrange([rights, -ups, -fronts], 'b n c -> n b c')
    t = -torch.einsum('nij, nj -> ni', R, eyes)

    E = torch.cat([R, t[:, :, None]], dim=-1)
    unit = torch.tensor([0., 0, 0, 1])
    unit = repeat(unit, 'b -> n a b', n = E.shape[0], a=1)
    E = torch.cat((E, unit), dim=1)

    return R, t, E


def camera2world(points, R, t):
    R = R.T
    t = -R@t
    modpts = torch.einsum('ij, nj -> ni', R, points) + t[None, :]
    return modpts
