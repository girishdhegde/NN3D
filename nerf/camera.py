import numpy as np
import cv2
from einops import repeat, rearrange
import torch


__author__ = '__Girish_Hegde__'


def fovx2intrinsic(fovx, height, width=None):
    """ Function get intrinsic matrix given FOV in x dimension and width

    Args:
        fovx (float): FOV in x dimension in radians.
        height (int): height of image plane in pixels.
        width (int): width of image plane in pixels. 

    Retunrs:
        torch.tensor: [3, 3] - Pinhole intrinsic matrix.
    """
    cy, cx = height/2, width/2 if width is not None else height/2
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


def get_rays(h, w, K, c2w):
    """ Function to get pixelwise rays.

    Args:
        h (int): height of the image.
        w (int): width of the image.
        K (torch.Tensor): [3, 3] - intrinsic matrix.
        c2w (torch.Tensor): [4, 4] - camera to world transformation matrix.

    Returns:
        Tuple:
            torch.Tensor: origins - [h, w, 3] - rays origins. 
            torch.Tensor: directions - [h, w, 3] - rays directions. 
    """
    u, v = torch.meshgrid(
        torch.linspace(0, w - 1, w), torch.linspace(0, h - 1, h), 
        indexing='xy'
    )  # get grid coordinates

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, -1], K[1, -1]

    directions = torch.stack([
        (u - cx)/fx,  # normalize(shift(u, cx), fx)
        (cy - v)/fy,  # -normalize(shift(v, cy), fy); -ve -> image y-directoin is reverse  
        torch.full_like(u, -1)  # -1 -> towards -z direction
    ])


    directions = torch.einsum(
        'chw, ic-> hwi', directions, c2w[:3, :3]  # Rotation@xyz
    )
    directions /=torch.norm(directions, dim=-1, keepdim=True)

    origins = c2w[:3, 3].expand(directions.shape)

    return origins, directions


# def vecs2extrinsic(eyes, fronts, ups, rights):
#     """ Function to convert camera vetors to Extrinsics.

#     Args:
#         eyes (torch.tensor): [nviews, 3] camera eyes.
#         fronts (torch.tensor): [nviews, 3] camera front/lookat unit vectors.
#         ups (torch.tensor): [nviews, 3] camera up unit vectors.
#         rights (torch.tensor): [nviews, 3] camera right unit vectos.
    
#     Returns:
#         tuple[torch.tensor]:
#             R - [nviews, 3, 3] rotations
#             t - [nviews, 3] translations
#             E - [nviews, 4, 4] extrinsics
#     """
#     R = rearrange([rights, -ups, -fronts], 'b n c -> n b c')
#     t = -torch.einsum('nij, nj -> ni', R, eyes)

#     E = torch.cat([R, t[:, :, None]], dim=-1)
#     unit = torch.tensor([0., 0, 0, 1])
#     unit = repeat(unit, 'b -> n a b', n = E.shape[0], a=1)
#     E = torch.cat((E, unit), dim=1)

#     return R, t, E


# def camera2world(points, R, t):
#     R = R.T
#     t = -R@t
#     modpts = torch.einsum('ij, nj -> ni', R, points) + t[None, :]
#     return modpts
