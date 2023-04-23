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
        center = torch.tensor([0, 0, 0]),
        nviews = 20,
        radius = 1,
        vertical_offset = 0,
    ):
    """ Function to get spherical camera poses around a object.

    Args:
        center (torch.tensor): [3, ] - center of scene.
        nviews (int): number of views.
        radius (float): radius of pose trajectory i.e. distance of cam from center.
        vertical_offset (float): vertical offset of cameras in up direction wrt center.

    Returns:
        torch.tensor: view_mats - [nviews, 4, 4] extrinsics/camera to world matrix.
    """
    up = torch.tensor([0, 0, 1.])  # +z direction 
    ups = repeat(up, 'c -> n c', n=nviews)
    thetas = torch.linspace(0, 2*np.pi, nviews)

    eyes = torch.stack([
            radius*torch.cos(thetas), 
            radius*torch.sin(thetas),
            torch.zeros_like(thetas),
        ], -1
    )

    eyes = eyes + center[None, :]
    eyes[:, 2] += vertical_offset

    look_ats = center[None, :] - eyes
    look_ats /= torch.norm(look_ats, dim=-1, keepdim=True)

    rights = torch.cross(ups, look_ats)
    rights /= torch.norm(rights, dim=-1, keepdim=True)

    ups = torch.cross(look_ats, rights)
    ups /= torch.norm(ups, dim=-1, keepdim=True)

    view_mats = torch.stack([-rights, ups, -look_ats, eyes], -1)
    temp = torch.zeros((nviews, 4, 4))
    temp[:, -1, -1] = 1.
    temp[:, :3, :] = view_mats
    view_mats = temp
    
    return view_mats


def get_spiral_poses(
        center = torch.tensor([0, 0, 0]),
        nviews = 20,
        radius = 1,
        vertical_range = (0, 1),
        rotations = 1,
    ):
    """ Function to get spiral camera poses around a object.

    Args:
        center (torch.tensor): [3, ] - center of scene.
        nviews (int): number of views.
        radius (float): radius of pose trajectory i.e. distance of cam from center.
        vertical_range (float): spiral vertical range in up direction wrt center.
        rotations (int): number of rotations around the object

    Returns:
        torch.tensor: view_mats - [nviews, 4, 4] extrinsics/camera to world matrix.
    """
    up = torch.tensor([0, 0, 1.])  # +z direction 
    ups = repeat(up, 'c -> n c', n=nviews)
    thetas = torch.linspace(0, 2*np.pi*rotations, nviews)

    eyes = torch.stack([
            radius*torch.cos(thetas), 
            radius*torch.sin(thetas),
            torch.zeros_like(thetas),
        ], -1
    )

    eyes = eyes + center[None, :]
    eyes[:, 2] += torch.linspace(vertical_range[0], vertical_range[1], nviews)

    look_ats = center[None, :] - eyes
    look_ats /= torch.norm(look_ats, dim=-1, keepdim=True)

    rights = torch.cross(ups, look_ats)
    rights /= torch.norm(rights, dim=-1, keepdim=True)

    ups = torch.cross(look_ats, rights)
    ups /= torch.norm(ups, dim=-1, keepdim=True)

    view_mats = torch.stack([-rights, ups, -look_ats, eyes], -1)
    temp = torch.zeros((nviews, 4, 4))
    temp[:, -1, -1] = 1.
    temp[:, :3, :] = view_mats
    view_mats = temp
    
    return view_mats


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
