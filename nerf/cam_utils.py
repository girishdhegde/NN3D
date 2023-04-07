import numpy as np
import cv2
from einops import repeat, rearrange
import open3d as o3d
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


def to_pcd(points, colors=None, normals=None, viz=False, filepath=None, name='Viz'):
    """ Function to convert points array into o3d.PointCloud

    Args:
        points (np.ndarray): [N, 3] - list of xyz of points.
        colors (np.ndarray/List): [N, 3] pcd colors or [r, g, b]. Defaults to None.
        normals (np.ndarray): [N, 3] point normals. Defaults to None.
        viz (bool): show point cloud. Defaults to False.
        filepath (str): save point cloud as. Defaults to None.
        name (str): window name.

    Returns:
        (o3d.PointCloud): point cloud
    """
    vec3 = o3d.utility.Vector3dVector
    pcd = o3d.geometry.PointCloud(vec3(points))
    if normals is not None: pcd.normals = vec3(normals)
    if colors is not None:
        colors = np.array(colors)
        if len(colors.shape) > 1:
            pcd.colors =  vec3(colors)
        else:
            pcd.paint_uniform_color(colors)
    if viz: o3d.visualization.draw_geometries([pcd], name)
    if filepath is not None: o3d.io.write_point_cloud(filepath, pcd)
    return pcd


def to_mesh(points, faces, colors=None, viz=False, filepath=None, name='Viz'):
    """ Function to convert points array into o3d.geometry.TriangleMesh

    Args:
        points (np.ndarray): [N, 3] - list of xyz of points.
        faces (np.ndarray): [M, 3] - list of triangle faces of points.
        colors (np.ndarray/List, optional): [N, 3] pcd colors or [r, g, b]. Defaults to None.
        viz (bool, optional): show point cloud. Defaults to False.
        filepath (str, optional): save point cloud as. Defaults to None.
        name (str): window name.

    Returns:
        (o3d.geometry.TriangleMesh): mesh
    """
    mesh = o3d.geometry.TriangleMesh()
    vec3 = o3d.utility.Vector3dVector
    mesh.vertices = vec3(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        colors = np.array(colors)
        if len(colors.shape) > 1:
            mesh.vertex_colors =  vec3(colors)
        else:
            mesh.paint_uniform_color(colors)
    if viz: o3d.visualization.draw_geometries([mesh], name, mesh_show_back_face=True)
    if filepath is not None: o3d.io.write_triangle_mesh(filepath, mesh)
    return mesh


def to_lines(points, edges, colors=None, viz=False, filepath=None, name='Viz'):
    """ Function to convert points and edges into o3d.geometry.LineSet

    Args:
        points (np.ndarray[float]): [N, 3] - list of xyz of points.
        edges (np.ndarray[int]): [M, 2] - list of edges.
        colors (np.ndarray/List, optional): [N, 3] edge colors or [r, g, b].
        viz (bool, optional): show point cloud. Defaults to False.
        filepath (str, optional): save point cloud as. Defaults to None.
        name (str): window name.

    Returns:
        (o3d.geometry.LineSet): lines
    """
    lines = o3d.geometry.LineSet()
    vec3 = o3d.utility.Vector3dVector
    lines.points = vec3(points)
    lines.lines = o3d.utility.Vector2iVector(edges)
    if colors is not None:
        colors = np.array(colors)
        if len(colors.shape) > 1:
            lines.colors =  vec3(colors)
        else:
            lines.paint_uniform_color(colors)
    if viz: o3d.visualization.draw_geometries([lines], name, mesh_show_back_face=True)
    if filepath is not None: o3d.io.write_line_set(filepath, lines)
    return lines


def spherical_viz(centroid, eyes, fronts, ups, rights, scene=None):
    nviews = eyes.shape[0]
    scale = 0.25*np.linalg.norm(eyes[0] - eyes[1])
    pts = np.vstack([eyes, eyes + scale*2*fronts])
    lines = np.hstack([np.arange(nviews)[:, None], (np.arange(nviews) + nviews)[:, None]])
    lookatlns = to_lines(pts, lines, (0, 0, 1))

    pts = np.vstack([eyes, eyes + scale*rights])
    rightlns = to_lines(pts, lines, (1, 0, 0))

    pts = np.vstack([eyes, eyes + scale*ups])
    uplns = to_lines(pts, lines, (0, 1, 0))

    obj = to_pcd(centroid[None, ...], (0, 0, 0))
    pcd = to_pcd(eyes, (0, 0, 0), )

    vizobjs = [obj, pcd, lookatlns, rightlns, uplns]
    if scene is not None:
        vizobjs = vizobjs + scene
    o3d.visualization.draw_geometries(vizobjs, mesh_show_back_face=True)
    
    return vizobjs