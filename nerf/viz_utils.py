import numpy as np
from einops import repeat, rearrange
import open3d as o3d


__author__ = '__Girish_Hegde__'


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


def create_octant_planes(span=1):
    """ Function to create open3d octant seperation planes
    """
    # Create points representing the octant separation planes
    xy_plane_points = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])*span
    xz_plane_points = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]])*span
    yz_plane_points = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]])*span

    # Create triangle mesh for the XY plane
    xy_triangles = np.array([[0, 1, 2], [0, 2, 3]])
    xy_plane_mesh = o3d.geometry.TriangleMesh()
    xy_plane_mesh.vertices = o3d.utility.Vector3dVector(xy_plane_points)
    xy_plane_mesh.triangles = o3d.utility.Vector3iVector(xy_triangles)
    xy_plane_mesh.paint_uniform_color([1, 0, 0])

    # Create triangle mesh for the XZ plane
    xz_triangles = np.array([[0, 1, 2], [0, 2, 3]])
    xz_plane_mesh = o3d.geometry.TriangleMesh()
    xz_plane_mesh.vertices = o3d.utility.Vector3dVector(xz_plane_points)
    xz_plane_mesh.triangles = o3d.utility.Vector3iVector(xz_triangles)
    xz_plane_mesh.paint_uniform_color([0, 0, 1])

    # Create triangle mesh for the YZ plane
    yz_triangles = np.array([[0, 1, 2], [0, 2, 3]])
    yz_plane_mesh = o3d.geometry.TriangleMesh()
    yz_plane_mesh.vertices = o3d.utility.Vector3dVector(yz_plane_points)
    yz_plane_mesh.triangles = o3d.utility.Vector3iVector(yz_triangles)
    yz_plane_mesh.paint_uniform_color([0, 1, 0])

    octants = xy_plane_mesh + xz_plane_mesh + yz_plane_mesh
    return octants


def spherical_viz(c2w, center=None, scene=None):
    i, j, k, eyes = c2w[:, :3, :].permute(2, 0, 1)
    nviews = eyes.shape[0]
    center = center or eyes.mean(0)

    scale = 0.25*np.linalg.norm(eyes[0] - eyes[1])
    pts = np.vstack([eyes, eyes + scale*i])
    lines = np.hstack([np.arange(nviews)[:, None], (np.arange(nviews) + nviews)[:, None]])
    ilns = to_lines(pts, lines, (1, 0, 0))

    pts = np.vstack([eyes, eyes + scale*j])
    jlns = to_lines(pts, lines, (0, 1, 0))

    pts = np.vstack([eyes, eyes + scale*k])
    klns = to_lines(pts, lines, (0, 0, 1))

    obj = to_pcd(center[None, ...], (0, 0, 0))
    pcd = to_pcd(eyes, (0, 0, 0), )

    vizobjs = [obj, pcd, ilns, jlns, klns]
    if scene is not None:
        vizobjs = vizobjs + scene
    o3d.visualization.draw_geometries(vizobjs, mesh_show_back_face=True)
    
    return vizobjs
