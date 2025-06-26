import numpy as np
import open3d as o3d

def filter_negative_z(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    mask = pts[:, 2] >= 0
    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered.colors = o3d.utility.Vector3dVector(colors[mask])
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        filtered.normals = o3d.utility.Vector3dVector(normals[mask])
    return filtered
