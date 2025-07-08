import numpy as np
import open3d as o3d


def  filter_by_min_z(
        pcd: o3d.geometry.PointCloud,
        z_min: float
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Return a copy of `pcd` keeping only points whose z-coordinate >= z_min,
    and also a per-point mask (1 = keep, 0 = discard).

    :param pcd:        Input point cloud
    :param z_min:      Minimum z-value to keep
    :returns:          (filtered point cloud, mask array of dtype uint8)
    """
    pts = np.asarray(pcd.points)
    # mask True for points we keep (z >= z_min)
    keep_mask = pts[:, 2] >= z_min

    # build filtered cloud
    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(pts[keep_mask])

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered.colors = o3d.utility.Vector3dVector(colors[keep_mask])

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        filtered.normals = o3d.utility.Vector3dVector(normals[keep_mask])

    # convert to 0/1 uint8
    mask_uint8 = keep_mask.astype(np.uint8)
    return filtered, mask_uint8
