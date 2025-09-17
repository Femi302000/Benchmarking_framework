from __future__ import annotations
import os
import numpy as np
import open3d as o3d


def load_points_pcd(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PCD file not found: {path}")
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError(f"No points in: {path}")
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask].astype(np.float32)


def denoise_points(pts: np.ndarray, method: str, sor_nb_neighbors: int, sor_std_ratio: float, ror_radius: float, ror_min_neighbors: int):
    if method == "none":
        return pts, np.ones(len(pts), dtype=bool)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    if method == "statistical":
        _, ind = pcd.remove_statistical_outlier(sor_nb_neighbors, sor_std_ratio)
    elif method == "radius":
        _, ind = pcd.remove_radius_outlier(ror_min_neighbors, ror_radius)
    else:
        raise ValueError("Invalid denoise_method")
    mask = np.zeros(len(pts), dtype=bool)
    mask[np.asarray(ind, dtype=int)] = True
    clean = pts[mask]
    if clean.size == 0:
        raise ValueError("All points removed by denoising.")
    return clean, mask