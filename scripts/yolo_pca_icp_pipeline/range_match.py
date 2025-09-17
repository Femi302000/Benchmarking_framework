from __future__ import annotations
from typing import Tuple
import numpy as np
import open3d as o3d


def _hash_key(row: np.ndarray, decimals: int) -> tuple[int, int, int]:
    q = np.round(row, decimals=decimals)
    scale = 10 ** decimals
    return (int(np.round(q[0]*scale)), int(np.round(q[1]*scale)), int(np.round(q[2]*scale)))


def match_ranges_from_h5_to_pcd(h5_xyz: np.ndarray, h5_range: np.ndarray, pcd_xyz: np.ndarray,
                                round_decimals: int, kdtree_tol: float) -> tuple[np.ndarray, np.ndarray]:
    M = pcd_xyz.shape[0]
    r_for_pcd = np.full(M, np.nan, dtype=np.float32)
    matched_mask = np.zeros(M, dtype=bool)

    # 1) rounded hashmap
    hashmap = {}
    for i, p in enumerate(h5_xyz):
        hashmap[_hash_key(p, round_decimals)] = i
    for j, q in enumerate(pcd_xyz):
        idx = hashmap.get(_hash_key(q, round_decimals), None)
        if idx is not None:
            r_for_pcd[j] = h5_range[idx]
            matched_mask[j] = True

    # 2) KD-tree fallback
    remaining = np.where(~matched_mask)[0]
    if remaining.size:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(h5_xyz.astype(np.float64)))
        kdt = o3d.geometry.KDTreeFlann(cloud)
        tol2 = float(kdtree_tol)**2
        for j in remaining:
            q = pcd_xyz[j].astype(np.float64)
            k, idxs, d2 = kdt.search_knn_vector_3d(q, 1)
            if k == 1 and d2[0] <= tol2:
                r_for_pcd[j] = h5_range[idxs[0]]
                matched_mask[j] = True
    return r_for_pcd, matched_mask