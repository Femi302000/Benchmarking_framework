from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


def cluster_1d(values: np.ndarray, eps: float, min_points: int) -> list[np.ndarray]:
    if values.size == 0:
        return []
    order = np.argsort(values)
    v = values[order]
    clusters: list[np.ndarray] = []
    start = 0
    for i in range(1, len(v)):
        if (v[i] - v[i-1]) > eps:
            idxs = order[start:i]
            if idxs.size >= min_points:
                clusters.append(idxs)
            start = i
    idxs = order[start:len(v)]
    if idxs.size >= min_points:
        clusters.append(idxs)
    return clusters


def pick_nose_cluster_by_range_with_r(r: np.ndarray, pts: np.ndarray, eps: float, min_points: int):
    if r.shape[0] != pts.shape[0]:
        raise ValueError("Length of r must match number of points")
    clusters = cluster_1d(r, eps=eps, min_points=min_points)
    if not clusters:
        idx = int(np.argmin(r))
        return pts[idx], idx, float(r[idx]), []
    means = [float(np.mean(r[c])) for c in clusters]
    c_idx = int(np.argmin(means))
    chosen = clusters[c_idx]
    local_idx = int(chosen[np.argmin(r[chosen])])
    return pts[local_idx], local_idx, float(r[local_idx]), clusters


def pick_closest_cluster(pts: np.ndarray, clusters: list[np.ndarray]) -> np.ndarray | None:
    if not clusters:
        return None
    best, best_min = None, float('inf')
    for c in clusters:
        dmin = float(np.linalg.norm(pts[c], axis=1).min())
        if dmin < best_min:
            best_min, best = dmin, c
    return best