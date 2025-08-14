#!/usr/bin/env python3
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from scipy.optimize import least_squares
import argparse
import json
import os

# -------------------------
# Utilities
# -------------------------

def minmax_norm(v: np.ndarray) -> np.ndarray:
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin < 1e-9:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


def _nearest_point_to(P: np.ndarray, target: np.ndarray, idxs: np.ndarray) -> int:
    """Return global index of the point in P[idxs] nearest to target (L2, in XYZ)."""
    if idxs.size == 0:
        return -1
    diffs = P[idxs] - target[None, :]
    j = np.argmin(np.einsum("ij,ij->i", diffs, diffs))
    return int(idxs[j])


def pointcloud_to_spheres(pcd, radius=0.2):
    """Convert each point in pcd to a small colored sphere mesh."""
    spheres = []
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(pts), 3))
    for p, c in zip(pts, cols):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color(c)
        sphere.translate(p)
        spheres.append(sphere)
    return spheres


# -------------------------
# Cleaning + Alignment
# -------------------------

def remove_ground_and_noise(pcd: o3d.geometry.PointCloud, ground_thresh=0.05):
    """Statistical outlier removal then plane segmentation to drop ground."""
    if len(pcd.points) == 0:
        return pcd
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    if len(pcd.points) == 0:
        return pcd
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=ground_thresh, ransac_n=3, num_iterations=1000
    )
    # Remove ground
    pcd_no_ground = pcd.select_by_index(inliers, invert=True)
    return pcd_no_ground


def align_pcd_pca(pcd: o3d.geometry.PointCloud):
    """Center and rotate so principal axes map to (X=fuselage, Y=wingspan, Z=height)."""
    pts = np.asarray(pcd.points)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]  # columns are principal axes (X->PC1, Y->PC2, Z->PC3)
    aligned = centered @ eigvecs
    # Force nose in +X direction (make front be the more "pointy/sparse" side)
    X = aligned[:, 0]
    hi = X > np.percentile(X, 90)
    lo = X < np.percentile(X, 10)
    if np.mean(X[hi]) < np.mean(X[lo]):
        aligned[:, 0] *= -1
        aligned[:, 1] *= -1  # preserve right-handedness (flip Y too)
        eigvecs[:, 0] *= -1
        eigvecs[:, 1] *= -1

    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned)
    return aligned_pcd, eigvecs, mean


# -------------------------
# Coloring / Clustering helpers
# -------------------------

def color_xy_z(
    pcd: o3d.geometry.PointCloud,
    mode: str = "rgb_xyz",         # "rgb_xyz" | "hsv_xy_z" | "xycluster_z" | "xycluster_zlevels"
    eps_xy: float = 0.4,
    min_xy: int = 30,
    z_bin_size: float = 0.25,
) -> o3d.geometry.PointCloud:
    P = np.asarray(pcd.points)
    if P.size == 0:
        return pcd
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    Zn = minmax_norm(Z)

    if mode == "rgb_xyz":
        Xn, Yn = minmax_norm(X), minmax_norm(Y)
        colors = np.c_[Xn, Yn, Zn]

    elif mode == "hsv_xy_z":
        ang = np.arctan2(Y - Y.mean(), X - X.mean())
        H = (ang + np.pi) / (2 * np.pi)
        S = np.ones_like(H)
        V = Zn
        h6 = H * 6.0
        i = np.floor(h6).astype(int) % 6
        f = h6 - i
        p = V * (1 - S)
        q = V * (1 - f * S)
        t = V * (1 - (1 - f) * S)
        rgb = np.zeros((len(P), 3))
        masks_and_vals = [
            (i == 0, np.c_[V, t, p]),
            (i == 1, np.c_[q, V, p]),
            (i == 2, np.c_[p, V, t]),
            (i == 3, np.c_[p, q, V]),
            (i == 4, np.c_[t, p, V]),
            (i == 5, np.c_[V, p, q]),
        ]
        for m, val in masks_and_vals:
            rgb[m] = val[m]
        colors = rgb

    elif mode == "xycluster_z":
        XY = P[:, :2]
        labels = DBSCAN(eps=eps_xy, min_samples=min_xy).fit_predict(XY)
        uniq = [l for l in np.unique(labels) if l != -1]
        rng = np.random.default_rng(42)
        palette = rng.random((max(len(uniq), 1), 3))
        shade = 0.5 + 0.5 * Zn
        colors = np.zeros((len(P), 3)) + 0.25
        cluster_to_color = {lab: palette[k % len(palette)] for k, lab in enumerate(uniq)}
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            if lab == -1:
                colors[idx] = np.c_[shade[idx], shade[idx], shade[idx]] * 0.3
            else:
                base = cluster_to_color[lab]
                colors[idx] = base * shade[idx][:, None]

    elif mode == "xycluster_zlevels":
        XY = P[:, :2]
        labels = DBSCAN(eps=eps_xy, min_samples=min_xy).fit_predict(XY)
        cluster_ids = [lab for lab in np.unique(labels) if lab != -1]
        cl_mean_z = {lab: Z[labels == lab].mean() for lab in cluster_ids}
        if z_bin_size <= 0:
            raise ValueError("z_bin_size must be > 0")
        cl_level = {lab: int(np.floor(cl_mean_z[lab] / z_bin_size)) for lab in cluster_ids}
        level_ids = sorted(set(cl_level.values()))
        rng = np.random.default_rng(123)
        level_palette = {lev: rng.random(3) for lev in level_ids}
        colors = np.zeros((len(P), 3)) + 0.25
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            if lab == -1:
                shade = 0.4 + 0.2 * Zn[idx]
                colors[idx] = np.c_[shade, shade, shade]
            else:
                lev = cl_level[lab]
                base = level_palette[lev]
                colors[idx] = base

    else:
        raise ValueError("Invalid mode")

    out = o3d.geometry.PointCloud(pcd)
    out.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    return out


def dbscan_xy(P: np.ndarray, eps_xy=0.35, min_xy=40):
    """Convenience: run DBSCAN on XY, return labels and cluster id list (exclude -1)."""
    labels = DBSCAN(eps=eps_xy, min_samples=min_xy).fit_predict(P[:, :2])
    cluster_ids = [lab for lab in np.unique(labels) if lab != -1]
    return labels, cluster_ids


# -------------------------
# Semantic keypoint detection (combined logic)
# -------------------------

def detect_aircraft_keypoints(
    pcd: o3d.geometry.PointCloud,
    # cleaning
    ground_thresh=0.05,
    # clustering
    eps_xy=0.35,
    min_xy=40,
    # level pairing
    z_bin_size=0.5,
    require_level_pair=True,
    # wing/engine bands (in aligned Z)
    wing_z_band=(40, 60),   # retained for backwards compat; new logic uses highest Z level
    engine_z_band=(40, 60),
    gear_z_pct=5,
):
    """
    Returns:
        keypoints_world: {name: (xyz in ORIGINAL coords)}
        debug: dict with intermediate arrays (aligned points, transforms, labels, etc.)
               ALSO includes keypoints_aligned for fitting the skeleton.
    """
    # 1) Clean
    pcd_clean = remove_ground_and_noise(pcd, ground_thresh=ground_thresh)
    if len(pcd_clean.points) == 0:
        return {}, {"error": "Empty point cloud after cleaning."}

    # 2) PCA align
    aligned_pcd, eigvecs, mean = align_pcd_pca(pcd_clean)
    A = np.asarray(aligned_pcd.points)
    X, Y, Z = A[:, 0], A[:, 1], A[:, 2]

    # 3) XY clustering for stable candidates
    labels, cluster_ids = dbscan_xy(A, eps_xy=eps_xy, min_xy=min_xy)

    # Compute cluster means (XYZ) and a mapping cluster->indices
    cl_to_indices = {lab: np.where(labels == lab)[0] for lab in cluster_ids}
    cl_means = {lab: A[idxs].mean(axis=0) for lab, idxs in cl_to_indices.items()}

    # Optionally: only keep "levels" (Z bins) that have ≥2 clusters
    if z_bin_size <= 0:
        raise ValueError("z_bin_size must be > 0")
    cl_level = {lab: int(np.floor(cl_means[lab][2] / z_bin_size)) for lab in cluster_ids}
    if require_level_pair:
        level_counts = {}
        for lab in cluster_ids:
            lev = cl_level[lab]
            level_counts[lev] = level_counts.get(lev, 0) + 1
        kept_levels = {lev for lev, c in level_counts.items() if c >= 2}
        kept_clusters = [lab for lab in cluster_ids if cl_level[lab] in kept_levels]
    else:
        kept_levels = set(cl_level[lab] for lab in cluster_ids)
        kept_clusters = cluster_ids[:]

    # Helper to pick cluster near a target condition from kept clusters
    def _closest_cluster(mask_indices: np.ndarray, prefer_extreme_axis=None, extreme="max"):
        """Pick index of point closest to mean of masked region, but choose cluster whose mean best matches an extreme if requested."""
        if mask_indices.size == 0 or len(kept_clusters) == 0:
            return None
        region_centroid = A[mask_indices].mean(axis=0)
        # If an extreme axis is specified, bias toward cluster with most extreme mean on that axis.
        candidate_clusters = kept_clusters[:]
        if prefer_extreme_axis is not None:
            axis = {"x": 0, "y": 1, "z": 2}[prefer_extreme_axis]
            if extreme == "max":
                candidate_clusters.sort(key=lambda c: cl_means[c][axis], reverse=True)
            else:
                candidate_clusters.sort(key=lambda c: cl_means[c][axis], reverse=False)
        # Pick nearest point to region centroid but restricted to the best-ranked cluster first
        for lab in candidate_clusters:
            idxs = cl_to_indices[lab]
            kp_idx = _nearest_point_to(A, region_centroid, idxs)
            if kp_idx >= 0:
                return kp_idx
        return None

    keypoints_aligned = {}

    # Nose via X max
    nose_mask = X > np.percentile(X, 98)
    nose_idx = _closest_cluster(np.where(nose_mask)[0], prefer_extreme_axis="x", extreme="max")
    if nose_idx is not None:
        keypoints_aligned["nose"] = A[nose_idx]

    # Tail (optional)
    if "nose" in keypoints_aligned:
        nose_x = keypoints_aligned["nose"][0]
        x_tail_thresh = np.percentile(X, 2)
        span_x = np.percentile(X, 98) - np.percentile(X, 2)
        delta_required = max(0.5, 0.75 * span_x)
        valid_tail_mask = (X < x_tail_thresh) & ((nose_x - X) > delta_required)
        tail_idx = _closest_cluster(np.where(valid_tail_mask)[0], prefer_extreme_axis="x", extreme="min")
        if tail_idx is not None:
            keypoints_aligned["tail"] = A[tail_idx]

    # Wing tips — choose the HIGHEST Z level that has >=2 clusters (high‑wing assumption)
    if len(kept_clusters) > 0:
        level_to_clusters = {}
        for lab in kept_clusters:
            lev = cl_level[lab]
            level_to_clusters.setdefault(lev, []).append(lab)

        candidate_levels = [lev for lev, labs in level_to_clusters.items() if len(labs) >= 2]
        if len(candidate_levels) > 0:
            top_level = max(candidate_levels)  # highest Z level
            labs = level_to_clusters[top_level]

            # Sort by mean Y to find lateral extremes
            labs_sorted_left = sorted(labs, key=lambda c: cl_means[c][1])          # Y ascending
            labs_sorted_right = sorted(labs, key=lambda c: cl_means[c][1], reverse=True)

            left_lab = labs_sorted_left[0]
            right_lab = labs_sorted_right[0]

            # within each chosen cluster, take the most lateral point
            left_idxs = cl_to_indices[left_lab]
            right_idxs = cl_to_indices[right_lab]
            if left_idxs.size > 0:
                left_point = A[left_idxs[np.argmin(A[left_idxs, 1])]]  # min Y
                keypoints_aligned["wing_left"] = left_point
            if right_idxs.size > 0:
                right_point = A[right_idxs[np.argmax(A[right_idxs, 1])]]  # max Y
                keypoints_aligned["wing_right"] = right_point

    # Landing gear — lowest Z band, split by Y
    gz = np.percentile(Z, gear_z_pct)
    gear_mask = Z < gz
    lg_left = np.where(gear_mask & (Y < 0))[0]
    lg_right = np.where(gear_mask & (Y > 0))[0]
    lg_left_idx = _closest_cluster(lg_left, prefer_extreme_axis="z", extreme="min")
    lg_right_idx = _closest_cluster(lg_right, prefer_extreme_axis="z", extreme="min")
    if lg_left_idx is not None:
        keypoints_aligned["gear_left"] = A[lg_left_idx]
    if lg_right_idx is not None:
        keypoints_aligned["gear_right"] = A[lg_right_idx]

    # Engines — near center Y, mid Z band
    ez0, ez1 = np.percentile(Z, engine_z_band[0]), np.percentile(Z, engine_z_band[1])
    eng_mask = (Z > ez0) & (Z < ez1) & (np.abs(Y) < np.percentile(np.abs(Y), 30))
    if np.count_nonzero(eng_mask):
        eng_left_idx = _closest_cluster(np.where(eng_mask & (Y < 0))[0], prefer_extreme_axis="y", extreme="min")
        eng_right_idx = _closest_cluster(np.where(eng_mask & (Y > 0))[0], prefer_extreme_axis="y", extreme="max")
        if eng_left_idx is not None:
            keypoints_aligned["engine_left"] = A[eng_left_idx]
        if eng_right_idx is not None:
            keypoints_aligned["engine_right"] = A[eng_right_idx]

    # Map to ORIGINAL coords
    keypoints_world = {k: (v @ eigvecs.T) + mean for k, v in keypoints_aligned.items()}

    debug = {
        "aligned_points": A,
        "eigvecs": eigvecs,
        "mean": mean,
        "labels": labels,
        "cluster_ids": cluster_ids,
        "kept_levels": kept_levels,
        "cl_level": cl_level,
        "keypoints_aligned": keypoints_aligned,  # <--- for skeleton fitting
    }
    return keypoints_world, debug


# -------------------------
# Skeleton model + fitting
# -------------------------

@dataclass
class AircraftParams:
    L_fuse: float          # fuselage length
    span: float            # total wingspan
    x_wing_frac: float     # wing station along fuselage [0..1] (0=tail, 1=nose)
    z_wing: float          # wing vertical level
    x_engine_frac: float   # engine station
    y_engine_frac: float   # engine lateral offset as fraction of half-span [0..1]
    z_engine: float        # engine vertical level
    x_gear_frac: float     # gear station
    y_gear_frac: float     # gear lateral offset as fraction of half-span [0..1]
    z_gear: float          # gear vertical level


def skeleton_nodes_aligned(p: AircraftParams) -> Dict[str, np.ndarray]:
    """Return parametric skeleton node positions in aligned PCA frame."""
    # put fuselage center at x=0, nose at +L/2, tail at -L/2
    L2 = 0.5 * p.L_fuse
    half_span = 0.5 * p.span
    x_wing = -L2 + p.x_wing_frac * p.L_fuse
    x_eng  = -L2 + p.x_engine_frac * p.L_fuse
    x_gear = -L2 + p.x_gear_frac * p.L_fuse
    y_eng  = p.y_engine_frac * half_span
    y_gear = p.y_gear_frac * half_span

    nodes = {
        "nose":        np.array([+L2, 0.0, 0.0]),
        "tail":        np.array([-L2, 0.0, 0.0]),
        "wing_left":   np.array([x_wing, -half_span, p.z_wing]),
        "wing_right":  np.array([x_wing, +half_span, p.z_wing]),
        "engine_left": np.array([x_eng,  -y_eng,     p.z_engine]),
        "engine_right":np.array([x_eng,  +y_eng,     p.z_engine]),
        "gear_left":   np.array([x_gear, -y_gear,    p.z_gear]),
        "gear_right":  np.array([x_gear, +y_gear,    p.z_gear]),
    }
    return nodes


def pack_params(p: AircraftParams) -> np.ndarray:
    return np.array([
        p.L_fuse, p.span, p.x_wing_frac, p.z_wing,
        p.x_engine_frac, p.y_engine_frac, p.z_engine,
        p.x_gear_frac, p.y_gear_frac, p.z_gear
    ], dtype=float)


def unpack_params(x: np.ndarray) -> AircraftParams:
    return AircraftParams(
        L_fuse=float(x[0]), span=float(x[1]), x_wing_frac=float(x[2]), z_wing=float(x[3]),
        x_engine_frac=float(x[4]), y_engine_frac=float(x[5]), z_engine=float(x[6]),
        x_gear_frac=float(x[7]), y_gear_frac=float(x[8]), z_gear=float(x[9])
    )


def initial_guess_from_points(A: np.ndarray,
                              kpa: Optional[Dict[str, np.ndarray]] = None) -> AircraftParams:
    """Heuristics from aligned points + optional aligned keypoints."""
    X, Y, Z = A[:,0], A[:,1], A[:,2]
    L_fuse = max(1e-3, float(np.percentile(X, 98) - np.percentile(X, 2)))
    span   = max(1e-3, float(2*np.percentile(np.abs(Y), 98)))

    # Defaults
    x_wing_frac = 0.45
    z_wing = float(np.percentile(Z, 85))
    x_engine_frac = 0.40
    y_engine_frac = 0.25
    z_engine = float(np.percentile(Z, 55))
    x_gear_frac = 0.20
    y_gear_frac = 0.30
    z_gear = float(np.percentile(Z, 5))

    # If keypoints exist, refine guesses
    if kpa:
        if "nose" in kpa and "tail" in kpa:
            L_fuse = float(np.linalg.norm(kpa["nose"] - kpa["tail"]))
        wl = kpa.get("wing_left"); wr = kpa.get("wing_right")
        if wl is not None and wr is not None:
            span = float(abs(wr[1] - wl[1]))
            z_wing = float(0.5*(wl[2] + wr[2]))
            x_wing_frac = float((0.5*(wl[0]+wr[0]) + 0.5*L_fuse) / L_fuse) if L_fuse > 1e-6 else x_wing_frac
        gl = kpa.get("gear_left"); gr = kpa.get("gear_right")
        if gl is not None and gr is not None:
            z_gear = float(0.5*(gl[2] + gr[2]))
            y_gear_frac = min(0.95, float( (abs(gl[1])+abs(gr[1])) / span )) if span > 1e-6 else y_gear_frac
            x_gear = float(0.5*(gl[0]+gr[0]))
            x_gear_frac = float((x_gear + 0.5*L_fuse) / L_fuse) if L_fuse > 1e-6 else x_gear_frac
        el = kpa.get("engine_left"); er = kpa.get("engine_right")
        if el is not None and er is not None:
            z_engine = float(0.5*(el[2] + er[2]))
            y_engine_frac = min(0.95, float( (abs(el[1])+abs(er[1])) / span )) if span > 1e-6 else y_engine_frac
            x_eng = float(0.5*(el[0]+er[0]))
            x_engine_frac = float((x_eng + 0.5*L_fuse) / L_fuse) if L_fuse > 1e-6 else x_engine_frac

    return AircraftParams(
        L_fuse=L_fuse,
        span=span,
        x_wing_frac=float(np.clip(x_wing_frac, 0, 1)),
        z_wing=z_wing,
        x_engine_frac=float(np.clip(x_engine_frac, 0, 1)),
        y_engine_frac=float(np.clip(y_engine_frac, 0, 1)),
        z_engine=z_engine,
        x_gear_frac=float(np.clip(x_gear_frac, 0, 1)),
        y_gear_frac=float(np.clip(y_gear_frac, 0, 1)),
        z_gear=z_gear,
    )


def _residuals(x, kp_aligned: Dict[str, np.ndarray], w: Dict[str, float]):
    p = unpack_params(x)
    nodes = skeleton_nodes_aligned(p)
    r = []
    for name, pt in kp_aligned.items():
        if name in nodes:
            r.append((np.linalg.norm(nodes[name] - pt)) * w.get(name, 1.0))
    # soft constraints (sanity)
    r += [
        10.0 * max(0.0, -p.L_fuse),
        10.0 * max(0.0, -p.span),
        2.0 * max(0.0, p.z_gear - p.z_wing),   # gear should be below wing
    ]
    return np.array(r, dtype=float)


def fit_skeleton_to_keypoints(aligned_points: np.ndarray,
                              keypoints_aligned: Dict[str, np.ndarray]):
    if not keypoints_aligned:
        return None, None, np.inf, None

    # weights (increase if you trust certain detections more)
    weights = {
        "nose": 2.0, "tail": 1.5,
        "wing_left": 3.0, "wing_right": 3.0,
        "engine_left": 1.0, "engine_right": 1.0,
        "gear_left": 1.5, "gear_right": 1.5,
    }

    p0 = initial_guess_from_points(aligned_points, keypoints_aligned)
    x0 = pack_params(p0)

    eps = 1e-2
    lb = np.array([eps, eps, 0.0, -np.inf, 0.0, 0.0, -np.inf, 0.0, 0.0, -np.inf])
    ub = np.array([np.inf, np.inf, 1.0,  np.inf, 1.0, 1.0,  np.inf, 1.0, 1.0,  np.inf])

    res = least_squares(
        _residuals, x0, bounds=(lb, ub),
        args=(keypoints_aligned, weights), loss="soft_l1", f_scale=0.1, max_nfev=300
    )
    p_opt = unpack_params(res.x)
    nodes_opt = skeleton_nodes_aligned(p_opt)
    rmse = float(np.sqrt(np.mean(res.fun**2))) if res.fun.size else np.inf
    return p_opt, nodes_opt, rmse, res


def skeleton_lineset(nodes: Dict[str, np.ndarray]) -> o3d.geometry.LineSet:
    """A minimal wireframe for visualization."""
    # index nodes
    names = ["nose","tail","wing_left","wing_right","engine_left","engine_right","gear_left","gear_right"]
    idx = {n:i for i,n in enumerate(names)}
    pts = np.array([nodes[n] for n in names], dtype=float)

    # lines: fuselage spine + wing + engines+gears span lines
    lines = [
        [idx["tail"], idx["nose"]],
        [idx["wing_left"], idx["wing_right"]],
        [idx["engine_left"], idx["engine_right"]],
        [idx["gear_left"], idx["gear_right"]],
    ]
    colors = [[0,0,0]] * len(lines)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, dtype=int))
    ls.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=float))
    return ls


# -------------------------
# Visualization
# -------------------------

def visualize_keypoints(pcd, keypoints, sphere_size=0.3):
    geometries = [pcd]
    color_map = {
        "nose": [0, 0, 1],
        "tail": [0, 1, 1],
        "wing_left": [0, 1, 0],
        "wing_right": [0, 1, 0],
        "gear_left": [1, 1, 0],
        "gear_right": [1, 1, 0],
        "engine_left": [1, 0, 1],
        "engine_right": [1, 0, 1],
    }
    for name, coord in keypoints.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)
        sphere.paint_uniform_color(color_map.get(name, [0, 0, 0]))
        sphere.translate(coord)
        geometries.append(sphere)
    o3d.visualization.draw_geometries(geometries)


def visualize_cloud_with_wireframe_world(
    pcd_world: o3d.geometry.PointCloud,
    nodes_aligned: Dict[str, np.ndarray],
    eigvecs: np.ndarray,
    mean: np.ndarray,
    line_color=(0.0, 0.0, 0.0)
):
    """Transform aligned skeleton nodes back to world, build LineSet, and draw."""
    # transform nodes to world
    nodes_world = {k: (v @ eigvecs.T) + mean for k, v in nodes_aligned.items()}
    names = ["nose","tail","wing_left","wing_right","engine_left","engine_right","gear_left","gear_right"]
    idx = {n:i for i,n in enumerate(names)}
    pts = np.array([nodes_world[n] for n in names], dtype=float)

    lines = [
        [idx["tail"], idx["nose"]],
        [idx["wing_left"], idx["wing_right"]],
        [idx["engine_left"], idx["engine_right"]],
        [idx["gear_left"], idx["gear_right"]],
    ]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(np.array(lines, dtype=int))
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(line_color, dtype=float), (len(lines), 1)))

    p = o3d.geometry.PointCloud(pcd_world)
    if not p.has_colors():
        p.paint_uniform_color([0.82, 0.82, 0.82])
    o3d.visualization.draw_geometries([p, ls])
    return nodes_world, ls


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Aircraft keypoints + skeleton fit")
    ap.add_argument("--in", dest="in_path", type=str, required=False,
                    default="/home/femi/Benchmarking_framework/scripts/pre_processing/outputs/scene_000_red_bbox_points.pcd",
                    help="Input point cloud path (.pcd/.ply)")
    ap.add_argument("--save-debug", action="store_true", help="Write debug colored PCDs")
    ap.add_argument("--no-vis", action="store_true", help="Disable visualization")
    args = ap.parse_args()

    in_path = args.in_path
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    pcd = o3d.io.read_point_cloud(in_path)

    # Optional debug color exports (before anything)
    if args.save_debug:
        o3d.io.write_point_cloud("debug_rgb_xyz_before.pcd", color_xy_z(pcd, "rgb_xyz"), write_ascii=True)

    # --- Detect keypoints ---
    keypoints_world, dbg = detect_aircraft_keypoints(
        pcd,
        ground_thresh=0.05,
        eps_xy=0.35,
        min_xy=40,
        z_bin_size=0.5,
        require_level_pair=True,
        wing_z_band=(75, 98),   # retained but not critical (highest-Z level logic is used)
        engine_z_band=(40, 60),
        gear_z_pct=5,
    )

    # print detected keypoints
    print("Detected keypoints (world):")
    for k, v in keypoints_world.items():
        print(f"  {k}: {v}")

    if "error" in dbg:
        print("Detection error:", dbg["error"])
        return

    # Save a lightly colored version post-clean (optional)
    if args.save_debug:
        pcd_clean = remove_ground_and_noise(pcd, ground_thresh=0.05)
        aligned_pcd, _, _ = align_pcd_pca(pcd_clean)
        o3d.io.write_point_cloud("debug_aligned_rgb_xyz.pcd", color_xy_z(aligned_pcd, "rgb_xyz"), write_ascii=True)

    # --- Fit skeleton in aligned frame ---
    A = dbg["aligned_points"]
    eigvecs, mean = dbg["eigvecs"], dbg["mean"]
    keypoints_aligned = dbg.get("keypoints_aligned", {})

    if not keypoints_aligned:
        print("No aligned keypoints available; skipping skeleton fit.")
        return

    p_opt, nodes_aligned, rmse, res = fit_skeleton_to_keypoints(A, keypoints_aligned)
    if nodes_aligned is None:
        print("Skeleton fit could not run (no keypoints).")
        return

    # Report + simple normalization for validation
    norm = max(1e-6, max(p_opt.L_fuse, p_opt.span))
    print("\n--- Skeleton fit ---")
    print("RMSE:", rmse, f"(normalized: {rmse/norm:.3%} of max(L_fuse, span))")
    print("Params:")
    print(json.dumps({
        "L_fuse": p_opt.L_fuse,
        "span": p_opt.span,
        "x_wing_frac": p_opt.x_wing_frac,
        "z_wing": p_opt.z_wing,
        "x_engine_frac": p_opt.x_engine_frac,
        "y_engine_frac": p_opt.y_engine_frac,
        "z_engine": p_opt.z_engine,
        "x_gear_frac": p_opt.x_gear_frac,
        "y_gear_frac": p_opt.y_gear_frac,
        "z_gear": p_opt.z_gear,
    }, indent=2))

    # --- Visualize in WORLD coords ---
    if not args.no_vis:
        base = o3d.geometry.PointCloud(pcd)
        base.paint_uniform_color([0.82, 0.82, 0.82])
        nodes_world, wire = visualize_cloud_with_wireframe_world(
            base, nodes_aligned, eigvecs, mean, line_color=(0,0,0)
        )

    # Optional: export skeleton as JSON (world)
    nodes_world_json = {k: list(map(float, ((v @ eigvecs.T) + mean))) for k, v in nodes_aligned.items()}
    with open("fitted_skeleton_world.json", "w") as f:
        json.dump(nodes_world_json, f, indent=2)
    print("\nSaved fitted skeleton nodes to fitted_skeleton_world.json")
    if args.save_debug and not args.no_vis:
        try:
            # Save LineSet to disk (Open3D supports write for LineSet)
            nodes_world, ls = visualize_cloud_with_wireframe_world(
                o3d.geometry.PointCloud(pcd), nodes_aligned, eigvecs, mean
            )
            o3d.io.write_line_set("fitted_skeleton.lineset.ply", ls)
            print("Saved wireframe to fitted_skeleton.lineset.ply")
        except Exception as e:
            print("LineSet export failed:", e)


if __name__ == "__main__":
    main()
