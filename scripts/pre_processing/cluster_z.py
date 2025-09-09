#!/usr/bin/env python3
"""
Point-cloud clustering pipeline with:
  1) DBSCAN in XY
  2) DBSCAN in Z within each XY cluster
  3) 1-D DBSCAN over cluster mean-Z to form Z-groups
  4) Keep only clusters that have at least one partner in the SAME Z-group
     with similar Y position (centroid mean_y within a tolerance)
  5) Greedy XY separation among the remaining clusters

Visualization toggles are provided for each stage.
"""

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

# ----------------- CONFIG -----------------
INPUT_FILE = "/home/femi/Benchmarking_framework/scripts/yolo/yolo_outputs/scene_000_red_bbox_points.pcd"

# Stage 1–2 clustering (XY then Z within each XY cluster)
EPS_XY = 0.6
EPS_Z = 0.15
MIN_SAMPLES_XY = 50
MIN_SAMPLES_Z = 10

# 1-D DBSCAN over cluster mean-Z to define "same height"
Z_GROUP_EPS = 0.5          # Z tolerance (same units as Z)
Z_GROUP_MIN_SAMPLES = 2    # >=2 so each cluster matches at least one "other"

# Pairing by Y position (centroid mean_y), within each Z-group
Y_PAIR_EPS = 0.30          # absolute tolerance on |mean_y_i - mean_y_j| (same units as Y)

# XY separation among remaining clusters
MIN_XY_SEPARATION = 2.0    # meters

# -------- Visualization toggles (all stages) --------
SHOW_STAGE_0_RAW              = True
SHOW_STAGE_1_XY               = True
SHOW_STAGE_2_XY_Z             = True
SHOW_STAGE_3_Z_GROUPS         = True
SHOW_STAGE_4_SIZE_FILTERED    = True   # reused for Y-pair visualization
SHOW_STAGE_5_XY_SEPARATED     = True
# ----------------------------------------------------


# ------------------------------------------
# Utilities
# ------------------------------------------

def draw_points(points, colors=None, window="Open3D"):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        p.colors = o3d.utility.Vector3dVector(colors)
    try:
        o3d.visualization.draw_geometries([p], window_name=window)
    except TypeError:
        # Older Open3D without window_name
        print(f"\n[VIEW] {window}")
        o3d.visualization.draw_geometries([p])


def random_palette(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, 3))


def color_by_labels(points, labels, noise_color=(0.85, 0.85, 0.85), seed=0):
    """Return Nx3 colors where each non-negative label gets a distinct color."""
    labels = np.asarray(labels)
    colors = np.ones((points.shape[0], 3), float) * np.array(noise_color)
    if np.any(labels >= 0):
        uniq = np.array(sorted([u for u in np.unique(labels) if u >= 0]))
        palette = random_palette(len(uniq), seed=seed)
        lut = {lbl: palette[i] for i, lbl in enumerate(uniq)}
        for i, lbl in enumerate(labels):
            if lbl >= 0:
                colors[i] = lut[lbl]
    return colors


def color_by_mask(points, mask, on_color=(0.12, 0.62, 0.98), off_color=(0.85, 0.85, 0.85)):
    colors = np.ones((points.shape[0], 3), float) * np.array(off_color)
    colors[mask] = on_color
    return colors


def cluster_point_cloud_xy_then_z(points: np.ndarray,
                                  eps_xy=0.5,
                                  eps_z=0.2,
                                  min_samples_xy=10,
                                  min_samples_z=5):
    """
    Cluster a point cloud first in XY with DBSCAN, then within each XY cluster along Z (1D).
    Returns:
      - labels_xy: per-point XY cluster labels (-1 noise)
      - labels_global: per-point final cluster labels after Z split (-1 noise)
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points with shape (N, 3). Got: {points.shape}")

    xy = points[:, :2]
    z = points[:, 2:3]  # (N, 1)

    # Step 1: cluster in XY
    labels_xy = DBSCAN(eps=eps_xy, min_samples=min_samples_xy).fit_predict(xy)

    labels_global = np.full(points.shape[0], -1, dtype=int)
    next_cluster_id = 0

    # Step 2: per-XY-cluster clustering along Z
    for lbl_xy in np.unique(labels_xy):
        if lbl_xy == -1:
            continue
        idx_xy = np.where(labels_xy == lbl_xy)[0]
        if idx_xy.size == 0:
            continue

        z_sub = z[idx_xy]  # (M, 1)
        labels_z = DBSCAN(eps=eps_z, min_samples=min_samples_z).fit_predict(z_sub)

        for lbl_z in np.unique(labels_z):
            if lbl_z == -1:
                continue
            idx_z_local = np.where(labels_z == lbl_z)[0]
            if idx_z_local.size == 0:
                continue
            idx_points = idx_xy[idx_z_local]
            labels_global[idx_points] = next_cluster_id
            next_cluster_id += 1

    return labels_xy, labels_global


def compute_cluster_stats(points: np.ndarray, labels: np.ndarray):
    """
    Per-cluster stats: indices, count, mean (x,y,z), AABB sizes (size_x, size_y, size_z).
    """
    valid = labels >= 0
    cluster_ids = np.unique(labels[valid])
    stats = {}
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        pts = points[idx]
        min_xyz = pts.min(axis=0)
        max_xyz = pts.max(axis=0)
        sizes = max_xyz - min_xyz
        stats[cid] = {
            "indices": idx,
            "count": idx.size,
            "mean_x": float(np.mean(pts[:, 0])),
            "mean_y": float(np.mean(pts[:, 1])),
            "mean_z": float(np.mean(pts[:, 2])),
            "size_x": float(sizes[0]),
            "size_y": float(sizes[1]),
            "size_z": float(sizes[2]),
        }
    return cluster_ids, stats


def group_clusters_by_mean_z(stats: dict, eps: float, min_samples: int):
    """
    1-D DBSCAN on mean-Z. Only groups with >= min_samples get non-negative labels.
    Returns mapping cluster_id -> zgroup_label (>=0) or -1 if not in any valid group,
    plus the raw z-group labels aligned to sorted cluster ids.
    """
    cids = np.array(sorted(stats.keys()))
    if cids.size == 0:
        return {}, np.array([])

    heights = np.array([[stats[c]["mean_z"]] for c in cids])  # (K,1)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    raw_labels = db.fit_predict(heights)  # -1 = noise
    cluster_to_group = {cid: lbl for cid, lbl in zip(cids, raw_labels)}
    return cluster_to_group, raw_labels


# ---------------- Pairing by Y position (new) ----------------

def filter_clusters_with_y_position_pairs(stats: dict,
                                          cluster_to_group: dict,
                                          y_pair_eps: float):
    """
    Keep clusters that have at least one other cluster in the SAME Z-group
    whose centroid mean_y is within y_pair_eps.

    Returns: set of cluster_ids that have a partner by Y position.
    """
    groups = {}
    for cid, g in cluster_to_group.items():
        if g >= 0:
            groups.setdefault(g, []).append(cid)

    kept = set()
    for g, members in groups.items():
        if len(members) < 2:
            continue
        # Check all unordered pairs for |mean_y_i - mean_y_j| <= y_pair_eps
        for i in range(len(members)):
            ci = members[i]
            yi = stats[ci]["mean_y"]
            for j in range(i + 1, len(members)):
                cj = members[j]
                yj = stats[cj]["mean_y"]
                if abs(yi - yj) <= y_pair_eps:
                    kept.add(ci)
                    kept.add(cj)
    return kept


def retain_with_xy_separation(stats: dict,
                              cluster_to_group: dict,
                              candidate_ids: set,
                              min_xy_sep: float):
    """Greedy selection per Z group: keep larger clusters first, enforce centroid XY distance."""
    groups = {}
    for cid in candidate_ids:
        g = cluster_to_group[cid]
        groups.setdefault(g, []).append(cid)

    retained = set()
    for g, members in groups.items():
        members_sorted = sorted(members, key=lambda c: stats[c]["count"], reverse=True)
        selected = []
        for cid in members_sorted:
            cx, cy = stats[cid]["mean_x"], stats[cid]["mean_y"]
            ok = True
            for sid in selected:
                sx, sy = stats[sid]["mean_x"], stats[sid]["mean_y"]
                if np.hypot(cx - sx, cy - sy) < min_xy_sep:
                    ok = False
                    break
            if ok:
                selected.append(cid)
        retained.update(selected)
    return retained


# ------------------------------------------
# Main
# ------------------------------------------

def main():
    # Load
    pcd = o3d.io.read_point_cloud(INPUT_FILE)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise RuntimeError(f"No points loaded from {INPUT_FILE}")
    print(f"Loaded {points.shape[0]} points from {INPUT_FILE}")

    # ---- Stage 0: Raw ----
    if SHOW_STAGE_0_RAW:
        draw_points(points, None, window="Stage 0 — Raw Point Cloud")

    # 1) Cluster XY -> then Z
    labels_xy, labels_final = cluster_point_cloud_xy_then_z(
        points,
        eps_xy=EPS_XY,
        eps_z=EPS_Z,
        min_samples_xy=MIN_SAMPLES_XY,
        min_samples_z=MIN_SAMPLES_Z,
    )

    # ---- Stage 1: XY clusters ----
    if SHOW_STAGE_1_XY:
        colors_xy = color_by_labels(points, labels_xy, seed=1)
        draw_points(points, colors_xy, window="Stage 1 — XY DBSCAN Clusters")

    # ---- Stage 2: XY→Z clusters ----
    if SHOW_STAGE_2_XY_Z:
        colors_xyz = color_by_labels(points, labels_final, seed=2)
        draw_points(points, colors_xyz, window="Stage 2 — XY→Z Clusters")

    # Stats
    cluster_ids, stats = compute_cluster_stats(points, labels_final)
    if cluster_ids.size == 0:
        print("All points are noise.")
        return

    print("\nPer-cluster stats (mean_z):")
    for cid in sorted(cluster_ids):
        print(f"  Cluster {cid:3d} | count={stats[cid]['count']:5d} | mean_z={stats[cid]['mean_z']:.6f}")

    # 3) Z-grouping on mean-Z (keep groups with >= Z_GROUP_MIN_SAMPLES)
    cluster_to_group, zgroup_labels = group_clusters_by_mean_z(stats, Z_GROUP_EPS, Z_GROUP_MIN_SAMPLES)

    # Per-point Z-group label (map final cluster label -> group)
    zgroup_per_point = np.array([cluster_to_group.get(lbl, -1) if lbl >= 0 else -1 for lbl in labels_final])

    # ---- Stage 3: Z groups (valid groups colored, others gray) ----
    if SHOW_STAGE_3_Z_GROUPS:
        colors_zgrp = color_by_labels(points, zgroup_per_point, seed=3)
        draw_points(points, colors_zgrp, window="Stage 3 — Z-Groups (by mean-Z)")

    # Candidate clusters: in any valid Z-group
    candidates = {cid for cid, g in cluster_to_group.items() if g >= 0}
    if not candidates:
        print("\nNo clusters share similar mean-Z with any other. Stopping after Stage 3.")
        return

    # 4) Keep clusters that have a same-Z-group partner with similar Y position (mean_y)
    y_pairs_ok = filter_clusters_with_y_position_pairs(
        stats,
        cluster_to_group,
        Y_PAIR_EPS
    )
    candidates = candidates.intersection(y_pairs_ok)

    # ---- Stage 4: Y-position pair visualization ----
    if SHOW_STAGE_4_SIZE_FILTERED:
        mask_y_ok = np.isin(labels_final, list(candidates))
        colors_y = color_by_mask(points, mask_y_ok, on_color=(0.10, 0.80, 0.35))
        draw_points(points, colors_y, window=f"Stage 4 — Y-Position Pair (|Δy| ≤ {Y_PAIR_EPS})")

    if not candidates:
        print("\nAfter Y-position pair filtering, nothing remains.")
        return

    # 5) XY separation ≥ MIN_XY_SEPARATION (greedy per Z-group)
    retained = retain_with_xy_separation(stats, cluster_to_group, candidates, MIN_XY_SEPARATION)

    # ---- Stage 5: final retained visualization ----
    if SHOW_STAGE_5_XY_SEPARATED:
        final_mask = np.isin(labels_final, list(retained))
        # Color final kept by Z-group, others gray
        kept_groups = sorted({cluster_to_group[cid] for cid in retained})
        palette = random_palette(len(kept_groups), seed=5) if kept_groups else np.empty((0, 3))
        g2c = {g: palette[i] for i, g in enumerate(kept_groups)}
        colors_final = np.ones((points.shape[0], 3), float) * 0.85
        for i, lbl in enumerate(labels_final):
            if lbl in retained:
                g = cluster_to_group.get(lbl, -1)
                colors_final[i] = g2c.get(g, np.array([0.2, 0.8, 0.2]))
        draw_points(
            points,
            colors_final,
            window="Stage 5 — Final Retained (by Z-Group, ≥{:.1f} m apart)".format(MIN_XY_SEPARATION)
        )

    # Print summary
    print(f"\nZ-grouping: eps={Z_GROUP_EPS}, min_samples={Z_GROUP_MIN_SAMPLES}")
    valid_groups = sorted({g for g in zgroup_labels if g >= 0})
    for g in valid_groups:
        members = [cid for cid, gg in cluster_to_group.items() if gg == g]
        kept = sorted([cid for cid in members if cid in retained])
        if members:
            min_sizes = np.min([[stats[c]['size_x'], stats[c]['size_y'], stats[c]['size_z']] for c in members], axis=0)
            wz = np.average([stats[c]['mean_z'] for c in members], weights=[stats[c]['count'] for c in members])
            print(f"  Z-Group {g}: weighted_mean_z={wz:.6f} | min_size_xyz={min_sizes} | kept={kept}")
        else:
            print(f"  Z-Group {g}: (no members)")

if __name__ == "__main__":
    main()
