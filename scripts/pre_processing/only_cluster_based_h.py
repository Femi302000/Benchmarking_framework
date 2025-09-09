#!/usr/bin/env python3
"""
Filter PCD by height and cluster it: XY first, then Z.
"""

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import random, os

# ---------------- CONFIG ----------------
INPUT_FILE  = "/home/femi/Benchmarking_framework/Data/sequence_from_scene/HAM_Airport_2024_08_08_movement_a320_ceo_Germany/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_scene0001_9.595s_filtered.pcd"                 # input PCD path
OUTPUT_DIR  = "out_clusters"              # where to save results

Z_MIN, Z_MAX = -1.5, -1                  # keep only this height range

EPS_XY, MIN_SAMPLES_XY = 5, 10         # DBSCAN for XY
EPS_Z,  MIN_SAMPLES_Z  = 0.10, 5          # DBSCAN for Z
WRITE_SPLITS = True                       # save each cluster separately?
# ----------------------------------------


def dbscan_xy(points_xy):
    if len(points_xy) == 0: return np.array([], dtype=int)
    return DBSCAN(eps=EPS_XY, min_samples=MIN_SAMPLES_XY, n_jobs=-1).fit_predict(points_xy)

def dbscan_z(z_vals):
    if len(z_vals) == 0: return np.array([], dtype=int)
    return DBSCAN(eps=EPS_Z, min_samples=MIN_SAMPLES_Z, n_jobs=-1).fit_predict(z_vals.reshape(-1,1))

def colorize(points, labels):
    colors = np.zeros((len(points),3))
    rng, palette = random.Random(0), {}
    for i, lab in enumerate(labels):
        if lab == -1: colors[i] = [0.6,0.6,0.6]
        else:
            if lab not in palette:
                palette[lab] = [rng.random(), rng.random(), rng.random()]
            colors[i] = palette[lab]
    return colors

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load
    pcd = o3d.io.read_point_cloud(INPUT_FILE)
    pts = np.asarray(pcd.points)
    if pts.size == 0: 
        print("Empty cloud"); return

    # 2) Filter by Z
    mask = (pts[:,2] >= Z_MIN) & (pts[:,2] <= Z_MAX)
    pts_f = pts[mask]
    print(f"Kept {len(pts_f)}/{len(pts)} points after height filter")

    # 3) XY clustering
    labels_xy = dbscan_xy(pts_f[:,:2])

    # 4) Z clustering within XY clusters
    final_labels = -np.ones(len(pts_f), dtype=int)
    next_label = 0
    for lab in np.unique(labels_xy):
        if lab == -1: continue
        idx = np.where(labels_xy==lab)[0]
        sub_z = dbscan_z(pts_f[idx,2])
        for sublab in np.unique(sub_z):
            if sublab == -1: continue
            idx_sub = idx[sub_z==sublab]
            final_labels[idx_sub] = next_label
            next_label += 1

    # 5) Save colorized result
    colored = o3d.geometry.PointCloud()
    colored.points = o3d.utility.Vector3dVector(pts_f)
    colored.colors = o3d.utility.Vector3dVector(colorize(pts_f, final_labels))
    #o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "clusters_colorized.pcd"), colored)
    # 7) Visualize interactively
    o3d.visualization.draw_geometries([colored])


    # 6) Save each split
    if WRITE_SPLITS:
        for lab in range(next_label):
            idx = np.where(final_labels==lab)[0]
            subpc = o3d.geometry.PointCloud()
            subpc.points = o3d.utility.Vector3dVector(pts_f[idx])
            o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, f"cluster_{lab:03d}.pcd"), subpc)

    print("Done.")

if __name__=="__main__":
    main()
