#!/usr/bin/env python3
"""
visualize_ground_and_sideview.py

Load ground and labeled points from an HDF5 dataset and display both:
  1) 3D Open3D window of ground points colored by label
  2) 2D side-view (X vs Z) Matplotlib scatter

Path to file is hardcoded below in H5_FILE_PATH.

Dependencies:
  pip install numpy h5py open3d matplotlib
"""
import os
import sys

import h5py
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# --- User Configuration ---
# Hardcoded path to your HDF5 file:
H5_FILE_PATH = "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"


def show_side_view(xyz: np.ndarray, colors: np.ndarray, title: str = ""):
    """
    Plot the side-view (X vs Z) of points in 2D.
    xyz: (N×3) array of [X, Y, Z] points
    colors: (N×3) array of RGB floats in [0,1]
    """
    x = xyz[:, 0]
    z = xyz[:, 2]
    print(f"Number of points in side view: {xyz.shape[0]}")

    plt.figure(figsize=(8, 6))
    plt.scatter(x, z, s=1, c=colors, marker='.')
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title(f"Side View {title}")
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def visualize_ground_points_with_labels(
    h5_path: str,
    pts_name: str    = "points_ground",
    ground_label: str = "is_ground",
    label_field: str  = "labels",
):
    """
    For each timestamp group in the HDF5 file:
      • load the NxM points array with a 'column_names' attribute
      • filter to points where `ground_label` == 1
      • color those where `label_field` == 1 in red (others gray)
      • display a side-view and a 3D Open3D view
    """
    if not os.path.isfile(h5_path):
        print(f"Error: file not found: {h5_path}", file=sys.stderr)
        return

    with h5py.File(h5_path, "r") as f:
        for grp_name, grp in f.items():
            if pts_name not in grp:
                print(f"[{grp_name}] missing '{pts_name}', skipping")
                continue

            ds = grp[pts_name]
            data = ds[()]  # shape (N, M)
            if data.ndim != 2 or data.shape[1] < 4:
                print(f"[{grp_name}] need at least 4 columns, got {data.shape}, skipping")
                continue

            col_attr = ds.attrs.get("column_names", None)
            if col_attr is not None:
                col_names = [c.decode() for c in col_attr]
            else:
                col_names = [f"col{i}" for i in range(data.shape[1])]

            if ground_label not in col_names:
                print(f"[{grp_name}] no '{ground_label}' column, skipping ground filter")
                continue
            ground_idx = col_names.index(ground_label)
            is_ground = data[:, ground_idx].astype(int)

            mask_ground = (is_ground == 0)
            if not np.any(mask_ground):
                print(f"[{grp_name}] no ground points, skipping display")
                continue

            xyz = data[mask_ground, :3].astype(np.float64)
            colors = np.tile(np.array([0.5, 0.5, 0.5]), (xyz.shape[0], 1))

            if label_field in col_names:
                label_idx = col_names.index(label_field)
                labels = data[mask_ground, label_idx].astype(int)
                colors[labels == 1] = np.array([1.0, 0.0, 0.0])
            else:
                print(f"[{grp_name}] no '{label_field}' column, rendering all ground points gray")

            print(f"Showing side-view for group {grp_name}")
            show_side_view(xyz, colors, title=grp_name)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            count = np.count_nonzero(mask_ground)
            print(f"Displaying {count} ground points for group {grp_name} (red = '{label_field}' == 1)")
            o3d.visualization.draw_geometries(
                [pcd], window_name=str(grp_name), width=1024, height=768
            )


if __name__ == "__main__":
    visualize_ground_points_with_labels(H5_FILE_PATH)
