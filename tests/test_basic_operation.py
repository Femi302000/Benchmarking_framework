#!/usr/bin/env python3
"""
pcd_to_range_image_side_by_side.py

Loads a PCD, projects it onto a 128×1024 spherical range image,
and displays it alongside the original point cloud in one Matplotlib window.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- User parameters ---
PCD_PATH = "/scripts/data_dir/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_model.pcd"
H, W = 128, 1024   # 128 rows (vertical), 1024 cols (horizontal)

def pcd_to_range_image(pcd, H, W):
    """
    Projects pcd (Open3D PointCloud) into a range image of shape (H,W).
    Assumes ±90° vertical FOV and 360° horizontal FOV.
    Rolls so forward-facing direction is centered.
    """
    pts = np.asarray(pcd.points)
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    r = np.linalg.norm(pts, axis=1)

    # spherical coords
    az = np.arctan2(y, x)              # [-π, +π]
    el = np.arcsin(z / (r + 1e-6))     # [-π/2, +π/2]

    # map to pixel coords
    col = ((az + np.pi) / (2 * np.pi) * W).astype(int)
    row = ((el + np.pi/2) / np.pi * H).astype(int)

    col = np.clip(col, 0, W-1)
    row = np.clip(row, 0, H-1)

    # initialize and fill with nearest range
    range_img = np.full((H, W), np.inf, dtype=np.float32)
    for ri, rr, cc in zip(r, row, col):
        if ri < range_img[rr, cc]:
            range_img[rr, cc] = ri

    range_img[range_img == np.inf] = np.nan
    # center the forward direction
    range_img = np.roll(range_img, W // 2, axis=1)
    return range_img

def main():
    # 1) Load PCD
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    pts = np.asarray(pcd.points)
    print(f"Loaded {len(pts)} points")

    # 2) Build range image
    img = pcd_to_range_image(pcd, H, W)
    print(f"Range image: {img.shape}")

    # 3) Set up side-by-side figure
    fig = plt.figure(figsize=(16, 6))

    # 3a) Left: 3D point cloud
    ax_pc = fig.add_subplot(1, 2, 1, projection='3d')
    ax_pc.scatter(
        pts[:,0], pts[:,1], pts[:,2],
        c=pts[:,2], cmap='jet', s=0.5, linewidth=0
    )
    ax_pc.set_title("Original Point Cloud")
    ax_pc.set_xlabel("X"); ax_pc.set_ylabel("Y"); ax_pc.set_zlabel("Z")
    ax_pc.view_init(elev=20, azim=120)
    ax_pc.set_box_aspect((1,1,0.5))

    # 3b) Right: range image
    ax_img = fig.add_subplot(1, 2, 2)
    clean = np.nan_to_num(img, nan=0.0)
    vmin, vmax = np.nanpercentile(clean, (1, 99))
    im = ax_img.imshow(
        clean, cmap="viridis", origin="lower",
        vmin=vmin, vmax=vmax, aspect="auto"
    )
    ax_img.set_title(f"Range Image ({H}×{W})")
    ax_img.set_xlabel("Azimuth bin")
    ax_img.set_ylabel("Elevation bin")
    fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
