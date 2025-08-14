#!/usr/bin/env python3
"""
visualize_all_views.py

For each scene_NNN in the HDF5 file:
  • SHOW_PCD: show point‐cloud panels
  • SHOW_IMAGE: show sensor images (with Open3D)
  • SHOW_GROUND: include ground‐removed views
  • SHOW_LABEL: include label‐colored views
  • TOGETHER_CLOUDS: combine 4 clouds in one window
  • TOGETHER_IMAGES: combine 4 images sequentially in Open3D
"""
import os
import sys

import h5py
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt  # only for box aspect helper

        # --- lazily compute z_min from TF (z_ref_target <- z_ref_source) ---
# Hardcoded HDF5 path
H5_FILE_PATH = (
    "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/"
    "HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"
)

# Flags – set as needed
SHOW_PCD        = True
SHOW_IMAGE      = True
SHOW_GROUND     = True
SHOW_LABEL      = True
TOGETHER_CLOUDS = False
TOGETHER_IMAGES = False


def plot_cloud(ax, xyz, colors, title):
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=colors, s=1)
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=120)
    ax.set_box_aspect((1,1,0.5))


def normalize_to_uint8(img: np.ndarray) -> np.uint8:
    """Scale a float image to [0,255] and convert to uint8."""
    mi, ma = np.nanmin(img), np.nanmax(img)
    if ma > mi:
        img2 = (img - mi) / (ma - mi)
    else:
        img2 = np.zeros_like(img)
    return (img2 * 255).astype(np.uint8)


def show_open3d_image(img: np.ndarray, title: str):
    """
    Wrap a 2D numpy array as an Open3D Image and show it.
    img: 2D float32 array
    """
    u8 = normalize_to_uint8(img)
    o3d_img = o3d.geometry.Image(u8)
    # You can set a window name via the visualizer class, but for simplicity:
    o3d.visualization.draw_geometries([o3d_img], window_name=title)


def visualize_all_views(h5_path):
    if not os.path.isfile(h5_path):
        print(f"Error: file not found: {h5_path}", file=sys.stderr)
        return

    with h5py.File(h5_path, "r") as f:
        for scene, grp in f.items():
            print(f"\n=== Scene {scene} ===")

            # 1) Point-cloud data
            pts = grp['points'][()]
            cols = [c.decode() for c in grp['points'].attrs['columns']]
            xyz = pts[:, :3]
            is_ground = pts[:, cols.index('is_ground')].astype(bool)
            labels = pts[:, cols.index('is_aircraft')].astype(int)

            raw_xyz = xyz
            ng_xyz = xyz[~is_ground] if SHOW_GROUND else None

            # Colors
            gray = np.tile([0.5,0.5,0.5], (len(raw_xyz),1))
            raw_colors = gray
            ng_colors = gray[:len(ng_xyz)] if SHOW_GROUND else None

            label_colors = gray.copy()
            if SHOW_LABEL:
                label_colors[labels==1] = [1,0,0]
            ng_label_colors = None
            if SHOW_GROUND and SHOW_LABEL:
                ng_label_colors = gray[:len(ng_xyz)].copy()
                ng_label_colors[labels[~is_ground]==1] = [1,0,0]

            # 1a) Show point clouds
            if SHOW_PCD:
                if TOGETHER_CLOUDS:
                    import matplotlib.pyplot as plt  # to get 3D axes
                    fig = plt.figure(figsize=(16,12))
                    axs = [fig.add_subplot(2,2,i, projection='3d') for i in [1,2,3,4]]
                    plot_cloud(axs[0], raw_xyz, raw_colors,      "Raw")
                    plot_cloud(axs[1], ng_xyz,  ng_colors,       "Ground Removed")
                    plot_cloud(axs[2], raw_xyz, label_colors,    "Raw Labeled")
                    plot_cloud(axs[3], ng_xyz,  ng_label_colors, "Ground Removed Labeled")
                    fig.suptitle(f"Point Clouds {scene}")
                    plt.tight_layout(); plt.show()
                else:
                    # raw
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(raw_xyz)
                    pcd.colors = o3d.utility.Vector3dVector(raw_colors)
                    o3d.visualization.draw_geometries([pcd], window_name=f"{scene} Raw")

                    # ground removed
                    if SHOW_GROUND:
                        pcd2 = o3d.geometry.PointCloud()
                        pcd2.points = o3d.utility.Vector3dVector(ng_xyz)
                        pcd2.colors = o3d.utility.Vector3dVector(ng_colors)
                        o3d.visualization.draw_geometries([pcd2], window_name=f"{scene} No Ground")

                    # raw labeled
                    if SHOW_LABEL:
                        pcd3 = o3d.geometry.PointCloud()
                        pcd3.points = o3d.utility.Vector3dVector(raw_xyz)
                        pcd3.colors = o3d.utility.Vector3dVector(label_colors)
                        o3d.visualization.draw_geometries([pcd3], window_name=f"{scene} Raw Labeled")

                    # ground removed labeled
                    if SHOW_GROUND and SHOW_LABEL:
                        pcd4 = o3d.geometry.PointCloud()
                        pcd4.points = o3d.utility.Vector3dVector(ng_xyz)
                        pcd4.colors = o3d.utility.Vector3dVector(ng_label_colors)
                        o3d.visualization.draw_geometries(
                            [pcd4],
                            window_name=f"{scene} No Ground Labeled"
                        )

            # 2) Sensor images via Open3D Image
            # 2) Sensor images from 'points' and reshape using H x W
            if SHOW_IMAGE:
                try:
                    H = int(f.attrs["height"])
                    W = int(f.attrs["width"])
                except KeyError:
                    print("Missing height/width attributes in HDF5 file. Skipping image view.")
                    return

                pts = grp['points'][()]
                cols = [c.decode() for c in grp['points'].attrs['columns']]

                def get_image_from_column(name):
                    idx = cols.index(name)
                    arr = pts[:, idx]
                    return arr.reshape(H, W)

                img_data = {
                    "Range": get_image_from_column("range"),
                    "Intensity": get_image_from_column("intensity"),
                    "Reflectivity": get_image_from_column("reflectivity"),
                    "Ambient": get_image_from_column("ambient"),
                }

                if TOGETHER_IMAGES:
                    for title, im in img_data.items():
                        show_open3d_image(im, f"{scene} {title}")
                else:
                    for title, im in img_data.items():
                        show_open3d_image(im, f"{scene} {title}")


if __name__ == "__main__":
    visualize_all_views(H5_FILE_PATH)
