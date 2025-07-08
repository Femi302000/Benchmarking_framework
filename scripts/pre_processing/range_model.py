#!/usr/bin/env python3
"""
overlay_label1_on_range_image.py

Visualizes label==1 points over a LiDAR range image using imshow instead of scatter.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt


def overlay_labels_on_range_image(range_img: np.ndarray, label_img: np.ndarray):
    H, W = range_img.shape

    # Debug label statistics
    print(f"[INFO] Range shape: {range_img.shape}, Label shape: {label_img.shape}")
    print(f"[INFO] Num label==1: {np.sum(label_img == 1)}")
    unique_rows, counts = np.unique(np.where(label_img == 1)[0], return_counts=True)
    print("Label==1 appears on rows:")
    for r, c in zip(unique_rows, counts):
        print(f"  row {r}: {c} points")

    # Clean range image
    clean_range = np.nan_to_num(range_img, nan=0.0)
    vmin, vmax = np.percentile(clean_range, (1, 99))

    # Binary mask display (just label==1 as white on black)
    plt.figure(figsize=(10, 4))
    binary_mask = (label_img == 1).astype(np.uint8)
    plt.imshow(binary_mask, cmap='gray', origin='lower', aspect='auto')
    plt.title("Binary Mask: label == 1 (white)")
    plt.xlabel("Azimuth bin")
    plt.ylabel("Laser ring")
    plt.tight_layout()
    plt.show()

    # Overlay visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Left: raw range image
    im1 = ax1.imshow(clean_range, cmap="viridis", origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax1.set_title("Range Image (viridis)")
    ax1.set_xlabel("Azimuth bin")
    ax1.set_ylabel("Ring index")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Right: overlay with white mask
    im2 = ax2.imshow(clean_range, cmap="plasma", origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

    # Mask only where label == 1
    mask = (label_img == 1).astype(np.float32)
    masked_overlay = np.ma.masked_where(mask == 0, mask)
    ax2.imshow(masked_overlay, cmap="gray", origin="lower", alpha=1.0, aspect="auto")  # White on plasma
    ax2.set_title("Label==1 Overlay on Plasma")
    ax2.set_xlabel("Azimuth bin")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def main():
    # Path to your HDF5 dataset
    h5_path = "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"
    stamp_ns = "1723111412334707011"

    with h5py.File(h5_path, "r") as f:
        grp = f[stamp_ns]
        range_img = grp["range_image"][()]  # shape (H, W)
        pts = grp["points_ground"][()]      # shape (H*W, M)
        raw_names = grp["points_ground"].attrs["column_names"]
        names = [n.decode("ascii", "ignore").strip("\x00") for n in raw_names]

    H, W = range_img.shape
    N, M = pts.shape

    print(f"[INFO] range_image shape: {H} x {W}")
    print(f"[INFO] points_ground shape: {N} x {M}")
    print(f"[INFO] H * W = {H * W}")

    if N != H * W:
        raise RuntimeError("Mismatch: cannot reshape flat labels to (H, W)")

    name_to_idx = {n: i for i, n in enumerate(names)}
    if 'labels' not in name_to_idx:
        raise RuntimeError("Missing 'labels' field in points_ground")

    label_idx = name_to_idx['labels']
    label_img = pts[:, label_idx].reshape(H, W)

    overlay_labels_on_range_image(range_img, label_img)


if __name__ == "__main__":
    main()
