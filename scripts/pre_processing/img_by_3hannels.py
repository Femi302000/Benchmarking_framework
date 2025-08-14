#!/usr/bin/env python3
"""
overlay_aircraft_mask_updated.py (with BEV & side-view projections)

• Saves preview images (range, mask, overlays, intensity) — all as 3-channel RGB
• Saves 16-bit depth and colorized intensity as 3-channel RGB
• Saves ambient, reflectivity, z as 3-channel RGB
• Saves two synthetic RGB images:
    <scene>_synthetic_rgb_range.png
    <scene>_synthetic_rgb_reflectivity.png
• Saves synthetic RGB combining ALL available channels via PCA:
    <scene>_synthetic_rgb_all.png
• Saves BEV RGB projection (top-down)
• Saves side-view RGB projections (profile views along X and Y)

Author: 2025-08-05 (updated to force 3-channel everywhere)
"""
import os
import sys
import numpy as np
import h5py
import matplotlib.cm as cm
import imageio.v2 as imageio

H5_PATH = (
    "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/"
    "HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"
)
SAVE_DIR = "./3_channel"
os.makedirs(SAVE_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Generic helpers (all output as 3-channel RGB via imageio)
# ----------------------------------------------------------------------
def _autoscale(img, nan_fill=0.0):
    clean = np.nan_to_num(img, nan=nan_fill)
    vmin, vmax = np.percentile(clean, (1, 99))
    return clean, vmin, vmax


def _save_image(img, path, *, cmap="gray", vmin=None, vmax=None):
    """
    Normalize `img`, map through `cmap`, drop alpha, write RGB uint8.
    """
    # normalize
    clean = np.nan_to_num(img, nan=vmin if vmin is not None else 0.0)
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(clean, (1, 99))
    norm = np.clip((clean - vmin) / (vmax - vmin + 1e-12), 0, 1)

    # colormap → RGBA float, drop A → RGB uint8
    rgba = cm.get_cmap(cmap)(norm)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)

    imageio.imwrite(path, rgb)


def _save_overlay(base_img, mask, path,
                  *, cmap="viridis", mask_color=(1, 0, 0), alpha=0.35):
    """
    Composite `mask` (binary) over `base_img`. Both mapped to RGB, then alpha-blended.
    """
    # base image → RGB
    clean = np.nan_to_num(base_img, nan=0.0)
    vmin, vmax = np.percentile(clean, (1, 99))
    norm = np.clip((clean - vmin) / (vmax - vmin + 1e-12), 0, 1)
    base_rgb = (cm.get_cmap(cmap)(norm)[..., :3] * 255).astype(np.uint8)

    # mask → single channel 0/1 → RGB mask bitmap
    mask_bin = (mask > 0).astype(np.uint8)
    mask_rgb = np.zeros_like(base_rgb)
    for c in range(3):
        mask_rgb[..., c] = (mask_bin * mask_color[c] * 255).astype(np.uint8)

    # alpha blend
    comp = ((1 - alpha) * base_rgb + alpha * mask_rgb).astype(np.uint8)
    imageio.imwrite(path, comp)


# ----------------------------------------------------------------------
# Depth & RGB + optional exports
# ----------------------------------------------------------------------
def save_depth_and_rgb(range_img, intensity_img, scene, *,
                       depth_dir=SAVE_DIR, cm_name="viridis"):
    depth_path = os.path.join(depth_dir, f"{scene}_depth16.png")
    rgb_path   = os.path.join(depth_dir, f"{scene}_intensity_rgb.png")

    # DEPTH (16-bit PNG)
    max_range = np.nanmax(range_img) or 1.0
    depth16 = np.clip(range_img / max_range, 0, 1)
    depth16 = (depth16 * 65535).astype(np.uint16)
    imageio.imwrite(depth_path, depth16)

    # INTENSITY RGB (force 3 channels)
    clean, vmin, vmax = _autoscale(intensity_img, nan_fill=0)
    norm = np.clip((clean - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgba = cm.get_cmap(cm_name)(norm)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    imageio.imwrite(rgb_path, rgb)

    print(f"[✓] {scene} → depth16, intensity_rgb saved.")


def save_optional_images(pts, cols, scene, h, w):
    channels = {}
    optional_fields = {
        "ambient":      "inferno",
        "reflectivity": "plasma",
        "z":            None
    }
    for key, cmap in optional_fields.items():
        if key in cols:
            img = pts[:, cols.index(key)].reshape(h, w)
            out_path = os.path.join(SAVE_DIR, f"{scene}_{key}.png")
            if key == "z":
                _save_image(img, out_path, cmap="gray")
            else:
                _save_image(img, out_path, cmap=cmap)
            channels[key] = img
        else:
            channels[key] = None
    return channels


def save_multiple_synthetic_rgbs(intensity_img, reflectivity_img,
                                 ambient_img, range_img,
                                 scene, output_dir=SAVE_DIR):
    def norm_to_uint8(img):
        clean, vmin, vmax = _autoscale(img)
        norm = np.clip((clean - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return (norm * 255).astype(np.uint8)

    if ambient_img is None or intensity_img is None:
        print(f"[!] {scene}: Missing ambient/intensity — skipping RGB synthesis")
        return

    # range-ambient-intensity
    if range_img is not None:
        rgb1 = np.stack([
            norm_to_uint8(range_img),
            norm_to_uint8(ambient_img),
            norm_to_uint8(intensity_img)
        ], axis=-1)
        imageio.imwrite(os.path.join(output_dir, f"{scene}_synthetic_rgb_range.png"), rgb1)
        print(f"[✓] {scene} → synthetic RGB (range) saved.")

    # reflectivity-ambient-intensity
    if reflectivity_img is not None:
        rgb2 = np.stack([
            norm_to_uint8(reflectivity_img),
            norm_to_uint8(ambient_img),
            norm_to_uint8(intensity_img)
        ], axis=-1)
        imageio.imwrite(os.path.join(output_dir, f"{scene}_synthetic_rgb_reflectivity.png"), rgb2)
        print(f"[✓] {scene} → synthetic RGB (reflectivity) saved.")


# ----------------------------------------------------------------------
# PCA-based synthetic RGB (all channels)
# ----------------------------------------------------------------------
def save_synthetic_rgb_all(pts, cols,
                           range_img, intensity_img,
                           channels, scene,
                           h, w,
                           output_dir=SAVE_DIR):
    data_list = []
    for arr in [
        range_img,
        channels.get('reflectivity'),
        channels.get('ambient'),
        intensity_img,
        channels.get('z')
    ]:
        if arr is not None:
            clean, vmin, vmax = _autoscale(arr)
            norm = np.clip((clean - vmin) / (vmax - vmin + 1e-12), 0, 1)
            data_list.append(norm.reshape(-1))

    if len(data_list) < 3:
        print(f"[!] {scene}: Not enough channels for PCA RGB — need ≥3")
        return

    data = np.stack(data_list, axis=1)
    mean = data.mean(axis=0)
    centered = data - mean
    cov = centered.T @ centered / (centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    vecs = eigvecs[:, np.argsort(eigvals)[::-1][:3]]
    pcs = centered @ vecs

    rgb_flat = np.zeros_like(pcs, dtype=np.uint8)
    for i in range(3):
        comp = pcs[:, i]
        mn, mx = np.percentile(comp, (1, 99))
        normc = np.clip((comp - mn) / (mx - mn + 1e-12), 0, 1)
        rgb_flat[:, i] = (normc * 255).astype(np.uint8)

    rgb = rgb_flat.reshape(h, w, 3)
    imageio.imwrite(os.path.join(output_dir, f"{scene}_synthetic_rgb_all.png"), rgb)
    print(f"[✓] {scene} → synthetic RGB (all channels) saved.")


# ----------------------------------------------------------------------
# BEV (top-down) and Side-view generation (always using 3-channel arrays)
# ----------------------------------------------------------------------
def generate_bev_rgb(pts, cols, synthetic_rgb, scene,
                     bev_size=(512, 512), output_dir=SAVE_DIR):
    # drop alpha if somehow present
    if synthetic_rgb.ndim == 3 and synthetic_rgb.shape[2] == 4:
        synthetic_rgb = synthetic_rgb[..., :3]

    if not {"x", "y"}.issubset(cols):
        print(f"[!] {scene}: Missing x/y → skipping BEV.")
        return

    x = pts[:, cols.index("x")]
    y = pts[:, cols.index("y")]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    H, W = bev_size
    bev = np.zeros((H, W, 3), dtype=np.uint8)
    flat = synthetic_rgb.reshape(-1, 3)

    for i in range(pts.shape[0]):
        u = int((x[i] - x_min) / (x_max - x_min + 1e-8) * (W - 1))
        v = int((y[i] - y_min) / (y_max - y_min + 1e-8) * (H - 1))
        bev[H - 1 - v, u] = flat[i]

    imageio.imwrite(os.path.join(output_dir, f"{scene}_bev_rgb.png"), bev)
    print(f"[✓] {scene} → BEV RGB saved.")


def generate_sideview_rgb(pts, cols, synthetic_rgb, scene,
                          axis='y', img_size=(512, 512), output_dir=SAVE_DIR):
    # drop alpha if present
    if synthetic_rgb.ndim == 3 and synthetic_rgb.shape[2] == 4:
        synthetic_rgb = synthetic_rgb[..., :3]

    if axis not in cols or 'z' not in cols:
        print(f"[!] {scene}: Missing {axis}/z → skipping side view.")
        return

    X = pts[:, cols.index(axis)]
    Z = pts[:, cols.index('z')]
    H, W = img_size
    view = np.zeros((H, W, 3), dtype=np.uint8)
    flat = synthetic_rgb.reshape(-1, 3)

    x_min, x_max = X.min(), X.max()
    z_min, z_max = Z.min(), Z.max()

    for i in range(pts.shape[0]):
        u = int((X[i] - x_min) / (x_max - x_min + 1e-8) * (W - 1))
        v = int((Z[i] - z_min) / (z_max - z_min + 1e-8) * (H - 1))
        view[H - 1 - v, u] = flat[i]

    imageio.imwrite(os.path.join(output_dir, f"{scene}_sideview_{axis}.png"), view)
    print(f"[✓] {scene} → side-view ({axis}) RGB saved.")


# ----------------------------------------------------------------------
# Per-scene processing
# ----------------------------------------------------------------------
def process_scene(range_img, intensity_img, mask, scene):
    _save_image(range_img, os.path.join(SAVE_DIR, f"{scene}_range.png"), cmap="viridis")
    _save_overlay(range_img, mask, os.path.join(SAVE_DIR, f"{scene}_range_overlay.png"))
    _save_image(intensity_img, os.path.join(SAVE_DIR, f"{scene}_intensity.png"), cmap="viridis")
    _save_overlay(intensity_img, mask, os.path.join(SAVE_DIR, f"{scene}_intensity_overlay.png"))
    _save_image(mask, os.path.join(SAVE_DIR, f"{scene}_mask.png"), cmap="gray")


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main():
    if not os.path.isfile(H5_PATH):
        sys.exit(f"File not found: {H5_PATH}")

    with h5py.File(H5_PATH, "r") as f:
        h, w = int(f.attrs["height"]), int(f.attrs["width"])

        for scene, grp in f.items():
            pts  = grp["points"][()]
            cols = [c.decode() for c in grp["points"].attrs["columns"]]
            print(f"Scene: {scene}, columns: {cols}")

            required = {"range", "intensity", "is_aircraft"}
            if missing := required - set(cols):
                print(f"[!] {scene}: missing {missing}, skipping")
                continue
            if pts.shape[0] != h * w:
                print(f"[!] {scene}: size mismatch, skipping")
                continue

            range_img     = pts[:, cols.index("range")].reshape(h, w)
            intensity_img = pts[:, cols.index("intensity")].reshape(h, w)
            mask          = (pts[:, cols.index("is_aircraft")] == 1).reshape(h, w).astype(np.uint8)

            process_scene(range_img, intensity_img, mask, scene)
            save_depth_and_rgb(range_img, intensity_img, scene)
            channels = save_optional_images(pts, cols, scene, h, w)

            save_multiple_synthetic_rgbs(
                intensity_img,
                channels.get("reflectivity"),
                channels.get("ambient"),
                range_img,
                scene
            )

            save_synthetic_rgb_all(
                pts, cols,
                range_img, intensity_img,
                channels, scene,
                h, w
            )

            for suffix in ["range", "reflectivity", "all"]:
                rgb_file = os.path.join(SAVE_DIR, f"{scene}_synthetic_rgb_{suffix}.png")
                if os.path.exists(rgb_file):
                    raw = imageio.imread(rgb_file)
                    rgb = raw[..., :3] if (raw.ndim == 3 and raw.shape[2] == 4) else raw
                    generate_bev_rgb(pts, cols, rgb, f"{scene}_{suffix}")
                    generate_sideview_rgb(pts, cols, rgb, f"{scene}_{suffix}", axis='y')
                    generate_sideview_rgb(pts, cols, rgb, f"{scene}_{suffix}", axis='x')
                else:
                    print(f"[!] {scene}_{suffix}: synthetic RGB missing.")

if __name__ == "__main__":
    main()
