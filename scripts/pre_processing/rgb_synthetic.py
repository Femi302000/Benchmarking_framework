import os
import sys
import itertools
import numpy as np
import h5py
import imageio.v2 as imageio

# ----------------------------------------------------------------------
# Synthetic RGB generation helpers
# ----------------------------------------------------------------------
def _autoscale(img, nan_fill=0.0):
    """
    Replace NaNs, then compute robust vmin/vmax as 1st and 99th percentiles.
    """
    clean = np.nan_to_num(img, nan=nan_fill)
    vmin, vmax = np.percentile(clean, (1, 99))
    return clean, vmin, vmax


def _norm_uint8(img):
    """
    Normalize to [0,255] uint8 using autoscale.
    """
    clean, vmin, vmax = _autoscale(img)
    norm = np.clip((clean - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return (norm * 255).astype(np.uint8)


def save_all_rgb_combinations(channels, scene, output_dir):
    """
    Generate synthetic RGB images for every combination of three distinct channels.
    Channels: dict of channel_name -> 2D numpy array.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Only include channels that exist
    avail = {k: v for k, v in channels.items() if v is not None}
    if len(avail) < 3:
        print(f"[!] {scene}: Need ≥3 available channels for combinations, got {len(avail)} — skipping")
        return

    # iterate all combinations of 3 channels
    for combo in itertools.combinations(avail.keys(), 3):
        # Build RGB: order channels alphabetically for consistency or custom order
        r_name, g_name, b_name = combo
        r_img = _norm_uint8(avail[r_name])
        g_img = _norm_uint8(avail[g_name])
        b_img = _norm_uint8(avail[b_name])

        rgb = np.stack([r_img, g_img, b_img], axis=-1)
        fname = f"{scene}_synrgb_{r_name}_{g_name}_{b_name}.png"
        path = os.path.join(output_dir, fname)
        imageio.imwrite(path, rgb)
        print(f"[✓] {scene}: saved synthetic RGB ({r_name},{g_name},{b_name})")


# ----------------------------------------------------------------------
# HDF5 loader and main
# ----------------------------------------------------------------------
H5_PATH = (
    "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/"
    "HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"
)
SAVE_DIR = "./3_channel"

if __name__ == "__main__":
    if not os.path.isfile(H5_PATH):
        sys.exit(f"File not found: {H5_PATH}")

    with h5py.File(H5_PATH, 'r') as f:
        h, w = int(f.attrs['height']), int(f.attrs['width'])

        for scene, grp in f.items():
            pts = grp['points'][()]
            cols = [c.decode() for c in grp['points'].attrs['columns']]
            print(f"Scene: {scene}, columns: {cols}")

            # required channels for mask or others not needed here
            # extract available channels
            idx = lambda name: cols.index(name)
            channels = {}
            for name in ['range', 'intensity', 'reflectivity', 'ambient', 'z']:
                if name in cols:
                    channels[name] = pts[:, idx(name)].reshape(h, w)
                else:
                    channels[name] = None

            # generate all synthetic RGB combinations
            save_all_rgb_combinations(channels, scene, SAVE_DIR)
