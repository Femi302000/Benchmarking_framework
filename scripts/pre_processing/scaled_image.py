import os
import sys
import itertools
import numpy as np
import h5py
import imageio.v2 as imageio

def scale01(img):
    """Min-max scale to [0,1], ignoring NaNs."""
    x = np.array(img, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    vmin = np.min(x)
    vmax = np.max(x)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(x, dtype=np.float32)
    return (x - vmin) / (vmax - vmin)

def print_stats(name, arr):
    """Print min, max, mean for a channel."""
    arr_f = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    print(f"{name:15s} min={arr_f.min():.6f}, max={arr_f.max():.6f}, mean={arr_f.mean():.6f}")

H5_PATH = (
    "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/"
    "HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"
)
SAVE_DIR = "./out_scaled_combinations"

# channels to combine
COMBO_CHANNELS = ["range", "intensity", "ambient", "reflectivity"]

if __name__ == "__main__":
    if not os.path.isfile(H5_PATH):
        sys.exit(f"File not found: {H5_PATH}")

    with h5py.File(H5_PATH, 'r') as f:
        h, w = int(f.attrs['height']), int(f.attrs['width'])

        for scene, grp in f.items():
            pts = grp['points'][()]
            cols = [c.decode() for c in grp['points'].attrs['columns']]
            get_idx = {name: i for i, name in enumerate(cols)}

            # load available channels
            channels = {}
            for name in COMBO_CHANNELS:
                if name in get_idx:
                    arr = pts[:, get_idx[name]].reshape(h, w)
                    channels[name] = arr
                    print_stats(name, arr)  # raw stats
                else:
                    channels[name] = None

            # generate all 3-channel combinations from the available ones
            avail = {k: v for k, v in channels.items() if v is not None}
            if len(avail) < 3:
                print(f"[!] {scene}: need ≥3 channels — skipping")
                continue

            for combo in itertools.combinations(avail.keys(), 3):
                r_name, g_name, b_name = combo
                r_s = scale01(avail[r_name])
                g_s = scale01(avail[g_name])
                b_s = scale01(avail[b_name])

                rgb = np.stack([r_s, g_s, b_s], axis=-1).astype(np.float32)

                # print scaled stats
                print(f"\n[✓] {scene}: scaled RGB from ({r_name}, {g_name}, {b_name})")
                print_stats(r_name + "_scaled", r_s)
                print_stats(g_name + "_scaled", g_s)
                print_stats(b_name + "_scaled", b_s)

                # save PNG
                os.makedirs(SAVE_DIR, exist_ok=True)
                rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                out_path = os.path.join(
                    SAVE_DIR, f"{scene}_rgb_{r_name}_{g_name}_{b_name}.png"
                )
                imageio.imwrite(out_path, rgb_u8)
                print(f"[file] {out_path}")
