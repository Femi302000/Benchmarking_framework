import os

import numpy as np

# Default directory for saving label files
LABEL_DIR = "/home/femi/Benchmarking_framework/Data/Machine_learning_dataset/label"


def save_labels(
        pts: np.ndarray,
        labels: np.ndarray,
        base_name: str,
        fmt: str = "txt",
        out_dir: str = LABEL_DIR
) -> None:
    """Save labeled points under a given base_name in the default label directory.

    Args:
        pts: (N,3) array of point coordinates.
        labels: (N,) array of integer labels.
        base_name: name of the scene (without extension).
        fmt: output format, either 'txt' or 'npz'. Defaults to 'txt'.
        out_dir: base directory where files will be saved. Defaults to LABEL_DIR.

    Depending on fmt, this will create either:
      - out_dir/base_name.txt   (text: x y z label per line)
      - out_dir/base_name.npz   (NumPy .npz archive storing 'points' and 'labels')
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Build full output path
    ext = fmt.lower()
    if ext not in ("txt", "npz"):
        raise ValueError(f"Unsupported format '{fmt}'. Use 'txt' or 'npz'.")
    out_path = os.path.join(out_dir, f"{base_name}.{ext}")

    if ext == "npz":
        # Save as compressed numpy archive
        np.savez_compressed(out_path, points=pts, labels=labels)
        print(f"Saved NPZ labels to {out_path}")
    else:
        # Save as text file
        with open(out_path, 'w') as f:
            for (x, y, z), lab in zip(pts, labels):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {lab}\n")
        print(f"Saved TXT labels to {out_path}")
