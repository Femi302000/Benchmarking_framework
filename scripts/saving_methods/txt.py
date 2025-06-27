import os

import numpy as np
LABEL_DIR = "/home/femi/Benchmarking_framework/Data/Machine_learning_dataset/label"


def save_labels_txt(
    pts: np.ndarray,
    labels: np.ndarray,
    base_name: str
) -> None:
    """Save labeled points to a text file: x y z label per line."""
    os.makedirs(LABEL_DIR, exist_ok=True)
    out_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
    with open(out_path, 'w') as f:
        for (x, y, z), lab in zip(pts, labels):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {lab}\n")
    print(f"Saved TXT labels to {out_path}")