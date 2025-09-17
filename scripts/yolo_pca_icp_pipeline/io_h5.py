
from __future__ import annotations
from typing import List, Tuple
import os
import numpy as np

try:
    import h5py
    H5_AVAILABLE = True
except Exception:
    H5_AVAILABLE = False


def load_h5_scene(h5_path: str, scene_id: str):
    if not H5_AVAILABLE:
        raise ImportError("h5py is required.")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, "r") as f:
        if scene_id not in f:
            raise KeyError(f"Scene '{scene_id}' not in H5")
        grp = f[scene_id]
        pts = grp["points"][()]  # (N, C)
        cols = [c.decode() for c in grp["points"].attrs["columns"]]
        h, w = int(f.attrs["height"]), int(f.attrs["width"])
    return pts, cols, h, w


def autoscale(img, nan_fill=0.0):
    clean = np.nan_to_num(img, nan=nan_fill)
    vmin, vmax = np.percentile(clean, (1, 99))
    return clean, vmin, vmax


def norm_uint8(img):
    clean, vmin, vmax = autoscale(img)
    norm = np.clip((clean - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return (norm * 255).astype(np.uint8)


def build_rai_rgb(pts: np.ndarray, cols: list[str], h: int, w: int) -> np.ndarray:
    if not all(k in cols for k in ("reflectivity", "ambient", "intensity")):
        raise ValueError("H5 is missing reflectivity/ambient/intensity columns to build RAI RGB.")
    idx = {name: cols.index(name) for name in ("reflectivity", "ambient", "intensity")}
    r = norm_uint8(pts[:, idx["reflectivity"]].reshape(h, w))
    g = norm_uint8(pts[:, idx["ambient"]].reshape(h, w))
    b = norm_uint8(pts[:, idx["intensity"]].reshape(h, w))
    return np.stack([r, g, b], axis=-1)



def read_scene_gt_transform(h5_path: str, scene_id: str) -> np.ndarray | None:
    """
    Reads ground-truth 4x4 transform from /{scene_id}/metadata/tf_matrix.
    Returns None if missing.
    """
    with h5py.File(h5_path, "r") as f:
        if scene_id not in f:
            return None
        g = f[scene_id]
        if "metadata" not in g or "tf_matrix" not in g["metadata"]:
            return None
        T = np.asarray(g["metadata"]["tf_matrix"][()], dtype=float)  # shape (4,4)
        if T.size != 16:
            raise ValueError(f"metadata/tf_matrix has unexpected shape {T.shape}")
        T = T.reshape(4, 4)
        # normalize last row just in case
        T[-1, :] = [0.0, 0.0, 0.0, 1.0]
        return T

        return T
def _to_4x4(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape == (4, 4):
        T = arr
    elif arr.size == 16:
        T = arr.reshape(4, 4)
    else:
        raise ValueError(f"Cannot reshape GT transform with shape {arr.shape} into 4x4")
    # ensure last row is [0,0,0,1]
    T = T.astype(float)
    if T.shape == (4, 4):
        T[-1, :] = [0.0, 0.0, 0.0, 1.0]
    return T
