from __future__ import annotations
from typing import List, Optional, Tuple
import os
import numpy as np
import cv2
import imageio.v2 as imageio

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

import open3d as o3d


def yolo_segment_to_mask(rgb: np.ndarray, conf: float, iou: float, img_size: int) -> tuple[np.ndarray, Optional[tuple[int,int,int,int]]]:
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics not available.")
    # Using default small seg model unless caller passes different weights to YOLO()
    model = YOLO("yolov8s-seg.pt")
    aircraft_cls = [i for i, n in model.names.items() if "airplane" in n.lower() or "aircraft" in n.lower()]
    res = model.predict(source=rgb, conf=conf, iou=iou, imgsz=img_size, retina_masks=True,
                        classes=(aircraft_cls if aircraft_cls else None), verbose=False)
    h, w = rgb.shape[:2]
    union_mask = np.zeros((h, w), dtype=np.uint8)
    best_box, best_score = None, 0.0

    for r in res:
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy().astype(float)
            for box, c, s in zip(xyxy, cls, confs):
                if (not aircraft_cls) or (c in aircraft_cls):
                    if s > best_score:
                        best_score, best_box = s, tuple(map(int, box))
        if r.masks is not None and r.masks.data is not None and len(r.masks.data):
            m = r.masks.data.cpu().numpy()  # [N,h',w']
            for i in range(m.shape[0]):
                mi = (m[i] * 255).astype(np.uint8)
                if mi.shape[:2] != (h, w):
                    mi = cv2.resize(mi, (w, h), interpolation=cv2.INTER_NEAREST)
                union_mask = np.maximum(union_mask, mi)
    return union_mask, best_box


def segmask_to_pcd_files(output_dir: str, scene_id: str, pts: np.ndarray, cols: list[str], mask_bool: np.ndarray,
                          use_gray_from: str = "reflectivity", remove_ground_pcd: bool = True) -> tuple[str | None, str | None]:
    os.makedirs(output_dir, exist_ok=True)
    mask_flat = mask_bool.reshape(-1).astype(bool)
    keep = mask_flat.copy()

    xyz = pts[:, :3][keep]
    if xyz.size == 0:
        return None, None

    # colors
    if use_gray_from in cols:
        scal = pts[:, cols.index(use_gray_from)][keep]
        vmin, vmax = np.percentile(scal, (1, 99))
        gray = np.clip((scal - vmin) / (vmax - vmin + 1e-12), 0, 1)
        cols_rgb = np.stack([gray, gray, gray], axis=1)
    else:
        cols_rgb = np.tile(np.array([[1.0, 0.6, 0.0]]), (xyz.shape[0], 1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols_rgb.astype(np.float64))
    path_all = os.path.join(output_dir, f"{scene_id}_seg_points.pcd")
    o3d.io.write_point_cloud(path_all, pcd)

    path_ng = None
    if ("is_ground" in cols) and remove_ground_pcd:
        g = pts[:, cols.index("is_ground")].astype(bool)
        keep_ng = keep & (~g)
        xyz_ng = pts[:, :3][keep_ng]
        if xyz_ng.size:
            if use_gray_from in cols:
                scal_ng = pts[:, cols.index(use_gray_from)][keep_ng]
                vmin, vmax = np.percentile(scal_ng, (1, 99))
                gray = np.clip((scal_ng - vmin) / (vmax - vmin + 1e-12), 0, 1)
                cols_ng = np.stack([gray, gray, gray], axis=1)
            else:
                cols_ng = np.tile(np.array([[1.0, 0.6, 0.0]]), (xyz_ng.shape[0], 1))
            pcd_ng = o3d.geometry.PointCloud()
            pcd_ng.points = o3d.utility.Vector3dVector(xyz_ng.astype(np.float64))
            pcd_ng.colors = o3d.utility.Vector3dVector(cols_ng.astype(np.float64))
            path_ng = os.path.join(output_dir, f"{scene_id}_seg_points_noground.pcd")
            o3d.io.write_point_cloud(path_ng, pcd_ng)

    return path_all, path_ng


def save_overlay_images(output_dir: str, scene_id: str, rgb: np.ndarray, mask_bool: np.ndarray, best_box):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    os.makedirs(output_dir, exist_ok=True)
    rgb_path = os.path.join(output_dir, f"{scene_id}_rgb_rai.png")
    imageio.imwrite(rgb_path, rgb)

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(rgb)
        ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none'))
        ax.axis('off')
        overlay_path = os.path.join(output_dir, f"{scene_id}_bbox.png")
        fig.savefig(overlay_path, bbox_inches="tight")
        plt.close(fig)

    seg_png = os.path.join(output_dir, f"{scene_id}_seg_union.png")
    rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
    rgba[:, :, 3] = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(seg_png, rgba)

    return rgb_path, seg_png