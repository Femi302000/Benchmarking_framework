import os
import sys
import glob

import numpy as np
import h5py
import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import open3d as o3d

# ----------------------------------------------------------------------
# Helpers: autoscale + normalize to uint8
# ----------------------------------------------------------------------
def _autoscale(img, nan_fill=0.0):
    clean = np.nan_to_num(img, nan=nan_fill)
    vmin, vmax = np.percentile(clean, (1, 99))
    return clean, vmin, vmax

def _norm_uint8(img):
    clean, vmin, vmax = _autoscale(img)
    norm = np.clip((clean - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return (norm * 255).astype(np.uint8)

# ----------------------------------------------------------------------
# Process one scene: build RGB, run YOLO, draw & crop & colorize PCD
# ----------------------------------------------------------------------
def process_scene(pts, cols, scene, h, w, yolo, output_dir):
    # extract channels
    idx = lambda name: cols.index(name)
    channels = { name: pts[:, idx(name)] if name in cols else None
                 for name in ('reflectivity','ambient','intensity','is_ground') }
    reflect = channels['reflectivity']
    ambient = channels['ambient']
    inten = channels['intensity']
    if reflect is None or ambient is None or inten is None:
        print(f"[!] {scene}: missing one of reflectivity/ambient/intensity, skipping")
        return

    # reshape for image operations
    r = _norm_uint8(reflect.reshape(h, w))
    g = _norm_uint8(ambient.reshape(h, w))
    b = _norm_uint8(inten.reshape(h, w))
    rgb = np.stack([r, g, b], axis=-1)

    # save RGB image
    rgb_path = os.path.join(output_dir, f"{scene}_rgb_rai.png")
    imageio.imwrite(rgb_path, rgb)
    print(f"[✓] {scene}: saved RGB → {rgb_path}")

    # YOLO detection on RGB
    results = yolo.predict(source=rgb, conf=0.05, iou=0.2)
    best_score, best_box = 0.0, None
    aircraft_idxs = [i for i,name in yolo.names.items()
                     if 'aircraft' in name.lower() or 'airplane' in name.lower()]
    for res in results:
        boxes  = res.boxes.xyxy.cpu().numpy()
        cls    = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()
        for box, c, s in zip(boxes, cls, scores):
            if c in aircraft_idxs and s > best_score:
                best_score, best_box = s, box

    if best_box is None:
        print(f"[!] {scene}: no aircraft detected, skipping")
        return

    x1, y1, x2, y2 = map(int, best_box)

    # save overlay
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(rgb)
    rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    overlay_path = os.path.join(output_dir, f"{scene}_bbox.png")
    fig.savefig(overlay_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] {scene}: saved overlay → {overlay_path}")

    # save aircraft crop
    crop = rgb[y1:y2, x1:x2]
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    crop_path = os.path.join(output_dir, f"{scene}_crop.png")
    cv2.imwrite(crop_path, crop_bgr)
    print(f"[✓] {scene}: saved crop → {crop_path}")

    # ------------------------------------------------------------------
    # Colorize PCD: mark bbox points red, remove ground points
    # ------------------------------------------------------------------
    mask2d = np.zeros((h, w), dtype=bool)
    mask2d[y1:y2, x1:x2] = True
    mask_flat = mask2d.reshape(-1)

    refl_flat = channels['reflectivity']
    _, vmin, vmax = _autoscale(refl_flat)
    gray = np.clip((refl_flat - vmin) / (vmax - vmin + 1e-12), 0, 1)
    colors = np.stack([gray, gray, gray], axis=1)

    # highlight bbox in gray+red cloud
    colors[mask_flat, :] = np.array([1.0, 0.0, 0.0])

    # remove ground if available
    if channels['is_ground'] is not None:
        ground_mask = channels['is_ground'].astype(bool)
    else:
        ground_mask = np.zeros_like(mask_flat, dtype=bool)
    keep_mask = ~ground_mask

    # build no-ground colored cloud
    xyz = pts[:, :3][keep_mask]
    colors_noground = colors[keep_mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors_noground)

    pcd_path = os.path.join(output_dir, f"{scene}_colored_noground.pcd")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"[✓] {scene}: saved colored PCD without ground → {pcd_path}")

    # ------------------------------------------------------------------
    # Save a second PCD with *only* the red (bbox) points, excluding ground
    # ------------------------------------------------------------------
    red_mask = mask_flat & ~ground_mask
    xyz_red = pts[red_mask, :3]
    colors_red = np.tile(np.array([1.0, 0.0, 0.0]), (xyz_red.shape[0], 1))

    pcd_red = o3d.geometry.PointCloud()
    pcd_red.points = o3d.utility.Vector3dVector(xyz_red)
    pcd_red.colors = o3d.utility.Vector3dVector(colors_red)

    red_pcd_path = os.path.join(output_dir, f"{scene}_red_bbox_points.pcd")
    o3d.io.write_point_cloud(red_pcd_path, pcd_red)
    print(f"[✓] {scene}: saved red-only PCD → {red_pcd_path}")

# ----------------------------------------------------------------------
# Visualize saved PCDs
# ----------------------------------------------------------------------
def visualize_saved_pcds(directory="./outputs"):
    # Find files
    noground_files = glob.glob(os.path.join(directory, "*_colored_noground.pcd"))
    red_files      = glob.glob(os.path.join(directory, "*_red_bbox_points.pcd"))

    # Show the first no-ground file
    if noground_files:
        print(f"🔍 Visualizing: {noground_files[0]}")
        pcd_ng = o3d.io.read_point_cloud(noground_files[0])
        o3d.visualization.draw_geometries(
            [pcd_ng],
            window_name="Colored No-Ground Point Cloud",
            width=800, height=600, point_show_normal=False
        )
    else:
        print(f"[!] No '*_colored_noground.pcd' found in {directory}")

    # Show the first red-only file
    if red_files:
        print(f"🔍 Visualizing: {red_files[0]}")
        pcd_red = o3d.io.read_point_cloud(red_files[0])
        o3d.visualization.draw_geometries(
            [pcd_red],
            window_name="Red-Only Bounding-Box Point Cloud",
            width=800, height=600, point_show_normal=False
        )
    else:
        print(f"[!] No '*_red_bbox_points.pcd' found in {directory}")

if __name__ == "__main__":
    H5_PATH      = "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"
    YOLO_WEIGHTS = "yolov8x6.pt"
    OUTPUT_DIR   = "yolo_outputs"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load YOLO model
    yolo = YOLO(YOLO_WEIGHTS)

    # Open dataset file
    if not os.path.isfile(H5_PATH):
        sys.exit(f"File not found: {H5_PATH}")
    with h5py.File(H5_PATH, "r") as f:
        h, w = int(f.attrs["height"]), int(f.attrs["width"])
        for scene, grp in f.items():
            pts  = grp["points"][()]
            cols = [c.decode() for c in grp["points"].attrs["columns"]]
            print(f"Processing {scene}…")
            process_scene(pts, cols, scene, h, w, yolo, OUTPUT_DIR)

    # Visualize both kinds of PCDs
    visualize_saved_pcds(OUTPUT_DIR)
