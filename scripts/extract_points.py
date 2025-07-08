import glob
import json
import os

import numpy as np
import open3d as o3d

from scripts.pre_processing.knn_search import label_scene_points
from scripts.pre_processing.labels_to_npz_txt import save_labels

# ---- Paths & Transform ----
source_path = "/Data/Aircraft_models/Air_France_25_01_2022_Air_France_aircraft_front_A319_Ceo_25-01-2022-12-21-23_fix.pcd"
target_folder = "/home/femi/Benchmarking_framework/Data/sequence_from_scene/HAM_Airport_2024_08_08_movement_a320_ceo_Germany"
distance_threshold = 0.2
label_dir = "/home/femi/Benchmarking_framework/Data/Machine_learning_dataset/label"
scene_dir = "/home/femi/Benchmarking_framework/Data/Machine_learning_dataset/scene"

# Ensure output directories exist
os.makedirs(label_dir, exist_ok=True)
os.makedirs(scene_dir, exist_ok=True)

# Pattern to match all filtered PCD files
target_pattern = os.path.join(target_folder, "*_filtered.pcd")
# Nearest-neighbor distance threshold (in meters)
 # adjust based on sensor noise and scale

for target_path in glob.glob(target_pattern):
    scene_base = os.path.splitext(os.path.basename(target_path))[0]


    clean_base = scene_base.removesuffix("_filtered")
    json_path = os.path.join(target_folder, f"{clean_base}_tf.json")
    if not os.path.exists(json_path):
        print(f"Warning: transformation file not found: {json_path}. Skipping {scene_base}.")
        continue

    # Load transformation matrix
    with open(json_path, 'r') as jf:
        data = json.load(jf)

    # Try common keys for the matrix
    transform = data.get('transform') or data.get('transformation') or data.get('matrix')
    if transform is None:
        # Search for any 4x4 matrix in JSON
        for v in data.values():
            if isinstance(v, list) and len(v) == 4 and all(isinstance(row, list) and len(row) == 4 for row in v):
                transform = v
                break
    if transform is None:
        print(f"Warning: no 4x4 matrix found in {json_path}. Skipping {scene_base}.")
        continue

    T = np.array(transform, dtype=float)

    # Load and transform source model
    pcd_src = o3d.io.read_point_cloud(source_path)
    pcd_src.transform(T)

    # Load target scene points
    pcd_tgt = o3d.io.read_point_cloud(target_path)
    pts = np.asarray(pcd_tgt.points)

    # Label points using the refactored function
    labels = label_scene_points(pcd_src, pts, distance_threshold)

    # Remove exact duplicate points and keep labels
    unique_pts, unique_idx = np.unique(pts, axis=0, return_index=True)
    unique_labels = labels[unique_idx]

    # Save labels and full scene
    save_labels(pts, labels, base_name=scene_base, fmt="npz")
    scene_out = os.path.join(scene_dir, f"{scene_base}.pcd")
    o3d.io.write_point_cloud(scene_out, pcd_tgt)
    print(f"Saved full scene point cloud to '{scene_out}'")
