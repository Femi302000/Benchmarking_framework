import os
import glob
import numpy as np
import open3d as o3d
import json

# ---- Paths & Transform ----
source_path = "/home/femi/Benchmarking_framework/Data/Aircraft_models/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_model.pcd"
target_folder = "/home/femi/Benchmarking_framework/Data/sequence_from_scene/HAM_Airport_2024_08_08_movement_a320_ceo_Germany"

label_dir = "/home/femi/Benchmarking_framework/Data/Machine_learning_dataset/label"
scene_dir = "/home/femi/Benchmarking_framework/Data/Machine_learning_dataset/scene"

# Ensure output directories exist
os.makedirs(label_dir, exist_ok=True)
os.makedirs(scene_dir, exist_ok=True)

# Pattern to match all filtered PCD files
target_pattern = os.path.join(target_folder, "*_filtered.pcd")

# Nearest-neighbor distance threshold (in meters)
distance_threshold = 4.5  # adjust based on sensor noise and scale

for target_path in glob.glob(target_pattern):
    scene_base = os.path.splitext(os.path.basename(target_path))[0]

    # Derive JSON filename
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

    # Build KD-Tree on source cloud
    pcd_src_tree = o3d.geometry.KDTreeFlann(pcd_src)

    # Load target scene
    pcd_tgt = o3d.io.read_point_cloud(target_path)
    pts = np.asarray(pcd_tgt.points)

    # Label points based on nearest-neighbor distance
    labels = np.zeros(len(pts), dtype=np.uint8)
    for i, pt in enumerate(pts):
        k, idx_knn, dist2 = pcd_src_tree.search_knn_vector_3d(pt, 1)
        if k > 0 and np.sqrt(dist2[0]) < distance_threshold:
            labels[i] = 1

    # Remove exact duplicate points and keep labels
    unique_pts, unique_idx = np.unique(pts, axis=0, return_index=True)
    unique_labels = labels[unique_idx]

    import numpy as np


    print(f"Saved points+labels to '{out_path}'")

    # Save full scene point cloud
    scene_out = os.path.join(scene_dir, f"{scene_base}.pcd")
    o3d.io.write_point_cloud(scene_out, pcd_tgt)
    print(f"Saved full scene point cloud to '{scene_out}'")
