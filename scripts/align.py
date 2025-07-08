import numpy as np
import open3d as o3d

# === User-specified paths ===
target_pcd_path = "/home/femi/Evitado/Benchmarking_framework/Data/sequence_from_scene/HAM_Airport_2024_08_08_movement_a321_ceo_unknown_2024-08-08T09-48-39/HAM_Airport_2024_08_08_movement_a321_ceo_unknown_2024-08-08T09-48-39_scene0000_0.000s_filtered.pcd"
source_pcd_path = "/home/femi/Evitado/Benchmarking_framework/Data/Aircraft_models/HAM_Airport_2024_08_08_movement_a321_ceo_unknown_2024-08-08T09-48-39_model.pcd"

# === Transformation matrix ===
matrix = np.array([
    [0.072, -0.997, 0.024, 3.086],
    [0.997, -0.073, -0.023, 0.490],
    [0.024, 0.022, 0.999, -2.028],
    [0.000, 0.000, 0.000, 1.000],
], dtype=float)

# === Load point clouds ===
source_pcd = o3d.io.read_point_cloud(str(source_pcd_path))
target_pcd = o3d.io.read_point_cloud(str(target_pcd_path))

# === Apply transform to source ===
source_pcd.transform(matrix)

# === Visualize aligned source (red) and target (green) together ===
source_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red for source
target_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # green for target
o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="Aligned Source vs Target")
