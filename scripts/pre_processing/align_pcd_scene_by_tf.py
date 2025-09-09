import numpy as np
import open3d as o3d

# === User-specified paths ===
target_pcd_path = "/home/femi/Benchmarking_framework/Data/sequence_from_scene/HAM_Airport_2024_08_08_movement_a320_ceo_Germany/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_scene0003_10.595s_filtered.pcd"
source_pcd_path = "/home/femi/Benchmarking_framework/Data/Aircraft_models/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_model.pcd"

# === Transformation matrix ===
matrix = np.array([
    [np.float64(0.16983626782962624), np.float64(-0.9827425479879087), np.float64(-0.07329852879040064), np.float64(-0.7)],
    [np.float64(0.9827425479879087), np.float64(0.17442894011998344), np.float64(-0.061575785260374716), np.float64(-4.296)],
    [np.float64(0.07329852879040064), np.float64(-0.061575785260374716), np.float64(0.9954073277096428), np.float64(1.51)],
    [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(1.0)],
])
# matrix = np.array([
#     [0.072, -0.997, 0.024, 3.086],
#     [0.997, -0.073, -0.023, 0.490],
#     [0.024, 0.022, 0.999, -2.028],
#     [0.000, 0.000, 0.000, 1.000],
# ], dtype=float)

# === Load point clouds ===
source_pcd = o3d.io.read_point_cloud(str(source_pcd_path))
target_pcd = o3d.io.read_point_cloud(str(target_pcd_path))

# === Apply transform to source ===
source_pcd.transform(matrix)

# === Visualize aligned source (red) and target (green) together ===
source_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red for source
target_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # green for target
o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="Aligned Source vs Target")
