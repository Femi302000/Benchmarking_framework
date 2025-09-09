import numpy as np
import open3d as o3d

# ========= CONFIG =========
PC_PATH = "/home/femi/Benchmarking_framework/scripts/method_trials/aircraft_completed/scene_000_seg_points_noground_completed.pcd"   # change to your file
AXIS_LEN = 0.5             # visual length of axes (units of your cloud)
SPHERE_RADIUS = 0.05       # centroid sphere radius
# =========================

# 1) Load cloud
pcd = o3d.io.read_point_cloud(PC_PATH)
if len(pcd.points) == 0:
    raise ValueError(f"No points loaded from {PC_PATH}")

pts = np.asarray(pcd.points)

# 2) Centroid
centroid = pts.mean(axis=0)
print("Centroid [x, y, z]:", centroid.tolist())

# 3) Centroid sphere
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
sphere.compute_vertex_normals()
sphere.paint_uniform_color([1, 0.6, 0])  # orange
sphere.translate(centroid)

# 4) Coordinate frame at centroid
# (X=red, Y=green, Z=blue)
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_LEN)
frame.translate(centroid)

# 5) Optional: axis lines centered at centroid
axis_pts = np.array([
    centroid, centroid + np.array([AXIS_LEN, 0, 0]),  # X
    centroid, centroid + np.array([0, AXIS_LEN, 0]),  # Y
    centroid, centroid + np.array([0, 0, AXIS_LEN]),  # Z
])
axis_lines = [[0,1],[2,3],[4,5]]
axis_colors = [[1,0,0],[0,1,0],[0,0,1]]

axes = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(axis_pts),
    lines=o3d.utility.Vector2iVector(axis_lines)
)
axes.colors = o3d.utility.Vector3dVector(axis_colors)

# 6) (Optional) colorize points lightly for contrast
if not pcd.has_colors():
    # simple grayscale by height (z) for visibility
    z = pts[:,2]
    z_norm = (z - z.min()) / max(1e-9, (z.max() - z.min()))
    colors = np.stack([z_norm, z_norm, z_norm], axis=1) * 0.8 + 0.2
    pcd.colors = o3d.utility.Vector3dVector(colors)

# 7) Visualize
o3d.visualization.draw_geometries([pcd, sphere, frame, axes])
