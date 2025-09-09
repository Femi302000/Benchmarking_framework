# def visualize_model_bird_eye(
#         model_pcd_path: str,
#         point_size: float = 1,
#         cmap: str = 'viridis',
#         figsize: tuple = (6, 6)
# ) -> None:
#     """
#     Display bird’s-eye view of the aircraft model.
#     """
#     import numpy as np;
#     import open3d as o3d;
#     import matplotlib.pyplot as plt
#     pts = np.asarray(o3d.io.read_point_cloud(model_pcd_path).points)
#     fig = plt.figure(figsize=figsize)
#     sc = plt.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=point_size, cmap=cmap, linewidth=0)
#     plt.gca().set_aspect('equal', 'box')
#     plt.xlabel('X [m]');
#     plt.ylabel('Y [m]');
#     plt.title("Aircraft Model Bird’s-Eye View")
#     cb = plt.colorbar(sc, fraction=0.046, pad=0.04);
#     cb.set_label('Height [m]')
#     plt.show()
#
# model_pcd_path="/home/femi/Benchmarking_framework/scripts/yolo/seg_outputs/scene_000_seg_points_with_axes___noground.pcd"
# visualize_model_bird_eye(model_pcd_path)
import open3d as o3d
import numpy as np

# --------- EDIT THESE TWO PATHS ----------
SCENE_PCD_PATH = "/home/femi/Benchmarking_framework/scripts/yolo/seg_outputs/scene_000_seg_points_noground.pcd"
KPS_PLY_PATH   = "/home/femi/Benchmarking_framework/scripts/keypoints/kps_pred_snapped.ply"
# ----------------------------------------

# Vis style knobs
BG_COLOR        = np.array([0, 0, 0])   # black background
SCENE_COLOR     = np.array([1.0, 1.0, 1.0])  # white point cloud
SCENE_POINTSIZE = 1.5
KP_SPHERE_RAD   = 0.25
KP_SPHERE_RES   = 12    # sphere mesh resolution

def load_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"No points in: {path}")
    return pcd

def make_sphere(center, radius, color, res=12):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=res)
    s.translate(center)
    s.paint_uniform_color(color)
    s.compute_vertex_normals()
    return s

def distinct_color(i):
    # simple deterministic color wheel
    return np.array([((i*37)%255)/255.0, ((i*97)%255)/255.0, ((i*17)%255)/255.0])

def main():
    # Load scene
    scene = load_cloud(SCENE_PCD_PATH)
    scene.paint_uniform_color(SCENE_COLOR)

    # Load predicted keypoints as a small point cloud
    kps = load_cloud(KPS_PLY_PATH)
    kp_xyz = np.asarray(kps.points)
    # If your PLY already has colors (from training script), we’ll use them; else generate.
    if kps.has_colors():
        kp_colors = np.asarray(kps.colors)
    else:
        kp_colors = np.array([distinct_color(i) for i in range(kp_xyz.shape[0])])

    # Build sphere meshes for keypoints
    kp_spheres = []
    for i, p in enumerate(kp_xyz):
        kp_spheres.append(make_sphere(p, KP_SPHERE_RAD, kp_colors[i], res=KP_SPHERE_RES))

    # Optional: a small coordinate frame at the scene centroid
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=scene.get_center()
    )

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Scene + Predicted Keypoints", width=1200, height=800)
    opt = vis.get_render_option()
    opt.background_color = BG_COLOR
    opt.point_size = SCENE_POINTSIZE
    vis.add_geometry(scene)
    for s in kp_spheres:
        vis.add_geometry(s)
    vis.add_geometry(frame)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()

