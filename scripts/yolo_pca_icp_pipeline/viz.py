from __future__ import annotations
import numpy as np
import open3d as o3d


def create_axis_lines(origin: np.ndarray, axes: np.ndarray, length: float = 2.0) -> o3d.geometry.LineSet:
    pts = np.stack([origin, origin + length*axes[:,0], origin + length*axes[:,1], origin + length*axes[:,2]], axis=0)
    lines = np.array([[0,1],[0,2],[0,3]], dtype=np.int32)
    colors = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def visualize_with_highlight_and_axes(
    pts: np.ndarray,
    target_pt: np.ndarray,
    origin: np.ndarray,
    axes: np.ndarray,
    axis_length: float,
    point_size: float,
    colors_cfg: dict,
    save_colored_pcd: bool,
    colored_pcd_path: str,
    highlight_mode: str = "sphere",
    sphere_radius: float = 0.1,
):
    colors = np.tile(np.array(colors_cfg.get("other_color", (0.65, 0.65, 0.65)), dtype=np.float64), (pts.shape[0], 1))

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geoms = [pcd]

    if highlight_mode == "sphere":
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(target_pt.astype(np.float64))
        sphere.paint_uniform_color(colors_cfg.get("sphere_color", (1.0, 0.0, 0.0)))
        sphere.compute_vertex_normals()
        geoms.append(sphere)

    geoms.append(create_axis_lines(origin, axes, length=axis_length))

    if save_colored_pcd:
        o3d.io.write_point_cloud(colored_pcd_path, pcd, write_ascii=False, compressed=False, print_progress=False)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PCD + PCA Axes + Target", width=1280, height=800)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = float(point_size)
        opt.background_color = np.array([0, 0, 0])
    vis.run()
    vis.destroy_window()