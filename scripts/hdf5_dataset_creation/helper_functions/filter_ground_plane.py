import numpy as np
import open3d as o3d

def filter_by_min_z(pcd: o3d.geometry.PointCloud, z_min: float) -> np.ndarray:
    pts = np.asarray(pcd.points)
    return (pts[:, 2] <= z_min).astype(np.uint8)  # 0 = keep

# # Load PCD file
# pcd_path = "/scripts/data_dir/scene_003_red_bbox_points.pcd"
# pcd = o3d.io.read_point_cloud(pcd_path)
#
# # Apply mask
# mask = filter_by_min_z(pcd, z_min=0.1)
# indices_to_keep = np.where(mask == 1)[0]
# pcd_filtered = pcd.select_by_index(indices_to_keep)
#
# # Paint points white
# pcd_filtered.paint_uniform_color([1.0, 1.0, 1.0])  # white points
#
# # Custom visualizer with black background
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="White Points on Black Background")
# vis.add_geometry(pcd_filtered)
#
# # Set render option: black background
# opt = vis.get_render_option()
# opt.background_color = np.asarray([0, 0, 0])  # black
# opt.point_size = 2.0  # make points bigger for visibility
#
# vis.run()
# vis.destroy_window()
