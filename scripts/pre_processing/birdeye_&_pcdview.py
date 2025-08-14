def visualize_model_bird_eye(
        model_pcd_path: str,
        point_size: float = 1,
        cmap: str = 'viridis',
        figsize: tuple = (6, 6)
) -> None:
    """
    Display bird’s-eye view of the aircraft model.
    """
    import numpy as np;
    import open3d as o3d;
    import matplotlib.pyplot as plt
    pts = np.asarray(o3d.io.read_point_cloud(model_pcd_path).points)
    fig = plt.figure(figsize=figsize)
    sc = plt.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=point_size, cmap=cmap, linewidth=0)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('X [m]');
    plt.ylabel('Y [m]');
    plt.title("Aircraft Model Bird’s-Eye View")
    cb = plt.colorbar(sc, fraction=0.046, pad=0.04);
    cb.set_label('Height [m]')
    plt.show()

model_pcd_path="/home/femi/Downloads/pcds/02691156/1e9ef313876bfba7d02c6d35cc802839.pcd"
visualize_model_bird_eye(model_pcd_path)
# import open3d as o3d
#
# pcd = o3d.io.read_point_cloud("clean_no_ground.pcd")
# print(pcd)  # shows point count and if colors are present
#
# o3d.visualization.draw_geometries([pcd],
#                                   window_name="Clustered Point Cloud",
#                                   width=1200,
#                                   height=800,
#                                   point_show_normal=False)
