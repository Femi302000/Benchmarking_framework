def visualize_model_bird_eye(
    model_pcd_path: str,
    point_size: float = 1,
    cmap: str = 'viridis',
    figsize: tuple = (6, 6)
) -> None:
    """
    Display bird’s-eye view of the aircraft model.
    """
    import numpy as np; import open3d as o3d; import matplotlib.pyplot as plt
    pts = np.asarray(o3d.io.read_point_cloud(model_pcd_path).points)
    fig = plt.figure(figsize=figsize)
    sc = plt.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=point_size, cmap=cmap, linewidth=0)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('X [m]'); plt.ylabel('Y [m]'); plt.title("Aircraft Model Bird’s-Eye View")
    cb = plt.colorbar(sc, fraction=0.046, pad=0.04); cb.set_label('Height [m]')
    plt.show()
    
model_pcd_path="/home/femi/Benchmarking_framework/Data/Aircraft_models/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_model.pcd"
visualize_model_bird_eye(model_pcd_path)