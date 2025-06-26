def visualize_before_after(
    raw_pcd_path: str,
    filt_pcd_path: str,
    point_size: float = 1,
    cmap: str = 'viridis',
    figsize: tuple = (10, 5)
) -> None:
    """
    Display two bird’s-eye view panels: before (raw) and after (filtered).
    """
    import numpy as np; import open3d as o3d; import matplotlib.pyplot as plt
    raw_pts = np.asarray(o3d.io.read_point_cloud(raw_pcd_path).points)
    filt_pts = np.asarray(o3d.io.read_point_cloud(filt_pcd_path).points)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    sc1 = ax1.scatter(raw_pts[:, 0], raw_pts[:, 1], c=raw_pts[:, 2], s=point_size, cmap=cmap, linewidth=0)
    ax1.set_aspect('equal', 'box'); ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]')
    ax1.set_title('Raw Bird’s-Eye View'); ax1.grid(True)
    cb1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04); cb1.set_label('Height [m]')
    sc2 = ax2.scatter(filt_pts[:, 0], filt_pts[:, 1], c=filt_pts[:, 2], s=point_size, cmap=cmap, linewidth=0)
    ax2.set_aspect('equal', 'box'); ax2.set_xlabel('X [m]'); ax2.set_ylabel('Y [m]')
    ax2.set_title('Filtered Bird’s-Eye View'); ax2.grid(True)
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04); cb2.set_label('Height [m]')
    plt.tight_layout()
    plt.show()

raw_pcd_path="/home/femi/Benchmarking_framework/Data/sequence_from_scene/HAM_Airport_2024_08_08_movement_a320_ceo_Germany/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_scene0000_8.996s_raw.pcd"
filt_pcd_path="/home/femi/Benchmarking_framework/Data/sequence_from_scene/HAM_Airport_2024_08_08_movement_a320_ceo_Germany/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_scene0000_8.996s_filtered.pcd"
visualize_before_after(raw_pcd_path,filt_pcd_path)