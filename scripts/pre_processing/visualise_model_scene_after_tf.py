import json
from pathlib import Path


def visualize_overlay(
        scene_pcd_path: str,
        model_pcd_path: str,
        point_size: float = 1,
        scene_color: tuple = (0.5, 0.5, 0.5),
        cmap: str = 'viridis',
        figsize: tuple = (6, 6)
) -> None:
    import numpy as np;
    import open3d as o3d;
    import matplotlib.pyplot as plt
    scene_pts = np.asarray(o3d.io.read_point_cloud(scene_pcd_path).points)
    # derive base stem without suffix
    scene_stem = Path(scene_pcd_path).stem.replace('_filtered', '').replace('_raw', '')
    tf_path = Path(scene_pcd_path).parent / f"{scene_stem}_tf.json"
    with open(tf_path, 'r') as f:
        mat = np.array(json.load(f)['matrix'], dtype=float)
    model_pcd = o3d.io.read_point_cloud(model_pcd_path);
    model_pcd.transform(mat)
    model_pts = np.asarray(model_pcd.points)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(scene_pts[:, 0], scene_pts[:, 1], c=[scene_color], s=point_size, linewidth=0)
    sc = ax.scatter(model_pts[:, 0], model_pts[:, 1], c=model_pts[:, 2], s=point_size, cmap=cmap, linewidth=0)
    ax.set_aspect('equal', 'box');
    ax.set_xlabel('X [m]');
    ax.set_ylabel('Y [m]');
    ax.set_title(scene_stem)
    cb = plt.colorbar(sc, fraction=0.046, pad=0.04);
    cb.set_label('Height [m]')
    plt.show()
# bag_path="/home/femi/Benchmarking_framework/Data/bag_files/HAM_Airport_2024_08_08_movement_a320_ceo_Germany"
# bag_name = Path(bag_path).stem
# out_dir = Path(bag_path).parents[1] / "sequence_from_scene" / bag_name
# scene_files = sorted(out_dir.glob(f"{bag_name}_scene????_*_filtered.pcd"))
# scene_pcd_path="/home/femi/Benchmarking_framework/Data/sequence_from_scene/HAM_Airport_2024_08_08_movement_a320_ceo_Germany/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_scene0000_8.996s_filtered.pcd"
# model_pcd_path="/home/femi/Benchmarking_framework/Data/Aircraft_models/Air_France_25_01_2022_Air_France_aircraft_front_A319_Ceo_25-01-2022-12-21-23_fix.pcd"
# for scene in scene_files:
#     visualize_overlay(str(scene), str(model_pcd_path))
