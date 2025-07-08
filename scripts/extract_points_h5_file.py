import copy
import h5py
import numpy as np
import open3d as o3d

from scripts.pre_processing.knn_search import label_scene_points

def process_hdf5_and_visualize_alignment(
    h5_path: str,
    source_pcd_path: str,
    distance_threshold: float = 0.2,
    points_name: str = "points_ground",
    tf_name: str = "tf_matrix",
    labels_name: str = "labels",
    visualize: bool = True
):
    """
    For each top-level group in `h5_path`:
      • read full NxM array from dataset `points_name` (XYZ + extras)
      • read 4×4 matrix from `tf_name`
      • transform the source model by T
      • label scene points
      • append labels and overwrite dataset
      • optionally visualize alignment: transformed model (blue) and scene points (gray, red for labeled)
    """
    # load the untransformed source model once
    src_model = o3d.io.read_point_cloud(source_pcd_path)

    with h5py.File(h5_path, "r+") as f:
        for grp_name, grp in f.items():
            print(f"\n[{grp_name}] processing...")

            # --- load points and column names ---
            if points_name not in grp:
                print(f"[{grp_name}] no '{points_name}' → skip")
                continue
            old_ds = grp[points_name]
            old_cols = old_ds.attrs.get("column_names", None)
            old_cols = [c.decode() for c in old_cols] if old_cols is not None else [f"col{i}" for i in range(old_ds.shape[1])]
            full_pts = old_ds[()]
            if full_pts.shape[1] < 3:
                print(f"[{grp_name}] '{points_name}' has <3 columns → skip")
                continue
            xyz = full_pts[:, :3].astype(np.float64)
            extras = full_pts[:, 3:] if full_pts.shape[1] > 3 else np.empty((xyz.shape[0], 0))

            # --- load transform ---
            if tf_name not in grp:
                print(f"[{grp_name}] no '{tf_name}' → skip")
                continue
            T = np.array(grp[tf_name][()], dtype=float)
            if T.shape != (4, 4):
                print(f"[{grp_name}] '{tf_name}' not 4×4 → skip")
                continue

            # --- transform and visualize alignment ---
            src_copy = copy.deepcopy(src_model)
            src_copy.transform(T)

            # create scene point cloud object
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(xyz)

            # label scene points
            labels = label_scene_points(src_copy, xyz, distance_threshold)
            labels = labels.reshape(-1).astype(int)

            # assign colors to scene: default gray, red for labeled
            colors = np.tile(np.array([0.5, 0.5, 0.5]), (xyz.shape[0], 1))
            red_mask = (labels == 1)
            colors[red_mask] = np.array([1.0, 0.0, 0.0])
            scene_pcd.colors = o3d.utility.Vector3dVector(colors)

            if visualize:
                # color model copy blue
                model_colors = np.tile(np.array([0.0, 0.0, 1.0]), (np.asarray(src_copy.points).shape[0], 1))
                src_copy.colors = o3d.utility.Vector3dVector(model_colors)
                print(f"[{grp_name}] visualizing alignment: blue=model, gray=scene, red=labeled")
                o3d.visualization.draw_geometries(
                    [src_copy, scene_pcd],
                    window_name=f"Alignment_{grp_name}",
                    width=1024, height=768
                )

            # --- recombine and overwrite dataset ---
            new_pts = np.hstack([xyz, extras, labels.reshape(-1,1)])
            new_cols = old_cols + [labels_name]
            del grp[points_name]
            ds = grp.create_dataset(points_name, data=new_pts, compression="gzip")
            ds.attrs["column_names"] = np.array(new_cols, dtype="S")
            print(f"[{grp_name}] updated dataset to shape {new_pts.shape} with columns {new_cols}")

        f.flush()


if __name__ == "__main__":
    H5_FILE = "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"
    SOURCE_PCD = "/home/femi/Benchmarking_framework/Data/Aircraft_models/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_model.pcd"

    process_hdf5_and_visualize_alignment(
        h5_path=H5_FILE,
        source_pcd_path=SOURCE_PCD,
        distance_threshold=0.2,
        visualize=False
    )
