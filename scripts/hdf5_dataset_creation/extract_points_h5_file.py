#!/usr/bin/env python3
import copy
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d

from scripts.hdf5_dataset_creation.helper_functions.knn_search import label_scene_points


def process_hdf5_and_visualize_alignment(
    h5_path: str,
    source_pcd_path: str,
    distance_threshold: float = 0.2,
    points_name: str = "points",
    tf_name: str = "tf_matrix",
    labels_name: str = "is_aircraft",
):
    """
    For each scene_NNN in the HDF5:
      • load point array from <scene>/points
      • load TF from <scene>/metadata/tf_matrix
      • transform source model and label points via KNN
      • append labels as new column (labels_name), unless it already exists
      • update "columns" attr
    """
    # Load the untransformed source model once
    src_model = o3d.io.read_point_cloud(source_pcd_path)

    with h5py.File(h5_path, "r+") as f:
        for grp_name, grp in f.items():
            print(f"\n[{grp_name}] processing...")

            # 1) Load points dataset, handling group-vs-dataset
            if points_name not in grp:
                print(f"  → no '{points_name}' in {grp_name}, skipping")
                continue
            member = grp[points_name]
            if isinstance(member, h5py.Group):
                if "all" not in member:
                    print(f"  → '{points_name}' group has no 'all', skipping")
                    continue
                ds = member["all"]
            else:
                ds = member

            pts = ds[()]  # shape (N, >=3)

            # 1a) Columns
            cols_attr = ds.attrs.get("columns", None)
            if cols_attr is not None:
                cols = [c.decode('utf-8') for c in cols_attr]
            else:
                cols = ["x","y","z","range","intensity","reflectivity","ambient"]

            # Skip if labels already present
            if labels_name in cols:
                print(f"  → '{labels_name}' already present, skipping this scene")
                continue

            if pts.ndim != 2 or pts.shape[1] < 3:
                print(f"  → invalid shape {pts.shape}, skipping")
                continue

            xyz = pts[:, :3].astype(np.float64)
            extras = pts[:, 3:]

            # 2) Load transform
            meta = grp.get("metadata")
            if meta is None or tf_name not in meta:
                print(f"  → no metadata/{tf_name}, skipping")
                continue
            T = np.array(meta[tf_name][()], dtype=float)
            if T.shape != (4,4):
                print(f"  → invalid tf_matrix shape {T.shape}, skipping")
                continue

            # 3) Label via KNN
            src_copy = copy.deepcopy(src_model)
            src_copy.transform(T)
            labels = label_scene_points(src_copy, xyz, distance_threshold).reshape(-1,1).astype(np.uint8)

            # 4) Append labels column (recreate dataset because shape changes)
            new_pts = np.hstack([xyz.astype(np.float32), extras.astype(np.float32), labels.astype(np.float32)])
            new_cols = cols + [labels_name]

            # remove old
            if isinstance(member, h5py.Group):
                del member["all"]
                ds_parent = member
                new_name = "all"
            else:
                del grp[points_name]
                ds_parent = grp
                new_name = points_name

            # create new under same name
            new_ds = ds_parent.create_dataset(
                new_name,
                data=new_pts,
                compression="gzip"
            )
            new_ds.attrs["columns"] = np.array(new_cols, dtype="S")
            print(f"  → '{grp_name}/{new_name}' now shape {new_pts.shape}, columns={new_cols}")

        f.flush()


if __name__ == "__main__":
    H5_FILE = Path("/Data/machine_learning_dataset/HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5")
    MODEL   = Path("/Data/Aircraft_models/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_model.pcd")

    process_hdf5_and_visualize_alignment(
        h5_path=str(H5_FILE),
        source_pcd_path=str(MODEL),
        distance_threshold=0.2,
    )
