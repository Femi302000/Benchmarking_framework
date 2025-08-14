from pathlib import Path
import h5py
import numpy as np

def save_scene_to_hdf5(
    bag_path: str,
    base_out_dir,
    stamp_ns: int,
    tf_matrix: np.ndarray = None,
    points_all: np.ndarray = None,
    aircraft_model: str = None,
    height=None, width=None,
):

    """
    Append a LiDAR sweep into a single HDF5 per-bag file under a sequential scene subgroup:
      - Each call opens <base_out_dir>/<bagname>.h5 in append mode
      - Creates scene_NNN group (00-based index) and writes:
         metadata/, images/, points/, ring
    """
    # prepare paths
    base_out_dir = Path(base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    bag_stem = Path(bag_path).stem
    out_file = base_out_dir / f"{bag_stem}.h5"

    # open single HDF5 for the bag
    with h5py.File(out_file, "a") as f:
        # determine next scene index
        scene_idx = len([k for k in f.keys() if k.startswith("scene_")])
        scene_name = f"scene_{scene_idx:03d}"
        grp = f.create_group(scene_name)

        # ---- 1) Metadata ----
        grp_meta = grp.create_group("metadata")
        grp_meta.create_dataset("stamp_ns", data=stamp_ns)
        if tf_matrix is not None:
            grp_meta.create_dataset("tf_matrix", data=tf_matrix)

        f.attrs["height"] = height
        f.attrs["width"] = width

        if aircraft_model is not None and "aircraft_model" not in f.attrs:
            f.attrs["aircraft_model"] = aircraft_model

        # ---- 3) Full point cloud with all channels ----
        if points_all is not None:
            dset = grp.create_dataset(
                "points",
                data=points_all.astype(np.float32),
                compression="gzip"
            )
            cols = np.array(
                ["x", "y", "z", "range", "intensity", "reflectivity", "ambient" ,"is_ground"],
                dtype="S"
            )
            dset.attrs["columns"] = cols


