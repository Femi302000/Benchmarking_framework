import json
import numpy as np
import h5py
from pathlib import Path
from open3d import geometry

def save_scene_to_hdf5(
     bag_path: str,
     base_out_dir: Path,
     stamp_ns: int,
     raw_pcd: geometry.PointCloud,
     tf_matrix: np.ndarray,
     *,  # force all following args to be keyword-only
     range_image: np.ndarray       = None,
     intensity_image: np.ndarray   = None,
     reflectivity_image: np.ndarray= None,
     ambient_image: np.ndarray     = None,
     range_arr: np.ndarray         = None,
     intensity_arr: np.ndarray     = None,
     reflectivity_arr: np.ndarray  = None,
     ambient_arr: np.ndarray       = None,
     ring_arr: np.ndarray          = None,
     labels_arr: np.ndarray        = None,
     ground_mask: np.ndarray       = None,
 ):
     """
     Saves:
       • tf_matrix
       • optional 2D/3D arrays
       • points_ground → Nx6 [x,y,z,is_ground,ring,label]
       • attaches `column_names`=['x','y','z','is_ground','ring','labels'] to the dataset
     """
     bag_name = Path(bag_path).stem
     out_file = base_out_dir / f"{bag_name}.h5"
     with h5py.File(out_file, "a") as f:
         grp = f.require_group(str(stamp_ns))
         # clear old datasets
         for ds in list(grp.keys()):
             del grp[ds]

         # 1) tf_matrix
         grp.create_dataset("tf_matrix", data=tf_matrix)

         # 2) any 3D / 2D fields…
         if range_arr is not None:
             grp.create_dataset("range_3d", data=range_arr, compression="gzip")
         if intensity_arr is not None:
             grp.create_dataset("intensity_3d", data=intensity_arr, compression="gzip")
         if reflectivity_arr is not None:
             grp.create_dataset("reflectivity_3d", data=reflectivity_arr, compression="gzip")
         if ambient_arr is not None:
             grp.create_dataset("ambient_3d", data=ambient_arr, compression="gzip")
         if range_image is not None:
             grp.create_dataset("range_image", data=range_image, compression="gzip")
         if intensity_image is not None:
             grp.create_dataset("intensity_image", data=intensity_image, compression="gzip")
         if reflectivity_image is not None:
             grp.create_dataset("reflectivity_image", data=reflectivity_image, compression="gzip")
         if ambient_image is not None:
             grp.create_dataset("ambient_image", data=ambient_image, compression="gzip")

         # 3) pack raw points + mask + ring + labels into one Nx6 array
         if raw_pcd is not None:
             pts = np.asarray(raw_pcd.points, dtype=np.float64)  # (N,3)
             mask = (
                 ground_mask.reshape(-1,1).astype(np.uint8)
                 if ground_mask is not None
                 else np.zeros((pts.shape[0],1), dtype=np.uint8)
             )
             ring_col = (
                 ring_arr.reshape(-1,1).astype(np.uint16)
                 if ring_arr is not None
                 else np.zeros((pts.shape[0],1), dtype=np.uint16)
             )
             label_col = (
                 labels_arr.reshape(-1,1).astype(np.uint8)
                 if labels_arr is not None
                 else np.zeros((pts.shape[0],1), dtype=np.uint8)
             )

             pts_with_mask_ring_label = np.hstack([pts, mask, ring_col, label_col])  # (N,6)

             # create dataset
             ds = grp.create_dataset(
                 "points_ground",
                 data=pts_with_mask_ring_label,
                 compression="gzip"
             )

             # attach column names
             ds.attrs['column_names'] = np.array(
                 ['x', 'y', 'z', 'is_ground', 'ring', 'labels'],
                 dtype='S'
             )

         # 4) metadata attrs
         attrs = {
             "stamp_ns": stamp_ns,
             "num_points": int(pts_with_mask_ring_label.shape[0]) if raw_pcd is not None else 0,
         }
         if ground_mask is not None:
             attrs["num_ground_points"] = int(ground_mask.sum())
         grp.attrs.update(attrs)

     print(f"Saved scene at {stamp_ns} to {out_file.name}")
