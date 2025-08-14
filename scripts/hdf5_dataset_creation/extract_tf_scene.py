#!/usr/bin/env python3
# Single-pass extractor that saves scenes at a fixed header-time interval
# and (optionally) saves the /cloud_pcd model in the SAME pass.
# Uses a header-stamp gate so it does NOT save continuously.

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import open3d as o3d

import rosbag2_py
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosbag2_py import StorageFilter
from sensor_msgs.msg import PointCloud2

# --- your project helpers (adjust import paths if needed) ---
from scripts.hdf5_dataset_creation.helper_functions.destaggering import destagger
from scripts.hdf5_dataset_creation.helper_functions.filter_ground_plane import filter_by_min_z
from scripts.hdf5_dataset_creation.helper_functions.save_scene import save_scene_to_hdf5
from scripts.hdf5_dataset_creation.helper_functions.tf_lookup import BagTfProcessor


# =========================
# Math / TF utilities
# =========================
def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx, yy, zz, ww = qx*qx, qy*qy, qz*qz, qw*qw
    xy, xz, xw = qx*qy, qx*qz, qx*qw
    yz, yw, zw = qy*qz, qy*qw, qz*qw
    return np.array([
        [ww + xx - yy - zz, 2*(xy - zw),       2*(xz + yw)],
        [2*(xy + zw),       ww - xx + yy - zz, 2*(yz - xw)],
        [2*(xz - yw),       2*(yz + xw),       ww - xx - yy + zz],
    ], dtype=float)


def transform_stamped_to_matrix(ts: TransformStamped) -> np.ndarray:
    t = ts.transform.translation
    q = ts.transform.rotation
    R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    M = np.eye(4, dtype=float)
    M[:3, :3] = R
    M[:3, 3] = [t.x, t.y, t.z]
    return M


def stamp_to_nanosec(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


# =========================
# Model extraction (NO extra bag open)
# =========================
def extract_and_save_aircraft_model_from_msg(
    topic_name: str,
    raw_buf: bytes,
    bag_path: str,
    model_topic: str = "/cloud_pcd",
    out_dir_name: str = "Aircraft_models",
    out_filename: Optional[str] = None,     # default: <bag_stem>_model.pcd
    overwrite: bool = False,
    voxel_size: Optional[float] = None,     # e.g., 0.02 = 2 cm
    color_rgb: Optional[Sequence[float]] = (1.0, 1.0, 1.0),
    transform_4x4: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """Handle ONE message; if it's `model_topic`, save a PCD and return its path."""
    if topic_name != model_topic:
        return None

    cloud_msg: PointCloud2 = deserialize_message(raw_buf, PointCloud2)
    pts = np.array(
        [[x, y, z] for x, y, z in pc2.read_points(
            cloud_msg, field_names=("x", "y", "z"), skip_nans=True
        )],
        dtype=np.float64,
    )
    if pts.size == 0:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if transform_4x4 is not None:
        if not (isinstance(transform_4x4, np.ndarray) and transform_4x4.shape == (4, 4)):
            raise ValueError("transform_4x4 must be a 4×4 numpy array.")
        pcd.transform(transform_4x4)

    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    if color_rgb is not None:
        if len(color_rgb) != 3:
            raise ValueError("color_rgb must be length 3.")
        pcd.paint_uniform_color([float(c) for c in color_rgb])

    model_dir = Path(bag_path).parents[1] / out_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)
    if out_filename is None:
        out_filename = f"{Path(bag_path).stem}_model.pcd"
    out_path = model_dir / out_filename

    if out_path.exists() and not overwrite:
        # already saved previously
        return out_path

    if not o3d.io.write_point_cloud(str(out_path), pcd):
        return None

    print(f"[ok] Saved aircraft model: {out_path}")
    return out_path


# =========================
# Core extractor (single reader for points + model, lazy z_min, header-time sampling)
# =========================
def extract_pcd_and_tf_single_open(
    bag_path: str,
    topic_points: str,
    interval_sec: float,
    source_frame: str,
    target_frame: str,
    tf_processor: BagTfProcessor,
    out_subdir: str = "machine_learning_dataset",
    aircraft_model: Optional[str] = None,
    pixel_shifts: Optional[np.ndarray] = None,    # provide array to enable destagger
    # z_min lazy reference frames:
    z_ref_source_frame: str = "towbar",
    z_ref_target_frame: str = "main_sensor_lidar",
    initial_z_min: Optional[float] = None,
    # model topic handling (no extra opens):
    model_topic: str = "/cloud_pcd",
    save_model: bool = True,
    model_voxel: Optional[float] = None,
    model_color: Optional[Sequence[float]] = (0.85, 0.85, 0.85),
    model_overwrite: bool = False,
) -> None:
    """
    Opens the bag ONCE for data and handles both `topic_points` and `model_topic`.
    TF must be preloaded via `tf_processor.read_tf_from_bag` (separate TF pass).

    Uses a HEADER-TIME sampling gate so frames are saved only when
    cloud_msg.header.stamp crosses the next threshold.
    """
    # Output dir: <bag_parent>/machine_learning_dataset
    out_dir = Path(bag_path).parents[1] / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single data reader (points + model)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )
    topics_filter = [topic_points]
    if save_model and model_topic:
        topics_filter.append(model_topic)
    reader.set_filter(StorageFilter(topics=topics_filter))

    # Sampling by header stamp
    next_save_stamp_ns: Optional[int] = None
    interval_ns = int(interval_sec * 1e9)

    # Lazy z_min
    z_min = initial_z_min

    # Save aircraft model only once
    model_saved = False


    RANGE_SCALE = 1e-3

    while reader.has_next():
        tname, raw_buf, _ = reader.read_next()

        # --- Try saving the model in this SAME pass (no extra open) ---
        if save_model and not model_saved and tname == model_topic:
            maybe = extract_and_save_aircraft_model_from_msg(
                topic_name=tname,
                raw_buf=raw_buf,
                bag_path=bag_path,
                model_topic=model_topic,
                voxel_size=model_voxel,
                color_rgb=model_color,
                overwrite=model_overwrite,
            )
            if maybe is not None:
                model_saved = True
            if tname != topic_points:
                continue  # proceed to next message

        if tname != topic_points:
            continue

        # ---- Deserialize the point cloud & compute times ----
        cloud_msg: PointCloud2 = deserialize_message(raw_buf, PointCloud2)
        stamp_ns = stamp_to_nanosec(cloud_msg.header.stamp)   # HEADER time (sensor time)
        ros_time = Time(nanoseconds=stamp_ns)
        H, W = int(cloud_msg.height), int(cloud_msg.width)

        # ---- Lazily compute z_min from TF (z_ref_target <- z_ref_source) ----
        if z_min is None:
            if tf_processor.tf_buffer.can_transform(
                z_ref_target_frame, z_ref_source_frame, ros_time, Duration(seconds=0.0)
            ):
                try:
                    tf_z = tf_processor.tf_buffer.lookup_transform(
                        z_ref_target_frame, z_ref_source_frame, ros_time
                    )
                    z_min = float(np.round(tf_z.transform.translation.z, 2))
                    print(f"[lazy] z_min = {z_min}")
                except Exception:
                    pass
            if z_min is None:
                # We can't build a ground mask yet; wait for a time where TF is available.
                continue

        # ---------- HEADER-TIME SAMPLING GATE ----------
        if next_save_stamp_ns is None:
            # First eligible frame: save now and schedule the next boundary
            should_save = True
            next_save_stamp_ns = stamp_ns + interval_ns
        else:
            should_save = (stamp_ns >= next_save_stamp_ns)

        if not should_save:
            # Not time yet; skip heavy work
            # print(f"[skip] {stamp_ns} < {next_save_stamp_ns}")
            continue
        # -----------------------------------------------

        # --- scene TF (target_frame <- source_frame) ---
        if not tf_processor.tf_buffer.can_transform(
            target_frame, source_frame, ros_time, Duration(seconds=0.0)
        ):
            # don't advance the save gate; try again on a later frame
            continue
        try:
            tf_stamped = tf_processor.tf_buffer.lookup_transform(
                target_frame, source_frame, ros_time
            )
        except Exception:
            # also keep the same gate so we can try the next frame
            continue

        # --- Build arrays (no visuals) ---
        pts_iter = pc2.read_points(
            cloud_msg,
            field_names=('x','y','z','range','intensity','reflectivity','ambient'),
            skip_nans=True
        )
        xs, ys, zs, r_raw, i_raw, rf_raw, a_raw = zip(*pts_iter)
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        zs = np.asarray(zs, dtype=np.float64)
        ranges      = np.asarray(r_raw, dtype=np.float32) * RANGE_SCALE
        intensities = np.asarray(i_raw, dtype=np.float32)
        reflects    = np.asarray(rf_raw, dtype=np.float32)
        ambients    = np.asarray(a_raw, dtype=np.float32)

        if pixel_shifts is not None:
            try:
                ranges      = destagger(ranges.reshape(H, W), pixel_shifts).ravel()
                intensities = destagger(intensities.reshape(H, W), pixel_shifts).ravel()
                reflects    = destagger(reflects.reshape(H, W), pixel_shifts).ravel()
                ambients    = destagger(ambients.reshape(H, W), pixel_shifts).ravel()
            except Exception as e:
                print(f"[warn] destagger skipped: {e}")

        # Ground mask via z_min
        pts_xyz = np.column_stack((xs, ys, zs))
        raw_pcd = o3d.geometry.PointCloud()
        raw_pcd.points = o3d.utility.Vector3dVector(pts_xyz)
        mask_keep = filter_by_min_z(raw_pcd, z_min).reshape(-1, 1).astype(np.float32)  # (N,1)

        # Final (N×8): xyz + 4 channels + is_ground
        points_all = np.column_stack([
            pts_xyz.astype(np.float32),
            ranges, intensities, reflects, ambients,
            mask_keep
        ])

        # Scene transform matrix
        M = transform_stamped_to_matrix(tf_stamped)

        # Save one scene (append to per-bag HDF5)
        save_scene_to_hdf5(
            bag_path=bag_path,
            base_out_dir=out_dir,
            stamp_ns=stamp_ns,
            tf_matrix=M,
            points_all=points_all,
            aircraft_model=aircraft_model,
            height=H,
            width=W,
        )

        # advance the next save boundary ONLY after a successful save
        next_save_stamp_ns += interval_ns

        # free temporaries
        del xs, ys, zs, ranges, intensities, reflects, ambients, pts_xyz, raw_pcd, points_all
