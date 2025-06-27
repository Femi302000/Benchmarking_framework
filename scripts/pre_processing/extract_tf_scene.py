import json
from pathlib import Path

import numpy as np

import rclpy
import rosbag2_py
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosbag2_py import StorageFilter
from scripts.pre_processing.tf_lookup import BagTfProcessor
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
from scripts.pre_processing.filter_ground_plane import filter_negative_z


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx, yy, zz, ww = qx*qx, qy*qy, qz*qz, qw*qw
    xy, xz, xw = qx*qy, qx*qz, qx*qw
    yz, yw, zw = qy*qz, qy*qw, qz*qw
    return np.array([
        [ ww + xx - yy - zz, 2*(xy - zw),       2*(xz + yw) ],
        [ 2*(xy + zw),       ww - xx + yy - zz, 2*(yz - xw) ],
        [ 2*(xz - yw),       2*(yz + xw),       ww - xx - yy + zz ],
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


def extract_pcd_and_tf(
    bag_path: str,
    topics: list,
    interval_sec: float,
    source_frame: str = "base_link",
    target_frame: str = "main_sensor"
) -> None:
    tf_processor = BagTfProcessor()
    tf_processor.read_tf_from_bag(bag_path)

    bag_name = Path(bag_path).stem
    out_dir = Path(bag_path).parents[1] / "sequence_from_scene" / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for topic in topics:
        print(f"Processing topic: {topic}")
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("", "")
        )
        reader.set_filter(StorageFilter(topics=[topic]))

        start_ns = None
        next_save = 0.0
        idx = 0

        while reader.has_next():
            tname, raw, time_ns = reader.read_next()
            if start_ns is None:
                start_ns = time_ns
            elapsed = (time_ns - start_ns) * 1e-9
            if elapsed + 1e-6 < next_save:
                continue

            # Deserialize point cloud message
            cloud_msg: PointCloud2 = deserialize_message(raw, PointCloud2)
            # Lookup transform at this timestamp
            stamp_ns = stamp_to_nanosec(cloud_msg.header.stamp)
            ros_time = Time(nanoseconds=stamp_ns)
            tf_stamped = tf_processor.lookup_example(
                source_frame,target_frame,  ros_time
            )

            if not tf_stamped:
                print(f"[SKIP] no TF for `{source_frame}`→`{target_frame}` at t={elapsed:.3f}s")
                continue

            # Only save scenes with an available TF
            pts = np.array(
                [[x, y, z] for x, y, z in pc2.read_points(
                    cloud_msg, field_names=("x", "y", "z"), skip_nans=True
                )],
                dtype=np.float64
            )
            raw_pcd = o3d.geometry.PointCloud()
            raw_pcd.points = o3d.utility.Vector3dVector(pts)
            filtered_pcd = filter_negative_z(raw_pcd)

            name = f"{bag_name}_scene{idx:04d}_{elapsed:.3f}s"
            raw_path = out_dir / f"{name}_raw.pcd"
            filt_path = out_dir / f"{name}_filtered.pcd"
            o3d.io.write_point_cloud(str(raw_path), raw_pcd)
            o3d.io.write_point_cloud(str(filt_path), filtered_pcd)

            # Save transform matrix
            M = transform_stamped_to_matrix(tf_stamped)
            M = np.round(M, 3)
            tf_path = out_dir / f"{name}_tf.json"
            with open(tf_path, 'w') as f:
                json.dump({"matrix": M.tolist()}, f, indent=2)

            print(f"Saved scene#{idx}: {raw_path.name}, {filt_path.name}, {tf_path.name}")

            idx += 1
            next_save += interval_sec

# bag_path = "/home/femi/Benchmarking_framework/Data/bag_files/HAM_Airport_2024_08_08_movement_a320_ceo_Germany"
# TOPICS = ["/main/points"]
# INTERVAL = 0.5
# SOURCE_F = "base_link"
# TARGET_F = "main_sensor"
# rclpy.init()
#
# extract_pcd_and_tf(bag_path, TOPICS, INTERVAL, SOURCE_F, TARGET_F)

