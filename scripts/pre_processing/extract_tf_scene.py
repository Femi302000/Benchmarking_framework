#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import open3d as o3d
import h5py

from ros2_numpy.point_cloud2 import point_cloud2_to_array

import rclpy
import rosbag2_py
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosbag2_py import StorageFilter
from scripts.pre_processing.destaggering import destagger
from scripts.pre_processing.filter_ground_plane import filter_by_min_z
from scripts.pre_processing.pixel_shift import extract_pixel_shift_by_row_field
from scripts.pre_processing.save_scene import save_scene_to_hdf5
from scripts.pre_processing.tf_lookup import BagTfProcessor
from scripts.pre_processing.z_value_from_tf import get_first_z_translation_from_bag
from sensor_msgs.msg import PointCloud2
from tests.test_basic_operation import cloud2_to_array_pure


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx, yy, zz, ww = qx * qx, qy * qy, qz * qz, qw * qw
    xy, xz, xw = qx * qy, qx * qz, qx * qw
    yz, yw, zw = qy * qz, qy * qw, qz * qw
    return np.array([
        [ww + xx - yy - zz, 2 * (xy - zw), 2 * (xz + yw)],
        [2 * (xy + zw), ww - xx + yy - zz, 2 * (yz - xw)],
        [2 * (xz - yw), 2 * (yz + xw), ww - xx - yy + zz],
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
        topic: str,
        interval_sec: float,
        z_min: float,
        source_frame: str = "base_link",
        target_frame: str = "main_sensor_lidar",
        target_frame_gd=str,
        source_frame_gd=str):
    """
    Extract pointcloud scenes and corresponding TF matrices from a ROS2 bag,
    saving range, intensity, reflectivity, ambient, and filtered 3D points.
    """
    # 1) load TF buffer
    tf_processor = BagTfProcessor()
    tf_processor.read_tf_from_bag(bag_path)

    # 2) prepare output directory
    bag_name = Path(bag_path).stem
    out_dir = Path(bag_path).parents[1] / "machine_learning_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) open rosbag reader and filter by topic
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )
    reader.set_filter(StorageFilter(topics=[topic]))

    start_ns = None
    next_save = 0.0
    idx = 0
    pixel_shifts = extract_pixel_shift_by_row_field(bag_path)
    while reader.has_next():
        _, raw_buf, time_ns = reader.read_next()

        if start_ns is None:
            start_ns = time_ns
        elapsed = (time_ns - start_ns) * 1e-9
        if elapsed + 1e-6 < next_save:
            continue

        # deserialize PointCloud2
        cloud_msg: PointCloud2 = deserialize_message(raw_buf, PointCloud2)
        stamp_ns = stamp_to_nanosec(cloud_msg.header.stamp)
        ros_time = Time(nanoseconds=stamp_ns)
        # cloud_2d=cloud2_to_array(cloud_msg)
        # PIX_SHIFT=extract_pixel_shift_by_row_field(bag_path)
        # dst = destagger_all_fields(cloud_2d,PIX_SHIFT)

        # get TF (skip if unavailable)
        if not tf_processor.tf_buffer.can_transform(
                target_frame, source_frame, ros_time, timeout=Duration(seconds=0.0)
        ):
            next_save += interval_sec
            continue

        try:
            tf_stamped = tf_processor.tf_buffer.lookup_transform(
                target_frame, source_frame, ros_time
            )
        except Exception:
            next_save += interval_sec
            continue

        # read points and separate fields
        points_iter = pc2.read_points(
            cloud_msg,
            field_names=('x', 'y', 'z', 'range', 'intensity', 'reflectivity', 'ambient'),
            skip_nans=True
        )
        RANGE_SCALE = 1e-3  # e.g. if 'range' is stored in millimeters
        REFLECTIVITY_SCALE = 1.0  # if 0–255 unitless
        AMBIENT_SCALE = 1.0

        cloud_2d = cloud2_to_array_pure(cloud_msg)
        range_raw = cloud_2d['range'].astype(np.float32) * RANGE_SCALE
        intensity_raw = cloud_2d['intensity'].astype(np.float32)
        reflect_raw = cloud_2d['reflectivity'].astype(np.float32)* REFLECTIVITY_SCALE
        ambient_raw = cloud_2d['ambient'].astype(np.float32)* AMBIENT_SCALE
        ring_raw = cloud_2d['ring'].astype(np.uint16)
        if pixel_shifts is not None:
            range_image = destagger(range_raw, pixel_shifts)
            intensity_image = destagger(intensity_raw, pixel_shifts)
            reflect_image = destagger(reflect_raw, pixel_shifts)
            ambient_image = destagger(ambient_raw, pixel_shifts)
            ring_image = destagger(ring_raw, pixel_shifts)

        else:
            range_image = np.array(range_raw, dtype=np.float32)
            intensity_image = np.array(intensity_raw, dtype=np.float32)
            reflect_image = np.array(reflect_raw, dtype=np.float32)
            ambient_image = np.array(ambient_raw, dtype=np.float32)
            ring_image = ring_raw

        xs, ys, zs, ranges, intensities, reflects, ambients = zip(*points_iter)
        pts = np.vstack((xs, ys, zs)).T.astype(np.float64)


        # apply TF to points if needed (optional)
        raw_pcd = o3d.geometry.PointCloud()
        raw_pcd.points = o3d.utility.Vector3dVector(pts)

        # filter ground
        filtered_pcd, non_ground_mask = filter_by_min_z(raw_pcd, z_min)
        ground_mask = 1 - non_ground_mask

        # get transform matrix
        M = transform_stamped_to_matrix(tf_stamped)
        save_scene_to_hdf5(
            bag_path=bag_path,
            base_out_dir=out_dir,
            stamp_ns=stamp_ns,
            raw_pcd=raw_pcd,
            tf_matrix=M,
            range_image=range_image,
            intensity_image=intensity_image,
            reflectivity_image= reflect_image,
            ambient_image=ambient_image,

            ring_arr=ring_raw,
            ground_mask=ground_mask,
        )

        idx += 1
        next_save += interval_sec


if __name__ == "__main__":
    rclpy.init()

    bag_path = "/home/femi/Benchmarking_framework/Data/bag_files/HAM_Airport_2024_08_08_movement_a320_ceo_Germany"
    topic = "/main/points"
    interval_sec = 0.5

    source_frame = "base_link"
    target_frame = "main_sensor_lidar"
    source_frame_gd="towbar"
    target_frame_gd="main_sensor_lidar"
    z_min = get_first_z_translation_from_bag(bag_path, topic=topic, source_frame=source_frame_gd,
                                             target_frame=target_frame_gd, interval_sec=interval_sec)
    extract_pcd_and_tf(
        bag_path=bag_path,
        topic=topic,
        interval_sec=interval_sec,
        z_min=z_min,
        source_frame=source_frame,
        target_frame=target_frame,
    )

    rclpy.shutdown()
