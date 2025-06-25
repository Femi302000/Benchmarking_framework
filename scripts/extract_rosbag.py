import os
import json
import math
import numpy as np
import open3d as o3d
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
import sensor_msgs_py.point_cloud2 as pc2_ros2
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


def invert_homogeneous_matrix(M: np.ndarray) -> np.ndarray:
    R = M[0:3, 0:3]
    t = M[0:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    M_inv = np.eye(4, dtype=float)
    M_inv[0:3, 0:3] = R_inv
    M_inv[0:3, 3] = t_inv
    return M_inv


def quaternion_to_rotation_matrix(qx, qy, qz, qw: float) -> np.ndarray:
    xx, yy, zz, ww = qx*qx, qy*qy, qz*qz, qw*qw
    xy, xz, xw = qx*qy, qx*qz, qx*qw
    yz, yw, zw = qy*qz, qy*qw, qz*qw
    R = np.array([
        [ ww + xx - yy - zz,    2*(xy - zw),       2*(xz + yw) ],
        [ 2*(xy + zw),           ww - xx + yy - zz, 2*(yz - xw) ],
        [ 2*(xz - yw),           2*(yz + xw),       ww - xx - yy + zz ],
    ], dtype=float)
    return R


def transform_stamped_to_matrix(ts: TransformStamped) -> np.ndarray:
    t = ts.transform.translation
    q = ts.transform.rotation
    R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    M = np.eye(4, dtype=float)
    M[0:3, 0:3] = R
    M[0:3, 3] = [t.x, t.y, t.z]
    return M


def stamp_to_nanosec(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def find_frame_chain(
    transforms: list,
    source_frame: str,
    target_frame: str
) -> list:
    from collections import deque
    neighbors = {}
    for ts in transforms:
        parent = ts.header.frame_id
        child = ts.child_frame_id
        neighbors.setdefault(parent, []).append((child, ts, False))
        neighbors.setdefault(child, []).append((parent, ts, True))
    queue = deque([(source_frame, [])])
    visited = {source_frame}
    while queue:
        current, path = queue.popleft()
        if current == target_frame:
            return path
        for nxt, ts_msg, inv in neighbors.get(current, []):
            if nxt in visited:
                continue
            new_path = path + [(current, nxt, ts_msg, inv)]
            if nxt == target_frame:
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return None


def extract_pcd_and_tf(
    bag_path: str,
    topics: list,
    interval_sec: float,
    source_frame: str = "base_link",
    target_frame: str = "main_sensor"
) -> None:
    """
    Extract PCDs and save the corresponding TF 4×4 matrix for each saved cloud.

    :param bag_path: ROS2 bag directory
    :param topics: list of PointCloud2 topic names
    :param interval_sec: save interval in seconds
    :param source_frame: TF source frame
    :param target_frame: TF target frame
    """
    # Load all TF transforms
    reader_tf = rosbag2_py.SequentialReader()
    reader_tf.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )
    all_tf = []
    while reader_tf.has_next():
        topic, raw, _ = reader_tf.read_next()
        if topic in ['/tf', '/tf_static']:
            msg = deserialize_message(raw, TFMessage)
            all_tf.extend(msg.transforms)
    all_tf.sort(key=lambda ts: stamp_to_nanosec(ts.header.stamp))

    bag_name = Path(bag_path).stem
    project_root = Path(bag_path).parents[1]
    data_root = project_root / "Data" / "sequence_from_scene" / bag_name
    data_root.mkdir(parents=True, exist_ok=True)

    # Process each topic
    for topic in topics:
        print(f"Processing topic: {topic}")
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("", "")
        )
        start_ns = None
        next_save = 0.0
        idx = 0

        while reader.has_next():
            tname, raw, time_ns = reader.read_next()
            if tname != topic:
                continue
            if start_ns is None:
                start_ns = time_ns
            elapsed = (time_ns - start_ns) * 1e-9
            if elapsed < next_save - 1e-6:
                continue

            # Deserialize and save PCD
            cloud_msg = deserialize_message(raw, PointCloud2)
            points = [[x, y, z] for x, y, z in pc2_ros2.read_points(
                cloud_msg, field_names=("x","y","z"), skip_nans=True
            )]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=float))
            base = f"{topic.strip('/').replace('/', '_')}_{idx:04d}_{elapsed:.3f}s"
            pcd_file = data_root / f"{base}.pcd"
            o3d.io.write_point_cloud(str(pcd_file), pcd)

            # Compute TF chain at this time
            stamp_ns = stamp_to_nanosec(cloud_msg.header.stamp)
            transforms = [ts for ts in all_tf if stamp_to_nanosec(ts.header.stamp) <= stamp_ns]
            latest = {}
            for ts in transforms:
                key = (ts.header.frame_id, ts.child_frame_id)
                ns = stamp_to_nanosec(ts.header.stamp)
                if key not in latest or stamp_to_nanosec(latest[key].header.stamp) < ns:
                    latest[key] = ts
            chain = find_frame_chain(list(latest.values()), source_frame, target_frame)
            if chain is None:
                raise RuntimeError(f"No TF chain at t={stamp_ns}")

            # Multiply hops
            M = np.eye(4)
            for _, _, ts, inv in chain:
                mat = transform_stamped_to_matrix(ts)
                M = mat.T @ M if inv else M @ mat

                        # Save matrix JSON (full 4x4)
            json_file = data_root / f"{base}_tf.json"
            with open(json_file, 'w') as jf:
                json.dump({
                    "matrix": M.tolist()
                }, jf, indent=2)

            print(f"Saved: {pcd_file} and {json_file}")
            idx += 1
            next_save += interval_sec

# Example usage:
# extract_pcd_and_tf(
#     "/path/to/bag",
#     ["/main/points", "/cloud_pcd"],
#     interval_sec=0.5,
#     source_frame="base_link",
#     target_frame="main_sensor"
# )
