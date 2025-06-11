#!/usr/bin/env python3
import sys
print("Python executable:", sys.executable)
print("sys.path:", sys.path)
import transforms3d





import os
import numpy as np
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2_ros2
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import PointCloud2


import os
from pathlib import Path
import numpy as np
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2_ros2
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

def read_first_cloud_from_ros2(bag_path: str, topic: str) -> o3d.geometry.PointCloud:
    """
    Reads the very first PointCloud2 on `topic` in a ROS2 bag, converts it to an Open3D PointCloud,
    and saves it as a .pcd file:
      - If topic == 'cloud_pcd', saves to data/aircraft_model/{bag_name}.pcd
      - If topic == '/main/points', saves to data/scene_raw/{bag_name}.pcd
    """
    # Set up the reader
    reader = SequentialReader()
    storage = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader.open(storage, converter)

    # Find first message on the desired topic
    first_msg_data = None
    while reader.has_next():
        topic_name, data, _ = reader.read_next()
        if topic_name == topic:
            first_msg_data = data
            break

    if first_msg_data is None:
        raise RuntimeError(f"No messages found on topic '{topic}' in bag '{bag_path}'.")

    # Deserialize to a ROS2 PointCloud2
    cloud_msg = deserialize_message(first_msg_data, PointCloud2)
    print(f":: Selected the first cloud message on '{topic}'.")

    # Convert points to a NumPy array
    pts = [
        [float(x), float(y), float(z)]
        for x, y, z in pc2_ros2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    ]
    points = np.asarray(pts, dtype=np.float64)

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    script_dir = Path(__file__).resolve().parent  # …/Benchmarking_framework/scripts
    project_dir = script_dir.parent  # …/Benchmarking_framework
    data_root = project_dir / "Data"  # …/Benchmarking_framework/Data

    bag_name = Path(bag_path).stem
    if topic == "/cloud_pcd" :
        # match your actual folder name on disk
        out_dir = data_root / "Aircraft_models"
    elif topic == "/main/points":
        out_dir = data_root / "scene_raw"
    else :
        raise RuntimeError(f"Unrecognized topic: '{topic}' – cannot determine save location.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bag_name}.pcd"

    # Save the point cloud
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f":: Saved point cloud to '{out_path}'")

    return pcd



import os
from pathlib import Path

import numpy as np
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2_ros2
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

def read_and_save_clouds_as_pcd(
    bag_path: str,
    topic: str,
    interval_sec: float
) -> None:
    """
    Reads PointCloud2 messages on `topic` from a ROS2 bag and
    saves each cloud at `interval_sec` intervals into:
      Data/scene_raw/sequence_from/<bag_name>/cloud_####_<t>s.pcd
    """
    # resolve your project’s Data folder
    script_dir  = Path(__file__).resolve().parent       # …/Benchmarking_framework/scripts
    project_dir = script_dir.parent                     # …/Benchmarking_framework
    data_root   = project_dir / "Data"

    bag_name = Path(bag_path).stem
    base_dir = data_root / "sequence_from_scene" / bag_name
    os.makedirs(base_dir, exist_ok=True)

    # open the bag
    reader = SequentialReader()
    storage_opts = StorageOptions(uri=bag_path, storage_id="sqlite3")
    conv_opts    = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader.open(storage_opts, conv_opts)

    start_ns       = None
    next_save_time = 0.0
    saved_count    = 0

    # iterate and save at interval_sec
    while reader.has_next():
        topic_name, data, time_ns = reader.read_next()
        if topic_name != topic:
            continue

        if start_ns is None:
            start_ns = time_ns

        elapsed = (time_ns - start_ns) * 1e-9
        if elapsed + 1e-6 >= next_save_time:
            # deserialize & convert
            cloud_msg = deserialize_message(data, PointCloud2)
            pts = [
                [float(x), float(y), float(z)]
                for x, y, z in pc2_ros2.read_points(
                    cloud_msg, field_names=("x","y","z"), skip_nans=True
                )
            ]
            points = np.asarray(pts, dtype=np.float64)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # write PCD
            fn = base_dir / f"cloud_{saved_count:04d}_{elapsed:.3f}s.pcd"
            o3d.io.write_point_cloud(str(fn), pcd)
            print(f"Saved #{saved_count:02d} at t={elapsed:.3f}s → {fn}")

            saved_count   += 1
            next_save_time += interval_sec

    if saved_count == 0:
        raise RuntimeError(f"No clouds saved: no messages on '{topic}'?")
    print(f"Done: {saved_count} clouds written to '{base_dir}'")


#!/usr/bin/env python3
import sys
import os
import time
import threading
from typing import Tuple, List
import numpy as np
import transforms3d
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from tf2_msgs.msg import TFMessage

class TransformBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._transforms: dict[Tuple[str, str], Tuple[np.ndarray, float]] = {}

    @staticmethod
    def _make_matrix(translation: Tuple[float, float, float], quaternion: Tuple[float, float, float, float]) -> np.ndarray:
        tx, ty, tz = translation
        qx, qy, qz, qw = quaternion
        R = transforms3d.quaternions.quat2mat([qw, qx, qy, qz])
        M = np.eye(4, dtype=float)
        M[:3, :3] = R
        M[:3, 3] = [tx, ty, tz]
        return M

    @staticmethod
    def _decompose_matrix(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = M[:3, 3]
        R = M[:3, :3]
        qw, qx, qy, qz = transforms3d.quaternions.mat2quat(R)
        return t, np.array([qx, qy, qz, qw])

    def add_transform(self, parent_frame: str, child_frame: str, translation: Tuple[float, float, float], quaternion: Tuple[float, float, float, float], timestamp: float) -> None:
        M = self._make_matrix(translation, quaternion)
        key = (parent_frame, child_frame)
        with self._lock:
            self._transforms[key] = (M, timestamp)

    def _get_neighbors(self, frame: str) -> List[Tuple[str, np.ndarray, float]]:
        neighbors = []
        with self._lock:
            for (p, c), (M, ts) in self._transforms.items():
                if p == frame:
                    neighbors.append((c, M, ts))
                elif c == frame:
                    M_inv = np.linalg.inv(M)
                    neighbors.append((p, M_inv, ts))
        return neighbors

    def lookup_transform(self, source_frame: str, target_frame: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float], float]:
        if source_frame == target_frame:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), time.time()

        visited = {source_frame}
        queue: list[tuple[str, list[tuple[str, np.ndarray, float]]]] = [(source_frame, [])]

        while queue:
            curr, path = queue.pop(0)
            for (nbr, M_edge, ts_edge) in self._get_neighbors(curr):
                if nbr in visited:
                    continue
                visited.add(nbr)
                new_path = path + [(nbr, M_edge, ts_edge)]
                if nbr == target_frame:
                    M_composed = np.eye(4)
                    ts_latest = 0.0
                    for (_, M_step, ts_step) in new_path:
                        M_composed = M_composed @ M_step
                        ts_latest = max(ts_latest, ts_step)
                    t_final, quat_final = self._decompose_matrix(M_composed)
                    return (float(t_final[0]), float(t_final[1]), float(t_final[2])), (float(quat_final[0]), float(quat_final[1]), float(quat_final[2]), float(quat_final[3])), ts_latest
                else:
                    queue.append((nbr, new_path))

        raise KeyError(f"No TF chain between '{source_frame}' and '{target_frame}'.")

    def all_frames(self) -> list[str]:
        with self._lock:
            keys = list(self._transforms.keys())
        frames = set()
        for (p, c) in keys:
            frames.add(p)
            frames.add(c)
        return sorted(frames)

def load_rosbag_to_tfbuffer(bag_path: str, tf_buffer: TransformBuffer) -> None:
    reader = SequentialReader()
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_opts = ConverterOptions('', '')
    reader.open(storage_opts, converter_opts)
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in ('/tf', '/tf_static'):
            continue
        bag_msg: TFMessage = deserialize_message(data, 'tf2_msgs/msg/TFMessage')
        for tf_stamped in bag_msg.transforms:
            parent = tf_stamped.header.frame_id
            child = tf_stamped.child_frame_id
            tr = tf_stamped.transform.translation
            rot = tf_stamped.transform.rotation
            translation = (tr.x, tr.y, tr.z)
            quaternion = (rot.x, rot.y, rot.z, rot.w)
            ts = tf_stamped.header.stamp.sec + tf_stamped.header.stamp.nanosec * 1e-9
            tf_buffer.add_transform(parent, child, translation, quaternion, ts)

def main():
    if len(sys.argv) < 2:
        bag_folder = input("Enter path to your rosbag2 directory: ").strip()
    else:
        bag_folder = sys.argv[1]

    if not os.path.isdir(bag_folder):
        print(f"Error: '{bag_folder}' is not a directory or cannot be found.")
        sys.exit(1)

    rclpy.init()
    tf_buffer = TransformBuffer()
    load_rosbag_to_tfbuffer(bag_folder, tf_buffer)

    frames = tf_buffer.all_frames()
    print("\nAll frames found in TF graph:")
    for f in frames:
        print("  -", f)

    examples = [
        ("map", "base_link"),
        ("odom", "camera_link"),
        ("camera_link", "map"),
    ]
    print("\nExample lookups:")
    for (src, tgt) in examples:
        try:
            (tx, ty, tz), (qx, qy, qz, qw), ts = tf_buffer.lookup_transform(src, tgt)
            print(f"  {src} → {tgt}:")
            print(f"    translation = (x={tx:.3f}, y={ty:.3f}, z={tz:.3f})")
            print(f"    quaternion  = (x={qx:.4f}, y={qy:.4f}, z={qz:.4f}, w={qw:.4f})")
            print(f"    latest-edge timestamp = {ts:.6f} sec")
        except KeyError:
            print(f"  {src} → {tgt}:  <no connection>")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
