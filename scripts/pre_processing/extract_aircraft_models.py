from pathlib import Path

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d



def extract_aircraft_models(
    bag_path: str,
    model_topic: str = "/cloud_pcd"
) -> None:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )
    model_dir = Path(bag_path).parents[1] / "Aircraft_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    while reader.has_next():
        tname, raw, _ = reader.read_next()
        if tname != model_topic: continue
        cloud_msg = deserialize_message(raw, PointCloud2)
        pts = np.array(
            [[x, y, z] for x, y, z in pc2.read_points(
                cloud_msg, field_names=("x","y","z"), skip_nans=True
            )], dtype=np.float64
        )
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
        bag_name = Path(bag_path).stem
        filename = f"{bag_name}_model.pcd"
        pcd_path = model_dir / filename
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        print(f"Saved aircraft model: {pcd_path}")
        break
# bag_path = "/home/femi/Benchmarking_framework/Data/bag_files/HAM_Airport_2024_08_08_movement_a320_ceo_Germany"
# bag_name = Path(bag_path).stem
# extract_aircraft_models(bag_path)