import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosbag2_py import StorageOptions, ConverterOptions, SequentialReader, StorageFilter
from scripts.pre_processing.tf_lookup import BagTfProcessor
from sensor_msgs.msg import PointCloud2

def stamp_to_nanosec(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
def get_first_z_translation_from_bag(
        bag_path: str,
        topic: str,
        source_frame: str,
        target_frame: str,
        interval_sec: float
) -> tuple[float, float] | None:
    """
    Read PointCloud2 messages from `topic` in the given ROS2 bag,
    sample at `interval_sec` seconds, look up the transform from source_frame
    to target_frame at each sample, and return the first (time_sec, z_translation).
    If no transform is found, returns None.
    """
    # Initialize TF lookup
    tf_processor = BagTfProcessor()
    tf_processor.read_tf_from_bag(bag_path)

    # Open the bag reader
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id="sqlite3"),
        ConverterOptions("", "")
    )

    reader.set_filter(StorageFilter(topics=[topic]))

    start_ns = None
    next_save = 0.0

    while reader.has_next():
        _, raw_buffer, time_ns = reader.read_next()
        if start_ns is None:
            start_ns = time_ns
        elapsed = (time_ns - start_ns) * 1e-9
        if elapsed + 1e-6 < next_save:
            continue

        # Deserialize to get the header stamp
        cloud_msg: PointCloud2 = deserialize_message(raw_buffer, PointCloud2)
        t_ns = stamp_to_nanosec(cloud_msg.header.stamp)
        ros_time = Time(nanoseconds=t_ns)

        # Fast availability check; returns False instead of throwing
        if not tf_processor.tf_buffer.can_transform(
                target_frame,
                source_frame,
                ros_time,
                timeout=Duration(seconds=0.0)
        ):
            next_save += interval_sec
            continue

        # Safe to lookup now
        try:
            tf_stamped = tf_processor.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                ros_time
            )
        except Exception:
            # only truly unexpected errors end up here
            next_save += interval_sec
            continue

        # Extract the z-component of the translation
        z = float(np.round(tf_stamped.transform.translation.z, 2))
        return z

        # advance to the next sampling time
        next_save += interval_sec

    return None


if __name__ == "__main__":
    rclpy.init()

    bag_path = "/home/femi/Benchmarking_framework/Data/bag_files/Air_France_25_01_2022_Air_France_aircraft_front_A319_Ceo_25-01-2022-12-21-23_fix"
    topic = "/main/points"
    source_frame = "plane_front_left_wheel_link"
    target_frame = 'main_sensor'
    interval_sec = 0.1

    result = get_first_z_translation_from_bag(
        bag_path=bag_path,
        topic=topic,
        source_frame=source_frame,
        target_frame=target_frame,
        interval_sec=interval_sec
    )

    if result:
        print(f"First sample at  z = {result}")
    else:
        print("No transform available at any sampled timestamp.")

    rclpy.shutdown()
