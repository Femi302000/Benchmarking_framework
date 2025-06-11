#!/usr/bin/env python3
import pathlib
from scripts.extract_rosbag import read_static_transform_from_ros2

if __name__ == "__main__":
    bag_path     = "/home/femi/my_airbus_ros2_bag2"
    cloud_topic  = "/tf_static"

    parent_frame = "base_link"
    child_frame  = "main_sensor"


    #pcd= read_first_cloud_from_ros2( bag_path, cloud_topic)
    #transform=read_static_transform_from_ros2(bag_path,parent_frame,child_frame,qos_yaml="/home/femi/Evitado/basics/qos_override.yaml")
    transform = read_static_transform_from_ros2(bag_path, parent_frame, child_frame)
    print(transform)
