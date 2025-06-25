#!/usr/bin/env python3
import pathlib
#from scripts.extract_rosbag import read_static_transform_from_ros2
from scripts.Visualise_pcd import visualize_pcd_views

if __name__ == "__main__":
    # bag_path     = "/home/femi/my_airbus_ros2_bag2"
    # cloud_topic  = "/tf_static"
    #
    # parent_frame = "base_link"
    # child_frame  = "main_sensor"
    #
    #
    # #pcd= read_first_cloud_from_ros2( bag_path, cloud_topic)
    # #transform=read_static_transform_from_ros2(bag_path,parent_frame,child_frame,qos_yaml="/home/femi/Evitado/basics/qos_override.yaml")
    # transform = read_static_transform_from_ros2(bag_path, parent_frame, child_frame)
    # # print(transform
    # visualize_pcd_views("/home/femi/Evitado/Benchmarking_framework/Data/sequence_from_scene/HAM_Bag/main_points_0008_4.097s.pcd",
    #                        z_limits=(0,25),
    #                        point_size=8, figsize=(20, 10))

    from scripts.pcd_and_transformation import extract_pcd_and_tf
    #
    bag = "/home/femi/Evitado/Benchmarking_framework/Data/Bag_Files/HAM_Bag"
    topics = ["/main/points"]
    extract_pcd_and_tf(bag, topics, interval_sec=30)
