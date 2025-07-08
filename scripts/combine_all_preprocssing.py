from pathlib import Path

import rclpy
from scripts.pre_processing.bird_eye_view import visualize_model_bird_eye
from scripts.pre_processing.extract_aircraft_models import extract_aircraft_models
from scripts.pre_processing.extract_tf_scene import extract_pcd_and_tf
from scripts.pre_processing.visualise_before_after_zfilter import visualize_before_after
from scripts.pre_processing.visualise_model_scene_after_tf import visualize_overlay
from scripts.pre_processing.z_value_from_tf import get_first_z_translation_from_bag

bag_path = "/home/femi/Benchmarking_framework/Data/bag_files/Air_France_25_01_2022_Air_France_aircraft_front_A319_Ceo_25-01-2022-12-21-23_fix"
TOPIC = "/main/points"
INTERVAL = 0.1
SOURCE_F = "base_link"
TARGET_F = "main_sensor_lidar"
VISUALIZE = False
EXTRACT_M = True
VIS_MODEL = True
VIS_OVERLAY = False
ALIGN_MODELS = True
VIS_OVERLAP = False
source_frame_for_z = "towbar"

if __name__ == "__main__":
    rclpy.init()

    bag_name = Path(bag_path).stem
    z_min = get_first_z_translation_from_bag(bag_path, topic=TOPIC, source_frame=source_frame_for_z,
                                             target_frame=TARGET_F, interval_sec=INTERVAL)
    extract_pcd_and_tf(
        bag_path=bag_path,
        topic=TOPIC,
        interval_sec=INTERVAL,
        z_min=z_min,
        source_frame=SOURCE_F,
        target_frame=TARGET_F,
    )

    if EXTRACT_M:
        extract_aircraft_models(bag_path)
    out_dir = Path(bag_path).parents[1] / "sequence_from_scene" / bag_name
    model_path = Path(bag_path).parents[1] / "Aircraft_models" / f"{bag_name}_model.pcd"
    scene_files = sorted(out_dir.glob(f"{bag_name}_scene????_*_filtered.pcd"))
    if VISUALIZE and scene_files:
        visualize_before_after(str(scene_files[0]).replace('_filtered', '_raw'), str(scene_files[0]))
    if VIS_MODEL:
        visualize_model_bird_eye(str(model_path))
    if VIS_OVERLAY:
        for scene in scene_files: visualize_overlay(str(scene), str(model_path))
