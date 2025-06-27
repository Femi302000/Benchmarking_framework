
from pathlib import Path

from scripts.pre_processing.bird_eye_view import visualize_model_bird_eye
from scripts.pre_processing.extract_aircraft_models import extract_aircraft_models
from scripts.pre_processing.visualise_before_after_zfilter import visualize_before_after
from scripts.pre_processing.extract_tf_scene import extract_pcd_and_tf
import rclpy
from scripts.pre_processing.visualise_model_scene_after_tf import visualize_overlay

bag_path = "/home/femi/Benchmarking_framework/Data/bag_files/Airbus_Airbus_13_03_2024_a320neo_2024-03-13-11-37-53"
TOPICS = ["/main/points"]
INTERVAL = 0.5
SOURCE_F = "base_link"
TARGET_F = "main_sensor"
VISUALIZE = True
EXTRACT_M = True
VIS_MODEL = True
VIS_OVERLAY = True
ALIGN_MODELS = True
VIS_OVERLAP = False

if __name__ == "__main__":
    rclpy.init()

    bag_name = Path(bag_path).stem
    extract_pcd_and_tf(bag_path, TOPICS, INTERVAL, SOURCE_F, TARGET_F)
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


