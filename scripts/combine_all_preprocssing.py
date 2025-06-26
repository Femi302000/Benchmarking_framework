
from pathlib import Path

from scripts.pre_processing.bird_eye_view import visualize_model_bird_eye
from scripts.pre_processing.extract_aircraft_models import extract_aircraft_models
from scripts.pre_processing.visualise_before_after_zfilter import visualize_before_after
from scripts.pre_processing.extract_tf_scene import extract_pcd_and_tf
import rclpy
from scripts.pre_processing.visualise_model_scene_after_tf import visualize_overlay

# === PyCharm Run Configuration ===
BAG_PATH = "/home/femi/Benchmarking_framework/Data/bag_files/HAM_Airport_2024_08_08_movement_a320_ceo_Germany"
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

    bag_name = Path(BAG_PATH).stem
    extract_pcd_and_tf(BAG_PATH, TOPICS, INTERVAL, SOURCE_F, TARGET_F)
    if EXTRACT_M:
        extract_aircraft_models(BAG_PATH)
    out_dir = Path(BAG_PATH).parents[1] / "sequence_from_scene" / bag_name
    model_path = Path(BAG_PATH).parents[1] / "Aircraft_models" / f"{bag_name}_model.pcd"
    scene_files = sorted(out_dir.glob(f"{bag_name}_scene????_*_filtered.pcd"))
    if VISUALIZE and scene_files:
        visualize_before_after(str(scene_files[0]).replace('_filtered', '_raw'), str(scene_files[0]))
    if VIS_MODEL:
        visualize_model_bird_eye(str(model_path))
    if VIS_OVERLAY:
        for scene in scene_files: visualize_overlay(str(scene), str(model_path))


