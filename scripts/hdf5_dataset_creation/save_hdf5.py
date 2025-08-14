from pathlib import Path

import rclpy
from scripts.hdf5_dataset_creation.extract_points_h5_file import process_hdf5_and_visualize_alignment
from scripts.hdf5_dataset_creation.extract_tf_scene import extract_pcd_and_tf_single_open
from scripts.hdf5_dataset_creation.helper_functions.pixel_shift import extract_pixel_shift_by_row_field
from scripts.hdf5_dataset_creation.helper_functions.tf_lookup import BagTfProcessor
from scripts.hdf5_dataset_creation.label_pcd_result_hd5 import visualize_all_views

if __name__ == "__main__":
    rclpy.init()
    visualise = False

    # ---- config ----
    bag_path = "/home/femi/Benchmarking_framework/Data/bag_files/HAM_Airport_2024_08_08_movement_a320_ceo_Germany"
    topic_points = "/main/points"
    source_frame = "base_link"
        # --- lazily compute z_min from TF (z_ref_target <- z_ref_source) ---
    target_frame = "main_sensor_lidar"
    interval_sec = 500
    aircraft_model = "A320"
    destagger=False


    if destagger:
        pixel_shifts = extract_pixel_shift_by_row_field(bag_path,topic="/main/ouster_info")
    else:
        pixel_shifts =None

    # ---- TF pass (OPEN #1 for /tf + /tf_static) ----
    tfp = BagTfProcessor()
    tfp.read_tf_from_bag(bag_path)

    # ---- Data pass (OPEN #2 for /main/points + /cloud_pcd) ----
    extract_pcd_and_tf_single_open(
        bag_path=bag_path,
        topic_points=topic_points,
        interval_sec=interval_sec,
        source_frame=source_frame,
        target_frame=target_frame,
        tf_processor=tfp,
        aircraft_model=aircraft_model,
        pixel_shifts=pixel_shifts,     # destagger only if provided
        z_ref_source_frame="towbar",
        z_ref_target_frame="main_sensor_lidar",
        initial_z_min=None,            # lazily computed from TF
        model_topic="/cloud_pcd",      # saved on first encounter
        save_model=True,
        model_voxel=0.03,              # optional: 3 cm downsample
        model_color=(0.85, 0.85, 0.85),
        model_overwrite=False,
    )
    tfp.destroy_node()
    rclpy.shutdown()
    H5_FILE = Path("/home/femi/Benchmarking_framework/Data/machine_learning_dataset/HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5")
    MODEL = Path("/home/femi/Benchmarking_framework/Data/Aircraft_models/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_model.pcd")

    process_hdf5_and_visualize_alignment(
        h5_path=str(H5_FILE),
        source_pcd_path=str(MODEL),
        distance_threshold=0.2,
    )
    if visualise:
        visualize_all_views(H5_FILE)


