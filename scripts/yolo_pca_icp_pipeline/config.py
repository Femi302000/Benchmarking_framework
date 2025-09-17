from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Config:


    # --- Data paths ---
    h5_path: str = \
        "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"
    scene_id: str = "scene_000"
    output_dir: str = "seg_outputs"

    # --- YOLO segmentation ---
    run_segmentation: bool = True
    yolo_weights: str = "yolov8s-seg.pt"
    conf: float = 0.15
    iou: float = 0.50
    img_size: int = 1280
    dilate_px_mask: int = 1

    # --- Mask -> PCD options ---
    use_gray_from: str = "reflectivity"
    remove_ground_pcd: bool = True
    choose_noground_for_axes: bool = True

    # --- Raster (bird's-eye) ---
    raster_out_png: str = "birdview_seg.png"
    pixel_size: float = 0.02
    dilate_px_raster: int = 1
    fixed_bounds: Optional[Tuple[float, float, float, float]] = None

    # --- Denoising for PCA cloud ---
    denoise_method: str = "statistical"  # 'statistical' | 'radius' | 'none'
    sor_nb_neighbors: int = 20
    sor_std_ratio: float = 2.0
    ror_radius: float = 0.10
    ror_min_neighbors: int = 8

    # --- Matching H5 <-> PCD ---
    round_decimals: int = 6
    kdtree_tol: float = 1e-4

    # --- Target selection by range clustering ---
    cluster_eps: float = 0.20
    cluster_min_points: int = 10

    icp_enabled: bool = False
    model_pcd_path: str = ""  # path to your CAD/model .pcd (source)
    icp_target_path: str = "/home/femi/Benchmarking_framework/Data/sequence_from_scene/HAM_Airport_2024_08_08_movement_a320_ceo_Germany/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_scene0000_9.095s_filtered.pcd"
    icp_voxel: float | None = None
    icp_up_axis: str = "z"
    icp_roi_margin_mult: float = 6.0
    icp_do_tiny_sweep: bool = True
    icp_sweep_deg: int = 20
    icp_sweep_step_deg: int = 5
    icp_visualize: bool = False

    # --- Visualization ---
    point_size: float = 3.0
    sphere_radius: float = 0.10
    highlight_mode: str = "sphere"  # 'color' | 'sphere'
    sphere_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    nose_color: tuple[float, float, float] = (1.0, 0.2, 0.2)
    closest_color: tuple[float, float, float] = (1.0, 0.9, 0.2)
    other_color: tuple[float, float, float] = (0.65, 0.65, 0.65)
    save_colored_pcd: bool = True
    colored_pcd_path: str = "seg_points_highlighted.pcd"

    # --- Axes ---
    axis_length_scale: float = 0.25
    align_to: str = "+X"  # '+X' or '-X'
    axis_selection_mode: str = "corr_range"
    compare_with_h5_gt: bool = True


CFG = Config()
