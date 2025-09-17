from __future__ import annotations
import os
import numpy as np
import cv2

from .config import Config
from .io_h5 import load_h5_scene, build_rai_rgb
from .segmentation import yolo_segment_to_mask, segmask_to_pcd_files, save_overlay_images
from .pcd_utils import load_points_pcd, denoise_points
from .range_match import match_ranges_from_h5_to_pcd
from .pca_axes import pca_align_rotation, euler_zyx_from_R
from .clustering import pick_nose_cluster_by_range_with_r
from .viz import visualize_with_highlight_and_axes
from .icp_refine import refine_icp_with_seed
from .io_h5 import read_scene_gt_transform
from .metrics import pose_error_metrics




def run_pipeline(cfg: Config):
    # 1) Load scene from H5
    pts_h5, cols, h, w = load_h5_scene(cfg.h5_path, cfg.scene_id)
    print(f"Loaded scene '{cfg.scene_id}' from H5 with {pts_h5.shape[0]} points, frame {h}x{w}")

    # 2) YOLO segmentation (optional)
    if cfg.run_segmentation:
        rgb = build_rai_rgb(pts_h5, cols, h, w)
        mask_u8, best_box = yolo_segment_to_mask(rgb, conf=cfg.conf, iou=cfg.iou, img_size=cfg.img_size)
        if cfg.dilate_px_mask > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*cfg.dilate_px_mask+1, 2*cfg.dilate_px_mask+1))
            mask_u8 = cv2.dilate(mask_u8, k)
        mask_bool = mask_u8 > 127
        save_overlay_images(cfg.output_dir, cfg.scene_id, rgb, mask_bool, best_box)
        path_all, path_ng = segmask_to_pcd_files(cfg.output_dir, cfg.scene_id, pts_h5, cols, mask_bool,
                                                 use_gray_from=cfg.use_gray_from,
                                                 remove_ground_pcd=cfg.remove_ground_pcd)
        seg_pcd_path = path_ng if (cfg.choose_noground_for_axes and path_ng is not None) else path_all
    else:
        candidate_ng = os.path.join(cfg.output_dir, f"{cfg.scene_id}_seg_points_noground.pcd")
        candidate_all = os.path.join(cfg.output_dir, f"{cfg.scene_id}_seg_points.pcd")
        seg_pcd_path = candidate_ng if (cfg.choose_noground_for_axes and os.path.isfile(candidate_ng)) else candidate_all
        if not os.path.isfile(seg_pcd_path):
            raise FileNotFoundError(f"No segmented PCD found at: {seg_pcd_path}")

    print(f"Using segmented PCD for axes: {seg_pcd_path}")

    # 3) Load segmented PCD and denoise
    seg_xyz = load_points_pcd(seg_pcd_path)
    denoised, mask = denoise_points(seg_xyz, cfg.denoise_method, cfg.sor_nb_neighbors, cfg.sor_std_ratio, cfg.ror_radius, cfg.ror_min_neighbors)
    print(f"Denoised PCD: {len(denoised)} points")

    # 4) Fetch H5 (x,y,z,range) and match range to denoised PCD
    xyz_h5 = pts_h5[:, :3].astype(np.float32)
    if "range" not in cols:
        raise ValueError("H5 does not contain 'range' column.")
    r_h5 = pts_h5[:, cols.index("range")].astype(np.float32)

    r_for_pcd, matched_mask = match_ranges_from_h5_to_pcd(xyz_h5, r_h5, seg_xyz, cfg.round_decimals, cfg.kdtree_tol)
    r_aligned = r_for_pcd[mask]
    print(f"Matched ranges: {matched_mask.sum()}/{matched_mask.size} on raw seg PCD; aligned after denoise: {np.isfinite(r_aligned).sum()} finite")

    # 5) PCA + axis selection
    R_align, mu, V = pca_align_rotation(denoised, r_aligned, align_to=cfg.align_to)

    np.set_printoptions(precision=5, suppress=True)
    print("\n[Means]\nmu (centroid) =\n" + np.array2string(mu, separator=", "))
    print("\n[Basis from PCA after orientation]\nV (columns = X, Y, Z) =\n" + np.array2string(V, separator=", "))
    print("\n[Alignment rotation]\nR_align (maps chosen +X to world {} ) =\n{}".format(cfg.align_to, np.array2string(R_align, separator=", ")))

    yaw, pitch, roll = euler_zyx_from_R(R_align)
    print(f"Align(chosen-major -> {cfg.align_to}) Euler ZYX (deg): yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}")

    # 6) Target by range clustering
    finite_mask = np.isfinite(r_aligned)
    if not finite_mask.any():
        raise ValueError("No finite ranges for clustering.")
    pts_for_cluster = denoised[finite_mask]
    r_for_cluster = r_aligned[finite_mask]

    target_sub, idx_sub, rmin, clusters_sub = pick_nose_cluster_by_range_with_r(
        r=r_for_cluster, pts=pts_for_cluster, eps=cfg.cluster_eps, min_points=cfg.cluster_min_points
    )
    idxs_full = np.flatnonzero(finite_mask)
    idx_full = int(idxs_full[idx_sub])
    target = denoised[idx_full]

    clusters = [idxs_full[c] for c in clusters_sub] if clusters_sub else []
    nose_cluster = None
    if clusters:
        for c in clusters:
            if idx_full in c:
                nose_cluster = c
                break

    dist = float(np.linalg.norm(target))
    print(f"[range_cluster_min] target -> idx={idx_full}, r={rmin:.6f}, euclid={dist:.6f}, coord={target.tolist()}")


    # 8) Visualize
    span = denoised.max(axis=0) - denoised.min(axis=0)
    axis_len = float(max(span)) * cfg.axis_length_scale if np.isfinite(span).all() else 2.0
    axis_len = max(axis_len, 0.1)
    visualize_with_highlight_and_axes(
        denoised,
        target,
        origin=target,
        axes=V,
        axis_length=axis_len,
        point_size=cfg.point_size,
        colors_cfg={
            "sphere_color": cfg.sphere_color,
            "nose_color": cfg.nose_color,
            "closest_color": cfg.closest_color,
            "other_color": cfg.other_color,
        },
        save_colored_pcd=cfg.save_colored_pcd,
        colored_pcd_path=cfg.colored_pcd_path,
        highlight_mode=cfg.highlight_mode,
        sphere_radius=cfg.sphere_radius,
    )

    # 9) 4x4 transform (rotation + translation)
    # 9) 4x4 transform (rotation + translation) from PCA/cluster stage
    T_target = np.eye(4)
    T_target[:3, :3] = R_align
    T_target[:3, 3] = target

    print("\n[4x4 Transform matrix] (seed from PCA/cluster)")
    print(np.array2string(T_target, separator=", "))

    # 10) Optional ICP refinement starting from T_target
    T_icp = None
    icp_info = None
    if cfg.icp_enabled:
        model_pcd = cfg.model_pcd_path
        if not model_pcd:
            raise ValueError("cfg.icp_enabled is True but cfg.model_pcd_path is empty.")
        # pick target for ICP: explicit override or the segmented PCD we produced
        icp_target = (cfg.icp_target_path or seg_pcd_path)
        icp_info = refine_icp_with_seed(
            source_model_pcd_path=model_pcd,
            target_scene_pcd_path=icp_target,
            start_transform_4x4=T_target,
            voxel=cfg.icp_voxel,
            up_axis=cfg.icp_up_axis,
            roi_margin_mult=cfg.icp_roi_margin_mult,
            do_tiny_sweep=cfg.icp_do_tiny_sweep,
            sweep_deg=cfg.icp_sweep_deg,
            sweep_step_deg=cfg.icp_sweep_step_deg,
            visualize=cfg.icp_visualize,
        )
        T_icp = icp_info["T_icp"]
        print("\n[ICP] refinement complete")
        print("Final T_icp =\n", np.array2string(T_icp, separator=", "))
        # --- Compare with GT from H5 (if present/enabled) ---
        gt_T = None
        seed_vs_gt = None
        icp_vs_gt = None

        if cfg.compare_with_h5_gt:
            gt_T = read_scene_gt_transform(cfg.h5_path, cfg.scene_id)
            if gt_T is not None:
                print("\n[GT] ground-truth transform found in H5")
                # compare GT vs seed (PCA/cluster)
                seed_vs_gt = pose_error_metrics(T_target, gt_T)
                print(f"[GT vs Seed]  transl_err = {seed_vs_gt['transl_err']:.4f} m,"
                      f" rot_err = {seed_vs_gt['rot_err_deg']:.3f}°")
                # compare GT vs ICP (if ran)
                if 'T_icp' in (icp_info or {}) and icp_info['T_icp'] is not None:
                    icp_vs_gt = pose_error_metrics(icp_info['T_icp'], gt_T)
                    print(f"[GT vs ICP ]  transl_err = {icp_vs_gt['transl_err']:.4f} m,"
                          f" rot_err = {icp_vs_gt['rot_err_deg']:.3f}°")
            else:
                print("\n[GT] no ground-truth transform found in H5 (skipping comparison)")

    return {
        "target_point": target,
        "R_align": R_align,
        "V_axes": V,
        "centroid": mu,
        "euler_zyx_deg": (yaw, pitch, roll),
        "transform_seed": T_target,  # <-- seed from PCA/cluster
        "transform_icp": T_icp,  # <-- None if ICP disabled
        "icp_info": icp_info,  # voxel, scale, per-stage log
        "raster_path": cfg.raster_out_png,
        "seg_pcd_path": seg_pcd_path,
        "colored_pcd_path": (cfg.colored_pcd_path if cfg.save_colored_pcd else None),
        "gt_transform": gt_T,
        "metrics_vs_gt_seed": seed_vs_gt,
        "metrics_vs_gt_icp": icp_vs_gt,

    }
