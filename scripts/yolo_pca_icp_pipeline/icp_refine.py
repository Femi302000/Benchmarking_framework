from __future__ import annotations
import math
from typing import Optional
import numpy as np
import open3d as o3d

# -------- helpers (trimmed to essentials) --------
def prep(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    p = pcd.voxel_down_sample(voxel)
    p.remove_non_finite_points()
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.5, max_nn=30))
    return p

def median_nn_spacing(pcd: o3d.geometry.PointCloud, k: int = 6) -> float:
    pts = np.asarray(pcd.points)
    n = len(pts)
    if n < k + 1:
        return 0.05
    kdt = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    step = max(1, n // 2000)
    for i in range(0, n, step):
        _, _, dist2 = kdt.search_knn_vector_3d(pts[i], k)
        if len(dist2) > 1:
            d = np.sqrt(dist2[1:])  # skip itself
            dists.append(np.median(d))
    return float(np.median(dists)) if dists else 0.05

def guess_scale(src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud) -> float:
    ds = median_nn_spacing(src)
    dt = median_nn_spacing(tgt)
    if ds == 0 or dt == 0:
        return 1.0
    return float(np.clip(dt / ds, 1e-3, 1e3))

def _R_about_axis(theta: float, axis: str) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    R = np.eye(4)
    if axis == "z":
        R[:3,:3] = np.array([[ c,-s, 0],[ s, c, 0],[ 0, 0, 1]])
    elif axis == "y":
        R[:3,:3] = np.array([[ c, 0, s],[ 0, 1, 0],[-s, 0, c]])
    elif axis == "x":
        R[:3,:3] = np.array([[1, 0,  0],[0, c, -s],[0, s,  c]])
    else:
        raise ValueError("axis must be 'x','y','z'")
    return R

def _nn_cost_for_T(src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud, T: np.ndarray, sample: int = 2000) -> float:
    stf = o3d.geometry.PointCloud(src)
    stf.transform(T)
    P = np.asarray(stf.points)
    if len(P) == 0:
        return float("inf")
    kd = o3d.geometry.KDTreeFlann(tgt)
    step = max(1, len(P)//sample)
    errs = []
    for i in range(0, len(P), step):
        _, _, dist2 = kd.search_knn_vector_3d(P[i], 1)
        errs.append(math.sqrt(dist2[0]))
    return float(np.mean(errs)) if errs else float("inf")

def crop_target_to_source_roi(src, tgt, T, margin: float):
    src_tf = o3d.geometry.PointCloud(src)
    src_tf.transform(T)
    aabb = src_tf.get_axis_aligned_bounding_box()
    minb = np.array(aabb.get_min_bound(), dtype=float) - margin
    maxb = np.array(aabb.get_max_bound(), dtype=float) + margin
    return tgt.crop(o3d.geometry.AxisAlignedBoundingBox(minb, maxb))

# -------- public entrypoint --------
def refine_icp_with_seed(
    source_model_pcd_path: str,
    target_scene_pcd_path: str,
    start_transform_4x4: np.ndarray,
    *,
    voxel: float | None = None,
    up_axis: str = "z",
    roi_margin_mult: float = 6.0,
    do_tiny_sweep: bool = True,
    sweep_deg: int = 20,
    sweep_step_deg: int = 5,
    visualize: bool = False,
) -> dict:
    """
    Refine alignment via GICP starting from a seed 4x4 transform.
    Returns {'T_icp': 4x4, 'voxel': float, 'scale_applied': float, 'icp_log': list[dict]}.
    """
    src_raw = o3d.io.read_point_cloud(source_model_pcd_path)
    tgt_raw = o3d.io.read_point_cloud(target_scene_pcd_path)

    # choose voxel automatically if not supplied
    if voxel is None:
        tmp_s = prep(src_raw, 0.05)
        tmp_t = prep(tgt_raw, 0.05)
        voxel = max(1.5 * np.median([median_nn_spacing(tmp_s), median_nn_spacing(tmp_t)]), 0.02)

    # preprocess
    src = prep(src_raw, voxel)
    tgt = prep(tgt_raw, voxel)

    # scale match
    s = guess_scale(src, tgt)
    if abs(math.log10(s)) > 0.2:  # > ~1.6×
        S = np.eye(4); S[0,0]=S[1,1]=S[2,2]=s
        src.transform(S)

    # Optional tiny sweep around up axis to lower NN cost before ICP
    T0 = start_transform_4x4.copy()
    if do_tiny_sweep:
        best_T, best_cost = T0, _nn_cost_for_T(src, tgt, T0)
        for deg in range(-sweep_deg, sweep_deg+1, sweep_step_deg):
            if deg == 0:
                continue
            T = _R_about_axis(math.radians(deg), up_axis) @ T0
            c = _nn_cost_for_T(src, tgt, T)
            if c < best_cost:
                best_T, best_cost = T, c
        T0 = best_T

    # crop ROI
    margin = roi_margin_mult * voxel
    tgt_roi = crop_target_to_source_roi(src, tgt, T0, margin)

    # GICP refinement (multi-stage)
    T = T0.copy()
    est = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    thresholds = [voxel * 6, voxel * 3, voxel * 1.5, voxel * 0.75]
    iters =      [200,       150,        120,          80]
    log = []
    for dist, iters_i in zip(thresholds, iters):
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt_roi, dist, T, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iters_i, relative_fitness=1e-7, relative_rmse=1e-7
            )
        )
        T = reg.transformation
        log.append({"thresh": float(dist), "fitness": float(reg.fitness), "rmse": float(reg.inlier_rmse)})

    if visualize:
        src_aligned = o3d.io.read_point_cloud(source_model_pcd_path)
        src_aligned.transform(T)
        tgt_vis = o3d.io.read_point_cloud(target_scene_pcd_path)
        o3d.visualization.draw_geometries(
            [src_aligned.paint_uniform_color([1.0, 0.706, 0.0]),
             tgt_vis.paint_uniform_color([0.0, 0.651, 0.929])],
            window_name="Alignment (source → target)"
        )

    return {"T_icp": T, "voxel": float(voxel), "scale_applied": float(s), "icp_log": log}
