#!/usr/bin/env python3
"""
Compare EST vs GT transforms (A320 model vs scene) and visualize axes.

- Loads GT from H5: /{scene_id}/metadata/tf_matrix
- Uses your EST transform (seed or ICP) that you paste below
- Computes delta metrics and detects ±90° about Z convention mismatch
- Visualizes scene (gray), model@EST (orange), model@GT (blue),
  and optionally model@EST_corrected (purple) if a 90° fix is applied.

Requires: numpy, open3d, h5py
"""

import numpy as np
import open3d as o3d
import h5py
import math

# ========== EDIT ME ==========
H5_PATH = "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"
SCENE_ID = "scene_000"
SCENE_PCD = "seg_outputs/scene_000_seg_points_noground.pcd"
MODEL_PCD = "/home/femi/Benchmarking_framework/Data/Aircraft_models/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_model.pcd"

# Paste one of your transforms here (seed or ICP). This is your ESTIMATE.
T_EST = np.array([
    [ 0.01690, -0.99964, -0.02085, -0.75734],
    [ 0.99932,  0.01757, -0.03245, -4.15760],
    [ 0.03281, -0.02028,  0.99926,  1.70615],
    [ 0.00000,  0.00000,  0.00000,  1.00000],
], dtype=float)

# Toggle: draw EST corrected by ±90° about Z if a convention mismatch is detected
DRAW_CORRECTED = True
AXIS_SIZE = 1.0
VOXEL_SCENE = 0.03
VOXEL_MODEL = 0.02
# =============================


def read_gt_tf_matrix(h5_path: str, scene_id: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if scene_id not in f:
            raise KeyError(f"scene '{scene_id}' not in H5")
        g = f[scene_id]
        if "metadata" not in g or "tf_matrix" not in g["metadata"]:
            raise KeyError(f"'{scene_id}/metadata/tf_matrix' not found in H5")
        T = np.asarray(g["metadata"]["tf_matrix"][()], dtype=float).reshape(4, 4)
        T[-1, :] = [0, 0, 0, 1]
        return T


def rot_angle_deg(R: np.ndarray) -> float:
    # robust acos(trace(R)-1)/2 with clamping
    tr = float(np.trace(R))
    c = (tr - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(math.acos(c)))


def euler_zyx_deg(R: np.ndarray) -> tuple[float, float, float]:
    sy = -R[2, 0]; cy = math.sqrt(max(0.0, 1.0 - sy * sy))
    if cy > 1e-8:
        yaw   = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        pitch = math.degrees(math.asin(sy))
        roll  = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    else:
        yaw   = math.degrees(math.atan2(-R[0, 1], R[1, 1]))
        pitch = math.degrees(math.asin(sy))
        roll  = 0.0
    return (yaw, pitch, roll)


def metrics_vs_gt(T_est: np.ndarray, T_gt: np.ndarray) -> dict:
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]
    R_gt,  t_gt  = T_gt[:3, :3],  T_gt[:3, 3]
    R_delta = R_est.T @ R_gt
    t_delta = R_est.T @ (t_gt - t_est)
    return {
        "rot_err_deg": rot_angle_deg(R_delta),
        "delta_euler_zyx_deg": euler_zyx_deg(R_delta),
        "transl_err": float(np.linalg.norm(t_delta)),
        "transl_err_xyz": tuple(map(float, t_delta)),
        "T_delta": np.block([[R_delta, t_delta.reshape(3,1)], [np.zeros((1,3)), np.array([[1.0]])]]),
    }


def is_near_Rz_90(R_delta: np.ndarray, tol_deg: float = 1.0) -> tuple[bool, float]:
    """Check if R_delta is approximately a rotation about Z by ±90°."""
    # two candidate rotations
    Rz_p90 = np.array([[0, -1, 0],
                       [1,  0, 0],
                       [0,  0, 1]], dtype=float)
    Rz_m90 = np.array([[ 0, 1, 0],
                       [-1, 0, 0],
                       [ 0, 0, 1]], dtype=float)
    # distance metric via angle of R_delta^T * Rz
    def ang(Ra, Rb):
        return rot_angle_deg(Ra.T @ Rb)
    a = ang(R_delta, Rz_p90)
    b = ang(R_delta, Rz_m90)
    if a <= tol_deg:
        return True, +90.0
    if b <= tol_deg:
        return True, -90.0
    return False, 0.0


def apply_Rz_deg(T: np.ndarray, deg: float) -> np.ndarray:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    Rz = np.array([[ c, -s, 0],
                   [ s,  c, 0],
                   [ 0,  0, 1]], dtype=float)
    Tout = T.copy()
    Tout[:3, :3] = T[:3, :3] @ Rz
    return Tout


def frame(T: np.ndarray, size: float = 1.0):
    fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    fr.transform(T.copy())
    return fr


def load_ds(pcd_path: str, voxel: float | None):
    p = o3d.io.read_point_cloud(pcd_path)
    if voxel and voxel > 0:
        p = p.voxel_down_sample(voxel)
    p.remove_non_finite_points()
    return p


def main():
    # Read GT
    T_gt = read_gt_tf_matrix(H5_PATH, SCENE_ID)
    print("[GT] T_gt =\n", np.array2string(T_gt, precision=5, suppress_small=True))

    # Metrics EST vs GT
    m = metrics_vs_gt(T_EST, T_gt)
    print("\n[EST vs GT]")
    print(f" rot_err = {m['rot_err_deg']:.3f}°")
    print(f" delta_euler_zyx = {tuple(round(v, 3) for v in m['delta_euler_zyx_deg'])}")
    print(f" transl_err = {m['transl_err']:.4f} m")
    print(f" transl_err_xyz = {tuple(round(v, 4) for v in m['transl_err_xyz'])}")
    print(" T_delta =\n", np.array2string(m["T_delta"], precision=5, suppress_small=True))

    # Detect ±90° about Z mismatch
    near90, sign = is_near_Rz_90(m["T_delta"][:3, :3], tol_deg=1.0)
    T_corr = None
    if near90 and DRAW_CORRECTED:
        T_corr = apply_Rz_deg(T_EST, sign)
        mc = metrics_vs_gt(T_corr, T_gt)
        print(f"\n[Convention] Detected ~{int(sign)}° Z-offset; drawing corrected EST as well.")
        print(f" [CORR vs GT] rot_err = {mc['rot_err_deg']:.3f}°, transl_err = {mc['transl_err']:.4f} m")

    # Visualize
    scene = load_ds(SCENE_PCD, VOXEL_SCENE).paint_uniform_color([0.70, 0.70, 0.70])
    model = load_ds(MODEL_PCD, VOXEL_MODEL)

    model_est = o3d.geometry.PointCloud(model)
    model_est.transform(T_EST)
    model_est.paint_uniform_color([1.0, 0.706, 0.0])  # orange

    model_gt = o3d.geometry.PointCloud(model)
    model_gt.transform(T_gt)
    model_gt.paint_uniform_color([0.0, 0.651, 0.929])  # blue

    geoms = [scene, model_est, model_gt, frame(T_EST, AXIS_SIZE), frame(T_gt, AXIS_SIZE)]

    if T_corr is not None:
        model_corr = o3d.geometry.PointCloud(model)
        model_corr.transform(T_corr)
        model_corr.paint_uniform_color([0.6, 0.2, 0.9])  # purple
        geoms += [model_corr, frame(T_corr, AXIS_SIZE)]

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Model vs GT axes (orange=EST, blue=GT, purple=EST corrected)",
    )


if __name__ == "__main__":
    main()
