from __future__ import annotations
import numpy as np

def _rot_angle_deg(R: np.ndarray) -> float:
    # clamp trace to valid range to avoid NaNs from rounding
    tr = float(np.trace(R))
    cos_theta = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return float(np.degrees(np.arccos(cos_theta)))

def _euler_zyx_deg(R: np.ndarray) -> tuple[float,float,float]:
    sy = -R[2, 0]; cy = np.sqrt(max(0.0, 1.0 - sy * sy))
    if cy > 1e-8:
        yaw   = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        pitch = float(np.degrees(np.arcsin(sy)))
        roll  = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    else:
        yaw   = float(np.degrees(np.arctan2(-R[0, 1], R[1, 1])))
        pitch = float(np.degrees(np.arcsin(sy)))
        roll  = 0.0
    return yaw, pitch, roll

def pose_error_metrics(T_est: np.ndarray, T_gt: np.ndarray) -> dict:
    """
    Returns key metrics between two 4x4 transforms:
    - transl_err (L2, meters)
    - transl_err_xyz (per-axis)
    - rot_err_deg (angle of ΔR in degrees)
    - delta_euler_zyx_deg (yaw, pitch, roll of ΔR)
    - T_delta (T_est^{-1} @ T_gt)
    """
    if T_est.shape != (4,4) or T_gt.shape != (4,4):
        raise ValueError("T_est and T_gt must be 4x4")
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]
    R_gt,  t_gt  = T_gt[:3, :3],  T_gt[:3, 3]

    # ΔT that maps est frame to gt frame
    R_delta = R_est.T @ R_gt
    t_delta = R_est.T @ (t_gt - t_est)

    transl_err_xyz = t_delta
    transl_err = float(np.linalg.norm(t_delta))
    rot_err_deg = _rot_angle_deg(R_delta)
    delta_euler = _euler_zyx_deg(R_delta)

    T_delta = np.eye(4)
    T_delta[:3, :3] = R_delta
    T_delta[:3, 3]  = t_delta

    return {
        "transl_err": transl_err,
        "transl_err_xyz": tuple(map(float, transl_err_xyz)),
        "rot_err_deg": float(rot_err_deg),
        "delta_euler_zyx_deg": tuple(map(float, delta_euler)),
        "T_delta": T_delta,
    }
