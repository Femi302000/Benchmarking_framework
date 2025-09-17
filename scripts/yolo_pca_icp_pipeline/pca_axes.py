from __future__ import annotations
import numpy as np


def _align_axis_to_target_x(axis: np.ndarray, y_hint: np.ndarray, align_to: str = "+X") -> np.ndarray:
    x_axis = axis / np.linalg.norm(axis)
    if align_to not in {"+X", "-X"}:
        raise ValueError("align_to must be '+X' or '-X'")
    target_x = np.array([1.0, 0.0, 0.0]) if align_to == "+X" else np.array([-1.0, 0.0, 0.0])
    v = np.cross(x_axis, target_x)
    s = np.linalg.norm(v)
    c = float(np.dot(x_axis, target_x))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        k = y_hint / np.linalg.norm(y_hint)
        K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    k = v / s
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    return np.eye(3) + K * s + (K @ K) * (1 - c)


def pca_align_rotation(pts: np.ndarray, r: np.ndarray, align_to: str = "+X"):
    if r is None or r.shape[0] != pts.shape[0]:
        raise ValueError("Need r with same length as pts for correlation-based axis selection.")

    mu = pts.mean(axis=0)
    Xc = pts - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T

    # right-handed
    x_axis, y_axis = V[:, 0], V[:, 1]
    z_axis = np.cross(x_axis, y_axis)
    n = np.linalg.norm(z_axis)
    if n < 1e-12:
        V = np.eye(3)
    else:
        z_axis /= n
        y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        V = np.stack([x_axis, y_axis, z_axis], axis=1)
        if np.linalg.det(V) < 0:
            V[:, 2] *= -1.0

    # correlations
    scores = Xc @ V
    r64 = r.astype(np.float64)
    corr_signed = np.full(3, np.nan)
    corr_abs = np.zeros(3)
    for j in range(3):
        sj = scores[:, j].astype(np.float64)
        finite = np.isfinite(sj) & np.isfinite(r64)
        if finite.sum() >= 4 and np.std(sj[finite]) > 1e-12 and np.std(r64[finite]) > 1e-12:
            c = float(np.corrcoef(sj[finite], r64[finite])[0, 1])
            corr_signed[j] = c
            corr_abs[j] = abs(c)
        else:
            corr_signed[j] = np.nan
            corr_abs[j] = 0.0

    # choose axis by |corr|, put in col 0
    idx_max = int(np.argmax(corr_abs))
    if idx_max != 0:
        cols = [0, 1, 2]
        cols[0], cols[idx_max] = cols[idx_max], cols[0]
        V = V[:, cols]

    # re-orthonormalize & handedness
    x_axis = V[:, 0] / np.linalg.norm(V[:, 0])
    tmp_z = np.cross(x_axis, V[:, 1])
    if np.linalg.norm(tmp_z) < 1e-12:
        tmp_z = V[:, 2]
    z_axis = tmp_z / np.linalg.norm(tmp_z)
    y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
    V = np.stack([x_axis, y_axis, z_axis], axis=1)
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1.0

    # force +X to increase range
    s0_raw = (pts @ V[:, 0]).astype(np.float64)
    finite = np.isfinite(s0_raw) & np.isfinite(r64)
    if finite.sum() >= 16 and np.std(r64[finite]) > 1e-12 and np.std(s0_raw[finite]) > 1e-12:
        slope = float(np.polyfit(s0_raw[finite], r64[finite], 1)[0])
        if slope < 0:
            x_axis = -V[:, 0]
            tmp_z = np.cross(x_axis, V[:, 1])
            tmp_z = tmp_z if np.linalg.norm(tmp_z) >= 1e-12 else V[:, 2]
            z_axis = tmp_z / np.linalg.norm(tmp_z)
            y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
            V = np.stack([x_axis, y_axis, z_axis], axis=1)
            if np.linalg.det(V) < 0:
                V[:, 2] *= -1.0

    # enforce Z ~ up so Y is lateral
    up = np.array([0.0, 0.0, 1.0])
    x_axis = V[:, 0] / np.linalg.norm(V[:, 0])
    cand1 = V[:, 1] / np.linalg.norm(V[:, 1])
    cand2 = V[:, 2] / np.linalg.norm(V[:, 2])
    z_axis = cand1 if abs(np.dot(cand1, up)) >= abs(np.dot(cand2, up)) else cand2
    if np.dot(z_axis, up) < 0:
        z_axis = -z_axis
    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-12:
        y_axis = np.cross(up, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis); z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
    V = np.stack([x_axis, y_axis, z_axis], axis=1)
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1.0

    R_align = _align_axis_to_target_x(V[:, 0], V[:, 1], align_to=align_to)
    return R_align, mu, V


def euler_zyx_from_R(R: np.ndarray):
    sy = -R[2, 0]
    cy = np.sqrt(max(0.0, 1.0 - sy * sy))
    if cy > 1e-8:
        yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        pitch = float(np.degrees(np.arcsin(sy)))
        roll = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    else:
        yaw = float(np.degrees(np.arctan2(-R[0, 1], R[1, 1])))
        pitch = float(np.degrees(np.arcsin(sy)))
        roll = 0.0
    return yaw, pitch, roll