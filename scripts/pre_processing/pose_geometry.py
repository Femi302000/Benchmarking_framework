"""
pose_open3d_demo.py
-------------------
Self-contained: estimate aircraft pose (no CAD, no RANSAC) and visualize flags with Open3D.

Body frame:
  x_b = forward (nose), y_b = right wing, z_b = up
Origin ~ nose extremity.

Deps (install):
  pip install numpy open3d
  # optional (for smoothing): pip install scipy
"""

from __future__ import annotations
import math, json
import numpy as np

# ===================== USER SETTINGS =====================
INPUT_PATH = "/home/femi/Benchmarking_framework/scripts/yolo/seg_outputs/scene_000_seg_points_noground.pcd"   # <-- set your file path (.ply/.pcd/.xyz or text Nx3)
VOXEL = 0.03                       # meters
NORMAL_K = 40
# =========================================================

# ---- Optional smoothing (scipy → Savitzky–Golay; else moving average) ----
try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ----------------------------- Utilities ----------------------------- #

def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        return points.copy()
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[np.sort(idx)]

def _knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    N = points.shape[0]
    batch = max(1, 2000000 // max(1, N))
    idx_all = np.empty((N, k), dtype=np.int64)
    for start in range(0, N, batch):
        end = min(N, start + batch)
        P = points[start:end]
        d2 = np.sum((P[:, None, :] - points[None, :, :])**2, axis=2)
        kth = min(k-1, d2.shape[1]-1)
        idx_part = np.argpartition(d2, kth=kth, axis=1)[:, :k]
        row = np.arange(end - start)[:, None]
        d2k = d2[row, idx_part]
        order = np.argsort(d2k, axis=1)
        idx_all[start:end] = idx_part[row, order]
    return idx_all

def _huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(residuals)
    w = np.ones_like(a)
    mask = a > delta
    w[mask] = (delta / a[mask])
    return w

def _orthonormal_basis_from_z(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = _normalize(z)
    if abs(z[2]) < 0.9:
        tmp = np.array([0, 0, 1.0])
    else:
        tmp = np.array([1.0, 0, 0])
    x = _normalize(np.cross(tmp, z))
    y = _normalize(np.cross(z, x))
    return x, y

# -------------------------- Normal estimation ------------------------- #

def estimate_normals_irls(points: np.ndarray, k: int = 40, huber_delta: float = 0.06) -> np.ndarray:
    N = points.shape[0]
    idx_knn = _knn_indices(points, k)
    normals = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        nbr = points[idx_knn[i]]
        c = nbr.mean(axis=0)
        X = nbr - c
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        n = Vt[-1]
        for _ in range(3):
            r = X @ n
            w = _huber_weights(r, huber_delta)
            C = (X * w[:, None]).T @ X
            _, _, Vt2 = np.linalg.svd(C, full_matrices=False)
            n = Vt2[:, -1]
        normals[i] = _normalize(n)
    return normals

# -------- Spherical density + axis great-circle voting (no RANSAC) ---- #

def spherical_kde_peaks(normals: np.ndarray, res_deg: float = 5.0, kappa: float = 20.0,
                        n_peaks: int = 6, nms_deg: float = 10.0) -> list[np.ndarray]:
    thetas = np.linspace(0, math.pi, int(round(180.0 / res_deg)) + 1)
    phis = np.linspace(0, 2*math.pi, int(round(360.0 / res_deg)), endpoint=False)
    G = np.array([[math.sin(th)*math.cos(ph), math.sin(th)*math.sin(ph), math.cos(th)]
                  for th in thetas for ph in phis], dtype=np.float64)
    G = _normalize(G)
    scores = np.exp(kappa * (G @ normals.T))
    density = scores.mean(axis=1)
    dens = density.copy()
    dirs = []
    cos_thr = math.cos(math.radians(nms_deg))
    for _ in range(n_peaks * 3):
        j = int(np.argmax(dens))
        if dens[j] <= 0: break
        d = G[j]; dirs.append(d)
        dots = G @ d
        dens[dots >= cos_thr] = 0.0
        if len(dirs) >= n_peaks: break
    return dirs

def spherical_axis_voting(normals: np.ndarray, grid_res_deg: float = 2.0, samples_per_gc: int = 180) -> np.ndarray:
    thetas = np.linspace(0, math.pi, int(round(180.0 / grid_res_deg)) + 1)
    phis = np.linspace(0, 2*math.pi, int(round(360.0 / grid_res_deg)), endpoint=False)
    G = np.array([[math.sin(th)*math.cos(ph), math.sin(th)*math.sin(ph), math.cos(th)]
                  for th in thetas for ph in phis], dtype=np.float64)
    G = _normalize(G)
    acc = np.zeros(G.shape[0], dtype=np.float64)
    for n in normals:
        n = _normalize(n)
        tmp = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = _normalize(np.cross(n, tmp))
        v = _normalize(np.cross(n, u))
        ts = np.linspace(0.0, 2*math.pi, samples_per_gc, endpoint=False)
        A = (np.cos(ts)[:, None] * u[None, :] + np.sin(ts)[:, None] * v[None, :])
        dots = A @ G.T
        idx = np.argmax(dots, axis=1)
        np.add.at(acc, idx, 1.0)
    acc = 0.5*acc + 0.25*np.roll(acc, 1) + 0.25*np.roll(acc, -1)  # light smoothing
    a_hat = G[np.argmax(acc)]
    return _normalize(a_hat)

# ------------------------------- Planes (Hough) ---------------------------- #

def plane_hough_from_normal(points: np.ndarray, normals: np.ndarray, n_dir: np.ndarray, d_bin: float):
    n_dir = _normalize(n_dir)
    cos_thr = math.cos(math.radians(10.0))
    close = (np.abs(normals @ n_dir) >= cos_thr)  # allow ±n
    if not np.any(close): return 0.0, 0, np.inf
    X = points[close]
    s = -(X @ n_dir)
    s_min, s_max = s.min(), s.max()
    nb = max(10, int(math.ceil((s_max - s_min) / max(1e-6, d_bin))))
    hist, edges = np.histogram(s, bins=nb, range=(s_min, s_max))
    j = int(np.argmax(hist)); support = int(hist[j])
    d_hat = 0.5*(edges[j] + edges[j+1])
    mask = (s >= edges[j]) & (s <= edges[j+1])
    if np.sum(mask) >= 3:
        dists = np.abs((X[mask] @ n_dir) + d_hat)
        thick = np.std(dists)
    else:
        thick = np.inf
    return float(d_hat), support, float(thick)

def detect_wing_plane(points: np.ndarray, normals: np.ndarray, axis_dir: np.ndarray,
                      d_bin: float, sphere_peaks: list[np.ndarray]):
    best = (None, 0.0, 0, float("inf"))  # (n, d, support, thickness)
    for n_dir in sphere_peaks:
        if abs(np.dot(n_dir, axis_dir)) > math.cos(math.radians(75.0)):
            continue
        d_hat, sup, thick = plane_hough_from_normal(points, normals, n_dir, d_bin=d_bin)
        if sup < 10: continue
        if sup > best[2] or (sup == best[2] and thick < best[3]):
            best = (n_dir, d_hat, sup, thick)
    return best

# ----------------------------- Axis line (IRLS) ---------------------------- #

def line_point_fixed_dir_irls(points: np.ndarray, a_hat: np.ndarray, delta: float = 0.06, iters: int = 5) -> np.ndarray:
    a_hat = _normalize(a_hat)
    c = points.mean(axis=0)
    for _ in range(iters):
        v = points - c
        perp = v - (v @ a_hat)[:, None] * a_hat[None, :]
        r = np.linalg.norm(perp, axis=1)
        w = _huber_weights(r, delta)
        Wsum = max(1e-12, w.sum())
        c_shift = ((points - (points @ a_hat)[:, None] * a_hat[None, :]) * w[:, None]).sum(axis=0) / Wsum
        c = (c @ a_hat) * a_hat + c_shift
    return c

# ------------------------- Cross-section circle fit ------------------------ #

def circle_fit_irls(xy: np.ndarray, delta: float = 0.03, iters: int = 5):
    x, y = xy[:,0], xy[:,1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x*x + y*y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        sol = np.zeros(3)
    cx, cy = sol[0], sol[1]
    r = math.sqrt(max(1e-12, sol[2] + cx*cx + cy*cy))
    c = np.array([cx, cy], dtype=np.float64)
    for _ in range(iters):
        d = np.linalg.norm(xy - c[None, :], axis=1)
        resid = d - r
        w = _huber_weights(resid, delta)
        d = np.clip(d, 1e-6, None)
        J = np.column_stack([(c[0] - x)/d, (c[1] - y)/d, -np.ones_like(d)])
        try:
            step, *_ = np.linalg.lstsq((J.T * w) @ J, (J.T * w) @ resid, rcond=None)
        except np.linalg.LinAlgError:
            break
        c = c - step[:2]
        r = r + step[2]
    d = np.linalg.norm(xy - c[None, :], axis=1)
    return c, float(r), float(np.mean(np.abs(d - r)))

def radius_profile(points: np.ndarray, axis_dir: np.ndarray, c_on_axis: np.ndarray,
                   window: float = 0.4, step: float = 0.2):
    a = _normalize(axis_dir)
    u, v = _orthonormal_basis_from_z(a)
    s = (points - c_on_axis) @ a
    s_min, s_max = float(np.min(s)), float(np.max(s))
    s_centers, r_vals, centers3d, centers2d = [], [], [], []
    cur = s_min + 0.5*window
    while cur <= s_max - 0.5*window + 1e-9:
        mask = (s >= cur - 0.5*window) & (s <= cur + 0.5*window)
        if np.sum(mask) >= 30:
            P = points[mask]
            origin = c_on_axis + cur * a
            X = P - origin
            xy = np.column_stack([X @ u, X @ v])
            c2, r, _ = circle_fit_irls(xy, delta=max(0.02, 0.5*window*0.02), iters=5)
            s_centers.append(cur)
            r_vals.append(r)
            centers2d.append(c2)
            centers3d.append(origin + c2[0]*u + c2[1]*v)
        cur += step
    if len(s_centers) == 0:
        return np.array([]), np.array([]), [], []
    return np.array(s_centers), np.array(r_vals), centers3d, centers2d

def _smooth_1d(y: np.ndarray, win: int = 7, poly: int = 2) -> np.ndarray:
    if y.size == 0: return y
    if _HAS_SCIPY and y.size >= win and win % 2 == 1:
        return savgol_filter(y, window_length=win, polyorder=min(poly, win-1))
    k = max(3, win); pad = k//2
    yp = np.pad(y, (pad, pad), mode='edge')
    return np.convolve(yp, np.ones(k)/k, mode='valid')

def choose_forward_from_taper(s: np.ndarray, r: np.ndarray) -> int:
    if s.size < 5: return +1
    r_s = _smooth_1d(r, win=7, poly=2)
    m = max(3, int(0.2 * s.size))
    def slope(seg_s, seg_r):
        if seg_s.size < 2: return 0.0
        A = np.column_stack([seg_s, np.ones_like(seg_s)])
        sol, *_ = np.linalg.lstsq(A, seg_r, rcond=None); return float(sol[0])
    slope_lo = slope(s[:m], r_s[:m]); slope_hi = slope(s[-m:], r_s[-m:])
    if slope_lo < slope_hi - 1e-6: return -1
    elif slope_hi < slope_lo - 1e-6: return +1
    else: return +1

# ------------------------------ Main pipeline ------------------------------ #

def estimate_pose(points: np.ndarray,
                  voxel_size: float = 0.03,
                  normal_k: int = 40,
                  huber_delta_normal: float = 0.06,
                  grid_res_axis_deg: float = 2.0,
                  kde_res_deg: float = 5.0,
                  kde_kappa: float = 20.0,
                  plane_d_bin: float = 0.03,
                  radius_win: float = 0.4,
                  radius_step: float = 0.2) -> dict:
    P0 = np.asarray(points, dtype=np.float64)
    assert P0.ndim == 2 and P0.shape[1] == 3, "points must be Nx3"

    P = _voxel_downsample(P0, voxel_size)
    idx = _knn_indices(P, k=min(20, max(5, P.shape[0]-1)))
    nbr = P[idx]
    dmean = np.mean(np.linalg.norm(nbr - P[:, None, :], axis=2), axis=1)
    mu, sigma = float(np.mean(dmean)), float(np.std(dmean) + 1e-9)
    P = P[dmean <= (mu + 2.5*sigma)]

    N = estimate_normals_irls(P, k=normal_k, huber_delta=huber_delta_normal)

    a_hat = spherical_axis_voting(N, grid_res_deg=grid_res_axis_deg,
                                  samples_per_gc=max(16, int(360/grid_res_axis_deg)))
    c_axis = line_point_fixed_dir_irls(P, a_hat, delta=3*voxel_size, iters=5)

    peaks = spherical_kde_peaks(N, res_deg=kde_res_deg, kappa=kde_kappa, n_peaks=6, nms_deg=10.0)
    n_wing, d_wing, support, thick = detect_wing_plane(P, N, a_hat, d_bin=max(plane_d_bin, voxel_size), sphere_peaks=peaks)

    s_axis, r_axis, _, _ = radius_profile(P, a_hat, c_axis, window=radius_win, step=radius_step)
    if s_axis.size == 0 or r_axis.size == 0:
        forward_sign = +1
        flags = {"nose_from_taper": False, "wing_found": n_wing is not None, "roll_ambiguous": n_wing is None}
    else:
        forward_sign = choose_forward_from_taper(s_axis, r_axis)
        flags = {"nose_from_taper": True, "wing_found": n_wing is not None, "roll_ambiguous": n_wing is None}

    x_b = _normalize(forward_sign * a_hat)

    if n_wing is not None:
        z_b = _normalize(n_wing)
        y_b = _normalize(np.cross(z_b, x_b))
        z_b = _normalize(np.cross(x_b, y_b))
    else:
        cand, best_dot = None, 1.0
        for d in peaks:
            val = abs(np.dot(d, x_b))
            if val < best_dot: best_dot, cand = val, d
        if cand is None or best_dot > math.cos(math.radians(60.0)):
            glob_up = np.array([0.0, 0.0, 1.0])
            z_b = _normalize(glob_up - np.dot(glob_up, x_b) * x_b)
        else:
            z_b = _normalize(cand - np.dot(cand, x_b) * x_b)
        y_b = _normalize(np.cross(z_b, x_b))
        z_b = _normalize(np.cross(x_b, y_b))

    R = np.column_stack([x_b, y_b, z_b])  # world <- body

    s_all = (P - c_axis) @ x_b
    if s_all.size > 0:
        s_min, s_max = float(np.min(s_all)), float(np.max(s_all))
        nose_s = s_min if forward_sign > 0 else s_max
        mask = np.abs(s_all - nose_s) <= max(0.02, 2*voxel_size)
        t = (c_axis + nose_s * x_b) if np.sum(mask) == 0 else P[mask].mean(axis=0)
    else:
        t = c_axis.copy()

    diagnostics = {
        "axis_dir": a_hat,
        "axis_point": c_axis,
        "wing_plane": None if n_wing is None else {"n": n_wing, "d": d_wing, "support": support, "thickness": thick},
        "radius_profile": {"s": s_axis, "r": r_axis},
    }
    return {"R": R, "t": t, "flags": flags, "diagnostics": diagnostics}

# ----------------------------- I/O helpers -------------------------------- #

def load_point_cloud(path: str) -> np.ndarray:
    path = str(path)
    try:
        import open3d as o3d  # optional fast path
        if path.lower().endswith((".ply", ".pcd", ".xyz")):
            pcd = o3d.io.read_point_cloud(path)
            return np.asarray(pcd.points, dtype=np.float64)
    except Exception:
        pass
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1: data = data[0:1, :]
    if data.shape[1] > 3: data = data[:, :3]
    return data

# ---------------------- Open3D visualization of flags --------------------- #

def visualize_flags_open3d(points, res, axis_span_scale=1.0, plane_rel_size=0.15):
    import open3d as o3d
    R = res["R"]; t = res["t"].reshape(3,)
    flags = res["flags"]; diag = res["diagnostics"]
    axis_dir = diag["axis_dir"]; axis_point = diag["axis_point"]
    wing = diag["wing_plane"]; rprof = diag["radius_profile"]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if len(points) > 0:
        pts = np.asarray(pcd.points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.65, 0.65, 0.65]]), (pts.shape[0], 1)))

    geoms = [pcd]

    # Nose (red sphere)
    bbox = points.max(axis=0) - points.min(axis=0) if len(points) else np.array([1,1,1], float)
    rad = max(0.02, 0.01*float(np.linalg.norm(bbox)))
    nose_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=rad)
    nose_sphere.paint_uniform_color([1, 0, 0])
    nose_sphere.translate(t)
    geoms.append(nose_sphere)

    # Body axes at nose (X red, Y green, Z blue)
    L = float(np.linalg.norm(bbox)) * 0.2 * axis_span_scale + 1e-6
    def line(origin, vec, color):
        P = np.stack([origin, origin + L*vec], axis=0)
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(P),
                                  o3d.utility.Vector2iVector([[0,1]]))
        ls.colors = o3d.utility.Vector3dVector([color])
        return ls
    geoms += [
        line(t, R[:,0], [1,0,0]),
        line(t, R[:,1], [0,1,0]),
        line(t, R[:,2], [0,0,1]),
    ]

    # Fuselage axis (white) from data span
    if len(points) > 0:
        s = (points - axis_point) @ axis_dir
        smin, smax = float(np.min(s)), float(np.max(s))
    else:
        smin, smax = -1.0, 1.0
    A0 = axis_point + smin * axis_dir
    A1 = axis_point + smax * axis_dir
    fus = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.stack([A0, A1], axis=0)),
                               o3d.utility.Vector2iVector([[0,1]]))
    fus.colors = o3d.utility.Vector3dVector([[1,1,1]])
    geoms.append(fus)

    # Wing plane (green quad) if found
    if flags.get("wing_found", False) and wing is not None:
        n = np.array(wing["n"], float); d = float(wing["d"])
        x0 = -d * n
        tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, tmp); u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u);   v = v / (np.linalg.norm(v) + 1e-12)
        h = plane_rel_size * max(1.0, float(np.linalg.norm(bbox)))
        corners = np.stack([x0 + h*u + h*v, x0 - h*u + h*v, x0 - h*u - h*v, x0 + h*u - h*v], axis=0)
        wing_mesh = o3d.geometry.TriangleMesh()
        wing_mesh.vertices = o3d.utility.Vector3dVector(corners)
        wing_mesh.triangles = o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
        wing_mesh.compute_vertex_normals()
        wing_mesh.paint_uniform_color([0.0, 0.8, 0.0])
        geoms.append(wing_mesh)

    # Radius profile markers along axis (yellow)
    s_vals = np.asarray(rprof.get("s", []))
    if s_vals.size > 0:
        centers3d = axis_point + s_vals[:,None] * axis_dir[None,:]
        rp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers3d))
        rp.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0,1.0,0.0]]), (centers3d.shape[0],1)))
        geoms.append(rp)

    # Little coordinate frame for reference
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=L*0.6)
    frame.translate(t)
    geoms.append(frame)

    print("Flags:", flags)
    o3d.visualization.draw_geometries(geoms)

# --------------------------------- Run ------------------------------------ #

if __name__ == "__main__":
    # Load points
    P = load_point_cloud(INPUT_PATH)

    # Estimate pose
    res = estimate_pose(P, voxel_size=VOXEL, normal_k=NORMAL_K)
    R, t = res["R"], res["t"]

    # Print results
    print("Rotation R (world <- body):\n", R)
    print("Translation t (world position of body origin at nose):\n", t)
    print("Flags:", res["flags"])
    print("Diagnostics keys:", list(res["diagnostics"].keys()))

    # Visualize flags in Open3D
    try:
        import open3d  # noqa: F401
    except Exception as e:
        raise SystemExit("Open3D is required for this visualization. Install with: pip install open3d") from e

    visualize_flags_open3d(P, res, axis_span_scale=1.0, plane_rel_size=0.15)
