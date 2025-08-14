import os
import json
import numpy as np
import open3d as o3d

# ---------- helpers ----------
def _read_point_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"]:
        return o3d.io.read_point_cloud(path)
    raise ValueError(f"Unsupported point cloud format: {ext}")

def _normalize_model_id(s: str) -> str:
    stem = os.path.splitext(str(s))[0].lower()
    parts = stem.replace("-", "_").split("_")
    return max(parts, key=len) if parts else stem

def _load_keypoints_for_model(json_path, class_id, model_id, debug=False):
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    target_class = str(class_id)
    target_model_norm = _normalize_model_id(model_id)

    best = None
    for entry in data:
        if str(entry.get("class_id", "")) != target_class:
            continue
        eid = str(entry.get("model_id", ""))
        eid_norm = _normalize_model_id(eid)
        if eid == model_id or eid_norm == target_model_norm \
           or eid_norm in target_model_norm or target_model_norm in eid_norm:
            best = entry
            break

    if best is None:
        if debug:
            print(f"[WARN] No keypoints for class_id='{class_id}', model_id='{model_id}'.")
        return []

    kps = []
    for kp in best.get("keypoints", []):
        xyz = kp.get("xyz", None)
        kps.append({
            "xyz": np.array(xyz, dtype=float) if xyz is not None else None,
            "point_index": kp.get("pcd_info", {}).get("point_index", None),
            "semantic_id": kp.get("semantic_id", None),
        })
    if debug:
        print(f"[OK] Loaded {len(kps)} keypoints for model_id='{best.get('model_id')}'.")
    return kps

def _estimate_radius_from_knn(pcd, sample=2000, k=8, scale=1.5):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return 1e-3
    idx = np.random.choice(len(pts), size=min(sample, len(pts)), replace=False)
    kdt = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in idx:
        _, _, d = kdt.search_knn_vector_3d(pcd.points[i], k)
        if len(d) > 1:
            dists.append(np.sqrt(np.median(d[1:])))
    base = np.median(dists) if dists else 0.01
    return max(1e-6, float(base) * scale)

def _spheres_for_keypoints(pcd, keypoints, radius, kp_color=(1.0, 0.0, 0.0)):
    geoms = []
    pts_np = np.asarray(pcd.points)
    for kp in keypoints:
        if kp["xyz"] is not None:
            pos = kp["xyz"]
        elif kp["point_index"] is not None and 0 <= kp["point_index"] < len(pts_np):
            pos = pts_np[kp["point_index"]]
        else:
            continue
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.compute_vertex_normals()
        s.paint_uniform_color(kp_color)  # bright red spheres
        s.translate(pos)
        geoms.append(s)
    return geoms

# ---------- main entry ----------
def visualize_ply_with_keypoints(ply_path, json_path, class_id, model_id,
                                 sphere_scale=1.5,
                                 cloud_color=(0.0, 1.0, 0.0),  # darker red cloud
                                 kp_color=(1.0, 0.0, 0.0),     # bright red keypoints
                                 debug=False):
    """
    Draw a single PLY/PCD with its keypoints as colored spheres.
    Cloud and keypoints in red.
    """
    pcd = _read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise RuntimeError("Point cloud has no points.")

    # paint the cloud
    pcd.paint_uniform_color(cloud_color)

    kps = _load_keypoints_for_model(json_path, class_id, model_id, debug=debug)
    if not kps:
        print("[INFO] No keypoints found; showing cloud only.")
        o3d.visualization.draw_geometries([pcd], window_name=f"{model_id}")
        return

    radius = _estimate_radius_from_knn(pcd, scale=sphere_scale)
    spheres = _spheres_for_keypoints(pcd, kps, radius=radius, kp_color=kp_color)

    o3d.visualization.draw_geometries([pcd, *spheres],
                                      window_name=f"{class_id}:{model_id} (KP count: {len(spheres)})")

# ---------- Example usage ----------
if __name__ == "__main__":
    visualize_ply_with_keypoints(
        ply_path="/home/femi/Downloads/ShapeNetCore.v2.ply/02691156/1bcbb0267f5f1d53c6c0edf9d2d89150.ply",
        json_path="/home/femi/Downloads/airplane.json",
        class_id="02691156",
        model_id="1bcbb0267f5f1d53c6c0edf9d2d89150",
        sphere_scale=2.0,
        debug=True
    )
