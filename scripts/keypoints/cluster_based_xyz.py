import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


def _nearest_point_to(P: np.ndarray, target: np.ndarray, idxs: np.ndarray) -> int:
    """Return global index of the point in P[idxs] nearest to target (L2 in XYZ)."""
    if idxs.size == 0:
        return -1
    diffs = P[idxs] - target[None, :]
    j = np.argmin(np.einsum("ij,ij->i", diffs, diffs))
    return int(idxs[j])


def generate_keypoints(
    pcd: o3d.geometry.PointCloud,
    eps_xy: float = 0.35,
    min_xy: int = 40,
    z_bin_size: float = 0.5,
    require_level_pair: bool = True,
):
    """
    1) Cluster points in XY using DBSCAN.
    2) Assign each cluster to a Z 'level' via floor(meanZ / z_bin_size).
    3) Optionally keep only levels that have >= 2 clusters (require_level_pair).
    4) Keypoints:
       - cluster keypoints: nearest real point to each cluster centroid (for kept levels)
       - level keypoints:   nearest real point to the centroid over all points in that level
    """
    P = np.asarray(pcd.points)
    if P.size == 0:
        return o3d.geometry.PointCloud(), o3d.geometry.PointCloud(), {}

    # Split coordinates
    Z = P[:, 2]
    XY = P[:, :2]

    # --- 1) DBSCAN in XY
    labels = DBSCAN(eps=eps_xy, min_samples=min_xy).fit_predict(XY)
    cluster_ids = [lab for lab in np.unique(labels) if lab != -1]
    if len(cluster_ids) == 0:
        # No clusters found; return empties but still include meta
        meta = {
            "labels": labels,
            "cluster_ids": [],
            "cluster_centroids": {},
            "cluster_keypoint_idxs": [],
            "cluster_to_level": {},
            "level_centroids": {},
            "level_keypoint_idxs": [],
            "kept_levels": set(),
            "z_bin_size": z_bin_size,
            "eps_xy": eps_xy,
            "min_xy": min_xy,
            "require_level_pair": require_level_pair,
        }
        return o3d.geometry.PointCloud(), o3d.geometry.PointCloud(), meta

    # Cache indices per cluster to avoid repeated np.where calls
    cluster_to_indices = {lab: np.where(labels == lab)[0] for lab in cluster_ids}

    # --- 2) Centroids and Z-levels per cluster
    if z_bin_size <= 0:
        raise ValueError("z_bin_size must be > 0")

    cl_centroids = {lab: P[idxs].mean(axis=0) for lab, idxs in cluster_to_indices.items()}
    cl_mean_z = {lab: Z[cluster_to_indices[lab]].mean() for lab in cluster_ids}
    cl_level = {lab: int(np.floor(cl_mean_z[lab] / z_bin_size)) for lab in cluster_ids}

    # Build level -> point indices
    level_to_point_idxs = {}
    for lab in cluster_ids:
        lev = cl_level[lab]
        level_to_point_idxs.setdefault(lev, []).extend(cluster_to_indices[lab].tolist())

    # --- 3) Keep only good levels if required
    if require_level_pair:
        # levels with >= 2 clusters
        level_to_cluster_count = {}
        for lab in cluster_ids:
            lev = cl_level[lab]
            level_to_cluster_count[lev] = level_to_cluster_count.get(lev, 0) + 1
        kept_levels = {lev for lev, cnt in level_to_cluster_count.items() if cnt >= 2}
    else:
        kept_levels = set(level_to_point_idxs.keys())

    # --- 4A) Cluster keypoints for kept levels
    # nearest *real* point in each cluster to its centroid
    kp_cluster_idxs = []
    cl_to_kp = {}
    for lab in cluster_ids:
        if cl_level[lab] not in kept_levels:
            continue
        idxs = cluster_to_indices[lab]
        centroid = cl_centroids[lab]
        kp_idx = _nearest_point_to(P, centroid, idxs)
        if kp_idx >= 0:
            kp_cluster_idxs.append(kp_idx)
            cl_to_kp[lab] = kp_idx

    # --- 4B) Level keypoints (one per kept level)
    kp_level_idxs = []
    level_centroids = {}
    for lev, idxs_list in level_to_point_idxs.items():
        if lev not in kept_levels:
            continue
        idxs = np.asarray(idxs_list, dtype=int)
        centroid = P[idxs].mean(axis=0)
        level_centroids[lev] = centroid
        kp_idx = _nearest_point_to(P, centroid, idxs)
        if kp_idx >= 0:
            kp_level_idxs.append(kp_idx)

    # --- 5) Build output point clouds
    kp_clusters = o3d.geometry.PointCloud()
    if len(kp_cluster_idxs) > 0:
        kp_clusters.points = o3d.utility.Vector3dVector(P[kp_cluster_idxs])
        rng = np.random.default_rng(7)
        kp_clusters.colors = o3d.utility.Vector3dVector(
            rng.random((len(kp_cluster_idxs), 3))
        )

    kp_levels = o3d.geometry.PointCloud()
    if len(kp_level_idxs) > 0:
        kp_levels.points = o3d.utility.Vector3dVector(P[kp_level_idxs])
        kp_levels.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[1.0, 0.0, 0.0]]), (len(kp_level_idxs), 1))
        )

    meta = {
        "labels": labels,
        "cluster_ids": cluster_ids,
        "cluster_centroids": cl_centroids,
        "cluster_keypoint_idxs": kp_cluster_idxs,
        "cluster_to_level": cl_level,
        "level_centroids": level_centroids,
        "level_keypoint_idxs": kp_level_idxs,
        "kept_levels": kept_levels,
        "z_bin_size": z_bin_size,
        "eps_xy": eps_xy,
        "min_xy": min_xy,
        "require_level_pair": require_level_pair,
    }
    return kp_clusters, kp_levels, meta


def pointcloud_to_spheres(pcd: o3d.geometry.PointCloud, radius: float = 0.05):
    """Convert each point in a small point cloud to a colored sphere mesh for visualization."""
    spheres = []
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return spheres
    cols = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(pts), 3))
    for p, c in zip(pts, cols):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color(c.tolist())
        sphere.translate(p.tolist())
        spheres.append(sphere)
    return spheres


if __name__ == "__main__":
    in_path = "/home/femi/Benchmarking_framework/scripts/data_dir/scene_003_red_bbox_points.pcd"
    pcd = o3d.io.read_point_cloud(in_path)

    kp_clusters, kp_levels, meta = generate_keypoints(
        pcd, eps_xy=0.35, min_xy=40, z_bin_size=0.5, require_level_pair=True
    )
    kp_final = o3d.geometry.PointCloud()
    if len(kp_levels.points) > 0:
        kp_final += kp_levels
    if len(kp_clusters.points) > 0:
        kp_final += kp_clusters

    o3d.io.write_point_cloud("keypoints_final.pcd", kp_final, write_ascii=True)
    print(f"final keypoints: {len(kp_final.points)} saved to keypoints_final.pcd")

    o3d.io.write_point_cloud("keypoints_clusters.pcd", kp_clusters, write_ascii=True)
    o3d.io.write_point_cloud("keypoints_levels.pcd", kp_levels, write_ascii=True)
    print(f"cluster keypoints: {len(kp_clusters.points)} | level keypoints: {len(kp_levels.points)}")

    # Visualize: base cloud in light gray + big spheres at keypoint locations
    base = o3d.geometry.PointCloud(pcd)
    base.paint_uniform_color([0.8, 0.8, 0.8])

    sphere_kp_levels = pointcloud_to_spheres(kp_levels, radius=0.5)
    sphere_kp_clusters = pointcloud_to_spheres(kp_clusters, radius=0.5)

    o3d.visualization.draw_geometries([base] + sphere_kp_levels + sphere_kp_clusters)
    # Merge base point cloud + sphere meshes into a single PointCloud
    merged_pcd = o3d.geometry.PointCloud()

    # Add base point cloud
    merged_pcd += base

    # Convert each sphere mesh to point cloud and add
    for sphere in sphere_kp_levels + sphere_kp_clusters:
        sphere_pcd = sphere.sample_points_uniformly(number_of_points=500)  # sample mesh to points
        merged_pcd += sphere_pcd

    # Save the merged result
    o3d.io.write_point_cloud("keypoints_visualization.pcd", merged_pcd, write_ascii=True)
    print(f"Saved combined point cloud with {len(merged_pcd.points)} points to keypoints_visualization.pcd")

