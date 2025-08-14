import open3d as o3d
import numpy as np

def fps_keypoints(pcd, K):
    pts = np.asarray(pcd.points)
    N = len(pts)
    # Initialize with a random point
    key_idxs = [np.random.randint(N)]
    # Precompute distances
    dist = np.full(N, np.inf)
    for _ in range(1, K):
        # Update distance to the selected point
        last = key_idxs[-1]
        d = np.linalg.norm(pts - pts[last], axis=1)
        dist = np.minimum(dist, d)
        # Pick the farthest point
        key_idxs.append(int(dist.argmax()))
    return pcd.select_by_index(key_idxs)

# Usage:
pcd = o3d.io.read_point_cloud("/scripts/yolo/yolo_outputs/scene_000_red_bbox_points.pcd")
fps_pts = fps_keypoints(pcd, K=30)
o3d.visualization.draw_geometries([pcd.paint_uniform_color([0.8,0.8,0.8]),
                                   fps_pts.paint_uniform_color([1,0,0])])

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

def remove_ground_and_noise(pcd, ground_thresh=0.05):
    # Remove statistical outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

    # Segment ground plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=ground_thresh,
                                             ransac_n=3,
                                             num_iterations=1000)
    # Remove ground
    pcd_no_ground = pcd.select_by_index(inliers, invert=True)
    return pcd_no_ground

def align_pcd_pca(pcd):
    pts = np.asarray(pcd.points)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # Force axis mapping: X = fuselage length, Y = wingspan, Z = height
    aligned = centered @ eigvecs
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned)
    return aligned_pcd, eigvecs, mean

def get_cluster_center(cluster_points):
    return np.mean(cluster_points, axis=0)

def detect_aircraft_keypoints(pcd):
    # Step 1: Clean
    pcd_clean = remove_ground_and_noise(pcd)

    # Step 2: Align
    aligned_pcd, eigvecs, mean = align_pcd_pca(pcd_clean)
    A = np.asarray(aligned_pcd.points)
    X, Y, Z = A[:,0], A[:,1], A[:,2]

    # Force nose in +X direction
    if np.mean(X[X > np.percentile(X, 90)]) < np.mean(X[X < np.percentile(X, 10)]):
        A[:,0] *= -1  # Flip X
        Y *= -1
        X = A[:,0]

    keypoints = {}

    # Nose / tail
    nose_region = A[X > np.percentile(X, 98)]
    tail_region = A[X < np.percentile(X, 2)]
    keypoints["nose"] = get_cluster_center(nose_region)
    keypoints["tail"] = get_cluster_center(tail_region)

    # Wings — mid-Z
    mid_z_mask = (Z > np.percentile(Z, 40)) & (Z < np.percentile(Z, 60))
    wing_region = A[mid_z_mask]
    wing_Y = wing_region[:, 1]

    left_wing_region = wing_region[wing_Y < np.percentile(wing_Y, 5)]
    right_wing_region = wing_region[wing_Y > np.percentile(wing_Y, 95)]
    keypoints["wing_left"] = get_cluster_center(left_wing_region)
    keypoints["wing_right"] = get_cluster_center(right_wing_region)

    # Landing gear — low Z
    low_z_mask = Z < np.percentile(Z, 5)
    gear_region = A[low_z_mask]
    left_gear = gear_region[gear_region[:,1] < 0]
    right_gear = gear_region[gear_region[:,1] > 0]
    keypoints["gear_left"] = get_cluster_center(left_gear)
    keypoints["gear_right"] = get_cluster_center(right_gear)

    # Engines — near center Y, mid-Z
    engine_mask = (Z > np.percentile(Z, 40)) & (Z < np.percentile(Z, 60)) & \
                  (np.abs(Y) < np.percentile(np.abs(Y), 30))
    engine_region = A[engine_mask]
    left_engine = engine_region[engine_region[:,1] < 0]
    right_engine = engine_region[engine_region[:,1] > 0]
    if len(left_engine) > 0 and len(right_engine) > 0:
        keypoints["engine_left"] = get_cluster_center(left_engine)
        keypoints["engine_right"] = get_cluster_center(right_engine)

    # Step 3: Map back to original coordinates
    for k in keypoints:
        keypoints[k] = keypoints[k] @ eigvecs.T + mean

    return keypoints


def visualize_keypoints(pcd, keypoints, sphere_size=0.3):
    geometries = [pcd]
    color_map = {
        "nose": [0,0,1],
        "tail": [0,1,1],
        "wing_left": [0,1,0],
        "wing_right": [0,1,0],
        "gear_left": [1,1,0],
        "gear_right": [1,1,0],
        "engine_left": [1,0,1],
        "engine_right": [1,0,1]
    }
    for name, coord in keypoints.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)
        sphere.paint_uniform_color(color_map.get(name, [0,0,0]))
        sphere.translate(coord)
        geometries.append(sphere)
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    in_path = "/scripts/yolo/yolo_outputs/scene_000_red_bbox_points.pcd"
    pcd = o3d.io.read_point_cloud(in_path)
    keypoints = detect_aircraft_keypoints(pcd)
    for k, v in keypoints.items():
        print(f"{k}: {v}")
    visualize_keypoints(pcd, keypoints)




# import open3d as o3d
#
# pcd = o3d.io.read_point_cloud("clean_no_ground.pcd")
# print(pcd)  # shows point count and if colors are present
#
# o3d.visualization.draw_geometries([pcd],
#                                   window_name="Clustered Point Cloud",
#                                   width=1200,
#                                   height=800,
#                                   point_show_normal=False)

