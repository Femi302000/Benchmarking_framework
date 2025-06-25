import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

# Parameters (set your file paths and DBSCAN settings here)
INPUT_PCD     = "/home/femi/Evitado/Benchmarking_framework/Data/sequence_from_scene/HAM_Bag/main_points_0000_0.000s.pcd"
FILTERED_PCD  = "/home/femi/Evitado/Benchmarking_framework/Data/sequence_from_scene/HAM_Bag/filtered.pcd"
CLUSTERED_PCD = "/home/femi/Evitado/Benchmarking_framework/Data/sequence_from_scene/HAM_Bag/clustered.pcd"

# DBSCAN parameters
eps        = 3       # neighborhood radius
min_points = 5       # minimum points to form a cluster
voxel_size = 0.01    # downsampling voxel size (0 = no downsampling)

# BEV / YOLO parameters
GRID_SIZE = 0.2      # meters per pixel
X_BOUNDS  = (-20, +20)
Y_BOUNDS  = (-20, +20)
Z_MAX     = 20.0     # clip heights [0..Z_MAX]
COUNT_MAX = 100      # clip counts [0..COUNT_MAX]
YOLO_MODEL = "yolov8n.pt"  # or your finetuned weight

def filter_negative_z(input_path: str, output_path: str) -> o3d.geometry.PointCloud:

    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)
    mask = points[:, 2] >= 0
    filtered_points = points[mask]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals[mask])
    o3d.io.write_point_cloud(output_path, filtered_pcd)
    print(f"Filtered point cloud saved to: {output_path} ({len(filtered_points)} points)")
    return filtered_pcd

def preprocess_downsample(pcd: o3d.geometry.PointCloud, size: float) -> o3d.geometry.PointCloud:
    # ... your existing code unchanged ...
    if size > 0:
        down = pcd.voxel_down_sample(voxel_size=size)
        print(f"Downsampled to {len(down.points)} points using voxel_size={size}")
        return down
    return pcd

def cluster_dbscan(pcd: o3d.geometry.PointCloud,
                   eps: float,
                   min_points: int,
                   output_path: str = None) -> o3d.geometry.PointCloud:
    # ... your existing code unchanged ...
    points = np.asarray(pcd.points)
    print(f"Clustering {len(points)} points with eps={eps}, min_points={min_points}")
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    unique_labels = np.unique(labels)
    num_clusters = int((unique_labels.max() + 1) if unique_labels.size > 0 else 0)
    print(f"Detected {num_clusters} clusters, labels range {labels.min()} to {labels.max()}")
    cmap = plt.get_cmap("tab20")
    frac = labels.astype(float) / (labels.max() if labels.max() > 0 else 1)
    colors = cmap(frac)
    colors[labels < 0] = [0.5, 0.5, 0.5, 1.0]
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    if output_path:
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Clustered point cloud saved to: {output_path}")
    return pcd



def show_pcd(pcd: o3d.geometry.PointCloud, window_name: str = "Point Cloud"):
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

if __name__ == "__main__":
    # 1) Filter
    filtered_pcd = filter_negative_z(INPUT_PCD, FILTERED_PCD)

    # 2) Downsample
    down_pcd = preprocess_downsample(filtered_pcd, voxel_size)

    # 3) Visualize filtered
    print("Displaying filtered point cloud…")
    show_pcd(down_pcd, "Filtered Point Cloud")

    # 4) Cluster
    clustered_pcd = cluster_dbscan(down_pcd, eps, min_points, CLUSTERED_PCD)

    # 5) Visualize clustered
    print("Displaying clustered point cloud…")
    show_pcd(clustered_pcd, "Clustered Point Cloud")




