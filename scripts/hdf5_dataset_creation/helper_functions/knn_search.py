import numpy as np
import open3d as o3d


def label_scene_points(pcd_src: o3d.geometry.PointCloud,
                       points: np.ndarray,
                       distance_threshold: float) -> np.ndarray:
    """
    Label each point in `points` as 1 if its nearest neighbor in `pcd_src`
    is within `distance_threshold`, else 0.

    Args:
        pcd_src: Source point cloud (already transformed).
        points: (N,3) array of target points.
        distance_threshold: Threshold in meters for labeling.

    Returns:
        labels: (N,) array of uint8 labels (0 or 1).
    """
    # Build KD-tree on source cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_src)

    labels = np.zeros(points.shape[0], dtype=np.uint8)
    for i, pt in enumerate(points):
        # search for the single nearest neighbor
        k, idx_knn, dist2 = pcd_tree.search_knn_vector_3d(pt, 1)
        if k > 0:
            # dist2 is squared distance
            if np.sqrt(dist2[0]) < distance_threshold:
                labels[i] = 1
    return labels
