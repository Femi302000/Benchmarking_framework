#!/usr/bin/env python3
import copy
import h5py
import numpy as np
import open3d as o3d
import cv2

def read_with_chunk_fallback(ds, cols=None, chunk_size=100_000):
    try:
        return ds[:, cols] if cols is not None else ds[()]
    except (OSError, RuntimeError):
        parts = []
        n = ds.shape[0]
        for i in range(0, n, chunk_size):
            block = ds[i : min(i + chunk_size, n), :]
            parts.append(block if cols is None else block[:, cols])
        return np.vstack(parts)

def project_points(pts3d, K, T):
    """
    pts3d: (N,3) array in world coords
    K:      (3,3) intrinsic matrix
    T:      (4,4) world->camera transform
    returns:
      uv:    (M,2) pixel coords of points in front of cam
      z:     (M,) depths
      mask:  (N,) bool mask of valid points
    """
    N = pts3d.shape[0]
    # to homogeneous
    homo = np.hstack([pts3d, np.ones((N,1))])          # (N,4)
    cam = (T @ homo.T).T                              # (N,4)
    xyz = cam[:, :3]
    z = xyz[:,2]
    valid = z>0
    xyz = xyz[valid]
    proj = (K @ xyz.T).T                              # (M,3)
    uv = proj[:,:2] / proj[:,2:3]
    uv = np.round(uv).astype(int)
    return uv, z[valid], valid

def overlay_on_image(img, uv, mask, color=(0,0,255), radius=2):
    """
    Draws points at uv[mask] on img.
    """
    h,w = img.shape[:2]
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (u,v) in uv:
        if 0<=u<w and 0<=v<h:
            cv2.circle(canvas, (u,v), radius, color, -1)
    return canvas

def visualize_2d_overlay_all(
    h5_path: str,
    source_pcd_path: str,
    points_name: str      = "points_ground",
    tf_name:     str      = "tf_matrix",
    K:           np.ndarray = None,
    max_frames:  int      = None
):
    # load source model once
    src_model = o3d.io.read_point_cloud(source_pcd_path)

    # default intrinsics (replace with your own)
    if K is None:
        fx = fy = 500.0
        cx, cy = 320.0, 240.0
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)

    with h5py.File(h5_path, "r") as f:
        for i, (frame, grp) in enumerate(f.items()):
            if max_frames is not None and i>=max_frames:
                break

            # find your range image dataset
            names = list(grp.keys())
            range_name = next((n for n in names if "range" in n.lower()), None)
            if range_name is None or tf_name not in grp or points_name not in grp:
                continue

            range_img = grp[range_name][()]                     # H×W float or uint
            T4x4      = np.array(grp[tf_name][()], dtype=float) # (4×4)
            ds        = grp[points_name]
            pts3d     = read_with_chunk_fallback(ds)[:, :3]     # (N,3)

            # transform and project
            uv, depths, mask = project_points(pts3d, K, T4x4)

            # optionally transform the full source PCD and re‐project its pts:
            src_copy = copy.deepcopy(src_model)
            src_copy.transform(T4x4)
            src_pts = np.asarray(src_copy.points)
            uv_src, _, _ = project_points(src_pts, K, np.eye(4))

            # overlay scene points in red, source model in green
            over = overlay_on_image(range_img, uv, mask, color=(0,0,255))
            over = overlay_on_image(over, uv_src, np.ones(len(uv_src),bool), color=(0,255,0))

            # show
            cv2.imshow(f"Frame {frame}", over)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()



if __name__ == "__main__":
    H5  = "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/" \
          "HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"
    PCD = "/home/femi/Benchmarking_framework/Data/Aircraft_models/" \
          "HAM_Airport_2024_08_08_movement_a320_ceo_Germany_model.pcd"

    visualize_2d_overlay_all(
        h5_path=H5,
        source_pcd_path=PCD,
        max_frames=10
    )
