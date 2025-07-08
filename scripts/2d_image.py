import h5py
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# --- Projection Utilities ---
def compute_valid_uv(pts: np.ndarray,
                     img_shape: tuple,
                     h_fov: tuple,
                     v_fov: tuple) -> tuple:
    """
    Returns only the (u_pix, v_pix) for points that land inside the image.
    """
    H, W = img_shape
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    r = np.linalg.norm(pts, axis=1)
    valid = r > 0

    # angles in degrees
    az = np.degrees(np.arctan2(y[valid], x[valid]))
    el = np.degrees(np.arcsin(np.clip(z[valid]/r[valid], -1.0,1.0)))

    # wrap az into [–180, +180]
    az = ((az + 180) % 360) - 180

    # linear map to [0 .. W-1] and [0 .. H-1]
    u = (az - h_fov[0]) / (h_fov[1] - h_fov[0]) * (W - 1)
    v = (el - v_fov[0]) / (v_fov[1] - v_fov[0]) * (H - 1)

    u_pix = np.round(u).astype(int)
    v_pix = np.round(v).astype(int)
    # flip vertical so high elevation is at top row
    v_pix = (H - 1) - v_pix

    # mask points inside the image
    mask = (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H)

    # return only the in‐bounds ones
    return u_pix[mask], v_pix[mask], int(mask.sum()), int((~mask).sum())


def main():
    # — YOUR FILE PATHS —
    h5_path  = "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/"\
               "HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"
    pcd_path = "/home/femi/Benchmarking_framework/Data/Aircraft_models/"\
               "HAM_Airport_2024_08_08_movement_a320_ceo_Germany_model.pcd"

    # — YOUR MEASURED FOV —
    h_fov = (-8.7, 123.9)
    v_fov = (-40.6,  12.3)

    # load once
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts_world = np.asarray(pcd.points)
    print(f"[INFO] Loaded PCD with {pts_world.shape[0]} points")

    with h5py.File(h5_path, "r") as f:
        for ts in f.keys():
            frame = f[ts]
            if 'range_image' not in frame:
                continue

            range_img = frame['range_image'][()]
            H, W = range_img.shape

            # transform to sensor frame if needed
            pts = pts_world.copy()
            if 'tf_matrix' in frame:
                tf3d   = frame['tf_matrix'][()]
                inv_tf = np.linalg.inv(tf3d)
                homo   = np.hstack([pts, np.ones((pts.shape[0],1))])
                pts    = (inv_tf @ homo.T).T[:, :3]

            # get only valid uv
            u_pix, v_pix, kept, dropped = compute_valid_uv(
                pts, (H, W), h_fov, v_fov
            )
            print(f"[DEBUG] Frame {ts}: Kept {kept}, Dropped {dropped}")

            # overlay scatter on real range image
            plt.figure(figsize=(8,5))
            plt.imshow(range_img, cmap='viridis')
            plt.scatter(u_pix, v_pix, s=1, c='r')
            plt.title(f"Frame {ts} — PCD overlaid ({kept} points)")
            plt.axis('off')
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
