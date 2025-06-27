
from pathlib import Path

from scripts.pre_processing.bird_eye_view import visualize_model_bird_eye
from scripts.pre_processing.extract_aircraft_models import extract_aircraft_models
from scripts.pre_processing.visualise_before_after_zfilter import visualize_before_after
from scripts.pre_processing.extract_tf_scene import extract_pcd_and_tf
import rclpy
from scripts.pre_processing.visualise_model_scene_after_tf import visualize_overlay

<<<<<<< HEAD
bag_path = "/home/femi/Benchmarking_framework/Data/bag_files/Airbus_Airbus_13_03_2024_a320neo_2024-03-13-11-37-53"
TOPICS = ["/main/points"]
INTERVAL = 0.5
SOURCE_F = "base_link"
TARGET_F = "main_sensor"
VISUALIZE = True
EXTRACT_M = True
VIS_MODEL = True
VIS_OVERLAY = True
=======


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx, yy, zz, ww = qx*qx, qy*qy, qz*qz, qw*qw
    xy, xz, xw = qx*qy, qx*qz, qx*qw
    yz, yw, zw = qy*qz, qy*qw, qz*qw
    return np.array([
        [ ww + xx - yy - zz, 2*(xy - zw),       2*(xz + yw) ],
        [ 2*(xy + zw),       ww - xx + yy - zz, 2*(yz - xw) ],
        [ 2*(xz - yw),       2*(yz + xw),       ww - xx - yy + zz ],
    ], dtype=float)


def invert_homogeneous_matrix(M: np.ndarray) -> np.ndarray:
    R = M[0:3, 0:3]
    t = M[0:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    M_inv = np.eye(4, dtype=float)
    M_inv[0:3, 0:3] = R_inv
    M_inv[0:3, 3] = t_inv
    return M_inv


def transform_stamped_to_matrix(ts: TransformStamped) -> np.ndarray:
    t = ts.transform.translation
    q = ts.transform.rotation
    R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    M = np.eye(4, dtype=float)
    M[:3, :3] = R
    M[:3, 3] = [t.x, t.y, t.z]
    return M


def stamp_to_nanosec(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def find_frame_chain(
    transforms: list,
    source_frame: str,
    target_frame: str
) -> list:
    from collections import deque
    neighbors = {}
    for ts in transforms:
        p, c = ts.header.frame_id, ts.child_frame_id
        neighbors.setdefault(p, []).append((c, ts, False))
        neighbors.setdefault(c, []).append((p, ts, True))
    queue = deque([(source_frame, [])])
    visited = {source_frame}
    while queue:
        curr, path = queue.popleft()
        if curr == target_frame:
            return path
        for nxt, ts_msg, inv in neighbors.get(curr, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            new_path = path + [(curr, nxt, ts_msg, inv)]
            if nxt == target_frame:
                return new_path
            queue.append((nxt, new_path))
    return None


def filter_negative_z(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    mask = pts[:, 2] >= 0
    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered.colors = o3d.utility.Vector3dVector(colors[mask])
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        filtered.normals = o3d.utility.Vector3dVector(normals[mask])
    return filtered


def extract_pcd_and_tf(
    bag_path: str,
    topics: list,
    interval_sec: float,
    source_frame: str = "base_link",
    target_frame: str = "main_sensor"
) -> None:
    tf_processor = BagTfProcessor()
    tf_processor.read_tf_from_bag(bag_path)

    bag_name = Path(bag_path).stem
    out_dir = Path(bag_path).parents[1] / "sequence_from_scene" / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for topic in topics:
        print(f"Processing topic: {topic}")
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("", "")
        )
        reader.set_filter(StorageFilter(topics=[topic]))

        start_ns = None
        next_save = 0.0
        idx = 0

        while reader.has_next():
            tname, raw, time_ns = reader.read_next()
            if start_ns is None:
                start_ns = time_ns
            elapsed = (time_ns - start_ns) * 1e-9
            if elapsed + 1e-6 < next_save:
                continue

            # Deserialize point cloud message
            cloud_msg: PointCloud2 = deserialize_message(raw, PointCloud2)
            # Lookup transform at this timestamp
            stamp_ns = stamp_to_nanosec(cloud_msg.header.stamp)
            ros_time = Time(nanoseconds=stamp_ns)
            tf_stamped = tf_processor.lookup_example(
                source_frame,target_frame,  ros_time
            )

            if not tf_stamped:
                print(f"[SKIP] no TF for `{source_frame}`→`{target_frame}` at t={elapsed:.3f}s")
                continue

            # Only save scenes with an available TF
            pts = np.array(
                [[x, y, z] for x, y, z in pc2_ros2.read_points(
                    cloud_msg, field_names=("x", "y", "z"), skip_nans=True
                )],
                dtype=np.float64
            )
            raw_pcd = o3d.geometry.PointCloud()
            raw_pcd.points = o3d.utility.Vector3dVector(pts)
            filtered_pcd = filter_negative_z(raw_pcd)

            name = f"{bag_name}_scene{idx:04d}_{elapsed:.3f}s"
            raw_path = out_dir / f"{name}_raw.pcd"
            filt_path = out_dir / f"{name}_filtered.pcd"
            o3d.io.write_point_cloud(str(raw_path), raw_pcd)
            o3d.io.write_point_cloud(str(filt_path), filtered_pcd)

            # Save transform matrix
            M = transform_stamped_to_matrix(tf_stamped)
            M = np.round(M, 3)
            tf_path = out_dir / f"{name}_tf.json"
            with open(tf_path, 'w') as f:
                json.dump({"matrix": M.tolist()}, f, indent=2)

            print(f"Saved scene#{idx}: {raw_path.name}, {filt_path.name}, {tf_path.name}")

            idx += 1
            next_save += interval_sec




def extract_aircraft_models(
    bag_path: str,
    model_topic: str = "/cloud_pcd"
) -> None:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )
    model_dir = Path(bag_path).parents[1] / "Aircraft_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    while reader.has_next():
        tname, raw, _ = reader.read_next()
        if tname != model_topic: continue
        cloud_msg = deserialize_message(raw, PointCloud2)
        pts = np.array(
            [[x, y, z] for x, y, z in pc2_ros2.read_points(
                cloud_msg, field_names=("x","y","z"), skip_nans=True
            )], dtype=np.float64
        )
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
        filename = f"{bag_name}_model.pcd"
        pcd_path = model_dir / filename
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        print(f"Saved aircraft model: {pcd_path}")
        break


def visualize_overlay(
    scene_pcd_path: str,
    model_pcd_path: str,
    point_size: float = 1,
    scene_color: tuple = (0.5, 0.5, 0.5),
    cmap: str = 'viridis',
    figsize: tuple = (6, 6)
) -> None:
    import numpy as np; import open3d as o3d; import matplotlib.pyplot as plt
    scene_pts = np.asarray(o3d.io.read_point_cloud(scene_pcd_path).points)
    # derive base stem without suffix
    scene_stem = Path(scene_pcd_path).stem.replace('_filtered','').replace('_raw','')
    tf_path = Path(scene_pcd_path).parent / f"{scene_stem}_tf.json"
    with open(tf_path, 'r') as f:
        mat = np.array(json.load(f)['matrix'], dtype=float)
    model_pcd = o3d.io.read_point_cloud(model_pcd_path); model_pcd.transform(mat)
    model_pts = np.asarray(model_pcd.points)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(scene_pts[:, 0], scene_pts[:, 1], c=[scene_color], s=point_size, linewidth=0)
    sc = ax.scatter(model_pts[:, 0], model_pts[:, 1], c=model_pts[:, 2], s=point_size, cmap=cmap, linewidth=0)
    ax.set_aspect('equal', 'box'); ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_title(scene_stem)
    cb = plt.colorbar(sc, fraction=0.046, pad=0.04); cb.set_label('Height [m]')
    plt.show()


def align_and_save_models(
    bag_name: str,
    scene_files: list,
    model_pcd_path: str,
    output_parent: Path
) -> None:
    aligned_dir = output_parent / "aligned_models" / bag_name
    aligned_dir.mkdir(parents=True, exist_ok=True)
    for scene in scene_files:
        # remove suffix to match TF file
        scene_stem = scene.stem.replace('_filtered','').replace('_raw','')
        tf_file = scene.parent / f"{scene_stem}_tf.json"
        with open(tf_file, 'r') as f:
            mat = np.array(json.load(f)['matrix'], dtype=float)
        model_pcd = o3d.io.read_point_cloud(model_pcd_path); model_pcd.transform(mat)
        idx = scene.stem.split('_')[1]
        aligned_name = f"{bag_name}_scene{idx}_aligned_model.pcd"
        out_path = aligned_dir / aligned_name
        o3d.io.write_point_cloud(str(out_path), model_pcd)
        print(f"Saved aligned model: {out_path}")

def visualize_before_after(
    raw_pcd_path: str,
    filt_pcd_path: str,
    point_size: float = 1,
    cmap: str = 'viridis',
    figsize: tuple = (10, 5)
) -> None:
    """
    Display two bird’s-eye view panels: before (raw) and after (filtered).
    """
    import numpy as np; import open3d as o3d; import matplotlib.pyplot as plt
    raw_pts = np.asarray(o3d.io.read_point_cloud(raw_pcd_path).points)
    filt_pts = np.asarray(o3d.io.read_point_cloud(filt_pcd_path).points)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    sc1 = ax1.scatter(raw_pts[:, 0], raw_pts[:, 1], c=raw_pts[:, 2], s=point_size, cmap=cmap, linewidth=0)
    ax1.set_aspect('equal', 'box'); ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]')
    ax1.set_title('Raw Bird’s-Eye View'); ax1.grid(True)
    cb1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04); cb1.set_label('Height [m]')
    sc2 = ax2.scatter(filt_pts[:, 0], filt_pts[:, 1], c=filt_pts[:, 2], s=point_size, cmap=cmap, linewidth=0)
    ax2.set_aspect('equal', 'box'); ax2.set_xlabel('X [m]'); ax2.set_ylabel('Y [m]')
    ax2.set_title('Filtered Bird’s-Eye View'); ax2.grid(True)
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04); cb2.set_label('Height [m]')
    plt.tight_layout()
    plt.show()


def visualize_model_bird_eye(
    model_pcd_path: str,
    point_size: float = 1,
    cmap: str = 'viridis',
    figsize: tuple = (6, 6)
) -> None:
    """
    Display bird’s-eye view of the aircraft model.
    """
    import numpy as np; import open3d as o3d; import matplotlib.pyplot as plt
    pts = np.asarray(o3d.io.read_point_cloud(model_pcd_path).points)
    fig = plt.figure(figsize=figsize)
    sc = plt.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=point_size, cmap=cmap, linewidth=0)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('X [m]'); plt.ylabel('Y [m]'); plt.title("Aircraft Model Bird’s-Eye View")
    cb = plt.colorbar(sc, fraction=0.046, pad=0.04); cb.set_label('Height [m]')
    plt.show()

def visualize_overlap_points(
    scene_pcd_path: str,
    model_pcd_path: str,
    distance_threshold: float = 0.05,
    point_size: float = 1,
    figsize: tuple = (8, 8)
) -> None:
    """
    Compute and visualize overlapping points between a filtered scene and the transformed aircraft model.
    Points in the scene cloud within `distance_threshold` of any point in the transformed model are considered overlapping.
    """
    import numpy as np
    import open3d as o3d
    import matplotlib.pyplot as plt
    # Load filtered scene
    scene_file = Path(scene_pcd_path)
    scene_pcd = o3d.io.read_point_cloud(scene_pcd_path)
    scene_pts = np.asarray(scene_pcd.points)
    # Load corresponding TF
    # Determine TF file path
    base = scene_file.stem.replace('_filtered', '')
    #tf_path = scene_file.parent / f"{base}_tf.json"
    if not tf_path.exists() :
        raise FileNotFoundError(f"TF JSON not found: {tf_path}")

    with open(str(tf_path), 'r') as f :
        mat = np.array(json.load(f)['matrix'], dtype=float)
    # Load and transform model
    model_pcd = o3d.io.read_point_cloud(model_pcd_path)
    model_pcd.transform(mat)
    model_pts = np.asarray(model_pcd.points)
    # Build KD-tree on model
    model_tree = o3d.geometry.KDTreeFlann(model_pcd)
    overlap_mask = np.zeros(len(scene_pts), dtype=bool)
    for i, pt in enumerate(scene_pts):
        _, idxs, dists = model_tree.search_knn_vector_3d(pt, 1)
        if np.sqrt(dists[0]) <= distance_threshold:
            overlap_mask[i] = True
    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    non_ol = scene_pts[~overlap_mask]
    ax.scatter(non_ol[:,0], non_ol[:,1], c='gray', s=point_size, label='Scene Non-Overlap')
    ol_pts = scene_pts[overlap_mask]
    ax.scatter(ol_pts[:,0], ol_pts[:,1], c='red', s=point_size, label='Overlap')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title(f"Overlap: {base}")
    ax.legend()
    plt.show()
# === PyCharm Run Configuration ===
BAG_PATH = "/home/femi/Evitado/Benchmarking_framework/Data/Bag_Files/HAM_Airport_2024_08_08_movement_a320_ceo_Germany, Hamburg, 22335, Hamburg, Flughafenstr. 1-3, 22335 Hamburg, Germany, 1-3, 53.62924366258085, 10.003206739202142_2024-08-08T10-03-18"
TOPICS    = ["/main/points"]
INTERVAL  = 0.5
SOURCE_F  = "base_link"
TARGET_F  = "main_sensor"
VISUALIZE = False
EXTRACT_M  = True
VIS_MODEL  = True
VIS_OVERLAY= True
>>>>>>> 22148d534276cc132983110260cdde45e72e35da
ALIGN_MODELS = True
VIS_OVERLAP = False

if __name__ == "__main__":
    rclpy.init()

    bag_name = Path(bag_path).stem
    extract_pcd_and_tf(bag_path, TOPICS, INTERVAL, SOURCE_F, TARGET_F)
    if EXTRACT_M:
        extract_aircraft_models(bag_path)
    out_dir = Path(bag_path).parents[1] / "sequence_from_scene" / bag_name
    model_path = Path(bag_path).parents[1] / "Aircraft_models" / f"{bag_name}_model.pcd"
    scene_files = sorted(out_dir.glob(f"{bag_name}_scene????_*_filtered.pcd"))
    if VISUALIZE and scene_files:
        visualize_before_after(str(scene_files[0]).replace('_filtered', '_raw'), str(scene_files[0]))
    if VIS_MODEL:
        visualize_model_bird_eye(str(model_path))
    if VIS_OVERLAY:
        for scene in scene_files: visualize_overlay(str(scene), str(model_path))


