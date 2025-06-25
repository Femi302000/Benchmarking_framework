import os
import glob
import numpy as np
import open3d as o3d

def get_base_dir():
    # Determine the base directory by looking one level up, then 'Data/Machine_learning_dataset'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, 'Data', 'Machine_learning_dataset')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return data_dir


def visualize_point_cloud(pcd, window_name="Open3D", background_color=(0, 0, 0)):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    opt.point_size = 2.0
    vis.run()
    vis.destroy_window()


def label_point_clouds(base_dir=None, target_label=1):
    if base_dir is None:
        base_dir = get_base_dir()

    scene_dir = os.path.join(base_dir, "scene")
    label_dir = os.path.join(base_dir, "label")
    result_dir = os.path.join(base_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    pcd_paths = glob.glob(os.path.join(scene_dir, "*.pcd"))
    txt_paths = glob.glob(os.path.join(label_dir, "*.txt"))

    pcd_bases = {os.path.splitext(os.path.basename(p))[0] for p in pcd_paths}
    txt_bases = {os.path.splitext(os.path.basename(t))[0] for t in txt_paths}
    common = pcd_bases & txt_bases

    for base_name in sorted(common):
        pcd_file = os.path.join(scene_dir, f"{base_name}.pcd")
        label_file = os.path.join(label_dir, f"{base_name}.txt")
        print(f"\nProcessing {base_name}...")

        # Load point cloud
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points)
        num_points = points.shape[0]

        # Load label data: x,y,z,label per row
        data = np.loadtxt(label_file)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        coords = data[:, :3]
        labels = data[:, 3].astype(int)
        print(f"Loaded {len(labels)} labeled points from {label_file}")

        # Build KD-tree for matching
        kd_tree = o3d.geometry.KDTreeFlann(pcd)

        # Initialize all colors to white
        colors = np.ones((num_points, 3))

        # Match each labeled point and color by label
        for coord, lab in zip(coords, labels):
            [_, idx, _] = kd_tree.search_knn_vector_3d(coord, 1)
            i = idx[0]
            if lab == target_label:
                colors[i] = [1.0, 0.0, 0.0]  # red
            # else: keep as white or customize other labels here

        # Assign and save
        pcd.colors = o3d.utility.Vector3dVector(colors)
        out_path = os.path.join(result_dir, f"{base_name}.pcd")
        o3d.io.write_point_cloud(out_path, pcd)
        print(f"Saved labeled point cloud to {out_path}")

        # Visualize
        print(f"Visualizing {base_name}...")
        result_pcd = o3d.io.read_point_cloud(out_path)
        visualize_point_cloud(result_pcd, window_name=f"Vis: {base_name}")

if __name__ == "__main__":
    label_point_clouds()
