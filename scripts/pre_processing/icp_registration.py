import json
import numpy as np
import open3d as o3d

source = o3d.io.read_point_cloud("/home/femi/Benchmarking_framework/Data/Aircraft_models/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_model.pcd")
target = o3d.io.read_point_cloud("/home/femi/Benchmarking_framework/Data/sequence_from_scene/HAM_Airport_2024_08_08_movement_a320_ceo_Germany/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_scene0000_8.996s_filtered.pcd")



voxel_size = 0.1

source_down = source.voxel_down_sample(voxel_size)
target_down = target.voxel_down_sample(voxel_size)
radius_normal = voxel_size * 2
for pc in (source_down, target_down):
    pc.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=60)
    )

with open("/home/femi/Benchmarking_framework/Data/sequence_from_scene/HAM_Airport_2024_08_08_movement_a320_ceo_Germany/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_scene0000_8.996s_tf.json", "r") as f:
    data = json.load(f)
init_transform = np.array(data["matrix"], dtype=float)

threshold = voxel_size * 2
icp_res = o3d.pipelines.registration.registration_icp(
    source_down,
    target_down,
    max_correspondence_distance=threshold,
    init=init_transform,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
)
# icp_res = o3d.pipelines.registration.registration_icp(
#     source_down,
#     target_down,
#     max_correspondence_distance=threshold,
#     init=init_transform,
#     # Use point-to-point instead of point-to-plane:
#     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
# )

print("init _transform:\n",init_transform)
print("Δ-transform from ICP refinement:\n", icp_res.transformation)

T_total = icp_res.transformation @ init_transform
print("Total transform (ICP × init):\n", T_total)

source.transform(T_total)
o3d.io.write_point_cloud("aligned_source.pcd", source)
print("Wrote aligned_source.pcd")

source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries(
    [source, target],
    window_name="ICP with JSON-loaded Init",
    width=800, height=600
)
