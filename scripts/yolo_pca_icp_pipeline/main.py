from scripts.yolo_pca_icp_pipeline import CFG, run_pipeline



from scripts.yolo_pca_icp_pipeline.config import Config



cfg = Config(
    icp_enabled=True,
    model_pcd_path="/home/femi/Benchmarking_framework/Data/Aircraft_models/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_model.pcd",
    compare_with_h5_gt=True,
    icp_visualize=True,
)

res = run_pipeline(cfg)

print("\nSeed vs GT:", res["metrics_vs_gt_seed"])
if res["metrics_vs_gt_icp"] is not None:
    print("ICP  vs GT:", res["metrics_vs_gt_icp"])



