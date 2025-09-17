from __future__ import annotations
import argparse
from .config import Config
from .pipeline import run_pipeline


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run LiDAR seg + PCA targeting pipeline")
    p.add_argument("--h5", dest="h5_path", required=False)
    p.add_argument("--scene", dest="scene_id", default="scene_000")
    p.add_argument("--out", dest="output_dir", default="seg_outputs")
    p.add_argument("--no-seg", dest="run_segmentation", action="store_false")
    p.add_argument("--weights", dest="yolo_weights", default="yolov8s-seg.pt")
    p.add_argument("--conf", dest="conf", type=float, default=0.15)
    p.add_argument("--iou", dest="iou", type=float, default=0.50)
    p.add_argument("--imgsz", dest="img_size", type=int, default=1280)
    p.add_argument("--dilate-mask", dest="dilate_px_mask", type=int, default=1)
    p.add_argument("--pixel", dest="pixel_size", type=float, default=0.02)
    p.add_argument("--cluster-eps", dest="cluster_eps", type=float, default=0.20)
    p.add_argument("--cluster-min", dest="cluster_min_points", type=int, default=10)
    p.add_argument("--align", dest="align_to", choices=["+X","-X"], default="+X")
    p.add_argument("--denoise", dest="denoise_method", choices=["statistical","radius","none"], default="statistical")
    return p


def main():
    args = make_parser().parse_args()
    cfg_kwargs = {k: v for k, v in vars(args).items() if v is not None}
    cfg = Config(**cfg_kwargs)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()