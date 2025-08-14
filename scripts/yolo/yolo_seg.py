import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# ----- Configuration -----
img_path = "/home/femi/Benchmarking_framework/scripts/filtered_ground_after_yolo.png"
yolo_weights = "yolov8n.pt"
conf_threshold = 0.05
iou_threshold = 0.2

# ----- 1. Load and verify image -----
assert os.path.isfile(img_path), f"File not found: {img_path}"
img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ----- 2. Load YOLO model -----
yolo = YOLO(yolo_weights)

# ----- 3. Run inference -----
results = yolo.predict(source=img_rgb, conf=conf_threshold, iou=iou_threshold)

# ----- 4. Find highest-scoring aircraft detection -----
best_score = 0.0
best_box = None
# Identify aircraft-related classes in the YOLO model
aircraft_class_idxs = [
    idx for idx, name in yolo.names.items()
    if 'aircraft' in name.lower() or 'airplane' in name.lower()
]

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    for box, cls, score in zip(boxes, classes, scores):
        if cls in aircraft_class_idxs and score > best_score:
            best_score = score
            best_box = box

assert best_box is not None, "No aircraft detected"

# ----- 5. Draw bounding box -----
x1, y1, x2, y2 = map(int, best_box)
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img_rgb)
rect = patches.Rectangle(
    (x1, y1), x2 - x1, y2 - y1,
    linewidth=2, edgecolor='red', facecolor='none'
)
ax.add_patch(rect)
ax.set_title(f"YOLO Aircraft Detection (score {best_score:.2f})")
ax.axis('off')

# ----- 6. Display & Save -----
plt.show()
# ----- 7. Crop the image to the bounding box -----
# Use the original RGB image (or BGR if you prefer)
crop_rgb = img_rgb[y1:y2, x1:x2]

# Optional: convert back to BGR if you want to use cv2.imwrite
crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

# ----- 8. Save the cropped image -----
crop_output_path = "aircraft_crop.png"
cv2.imwrite(crop_output_path, crop_bgr)
# print(f"Saved cropped aircraft image to {crop_output_path}")
#
#
# # Optional: save the result image
# output_path = "yolo_aircraft_bbox.png"
# fig.savefig(output_path, bbox_inches='tight')
# print(f"Saved annotated image to {output_path}")
# # ----- After your YOLO detection and best_box has been set -----
# # from segment_anything import sam_model_registry, SamPredictor
# # import torch
# #
# # # ----- 1. Load SAM v2 -----
# # # Replace with the path to your SAM checkpoint, e.g. "sam_v2_0.pt"
# # sam_checkpoint = "/home/femi/Downloads/sam_vit_h.pth"
# # model_type = "vit_h"  # or vit_l, vit_b, etc.
# #
# # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# # sam.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# # predictor = SamPredictor(sam)
# #
# # # ----- 2. Prepare the image -----
# # # predictor expects an HxWx3 RGB uint8 array
# # predictor.set_image(img_rgb)
# #
# # # ----- 3. Create the box prompt -----
# # # best_box is [x1, y1, x2, y2]
# # # ----- After predictor.set_image(img_rgb) -----
# # import numpy as np
# #
# # # Prepare the box prompt as an (1,4) float32 array
# # box_array = np.array([[x1, y1, x2, y2]], dtype=np.float32)
# #
# # # Run SAM inference
# # masks, scores, logits = predictor.predict(
# #     point_coords=None,
# #     point_labels=None,
# #     box=box_array,
# #     multimask_output=False
# # )
# #
# # # masks is now an array of shape (1, H, W)
# # mask = masks[0]
# #
# # # masks will be a boolean array of shape (1, H, W)
# # mask = masks[0]
# #
# # # ----- 5. Visualize the mask -----
# # fig, ax = plt.subplots(figsize=(8, 6))
# # ax.imshow(img_rgb)
# # # Overlay the mask in semi-transparent red
# # ax.imshow(mask, cmap='Reds', alpha=0.4)
# # # Draw the bounding box
# # rect = patches.Rectangle(
# #     (x1, y1), x2 - x1, y2 - y1,
# #     linewidth=2, edgecolor='cyan', facecolor='none'
# # )
# # ax.add_patch(rect)
# # ax.set_title(f"SAM v2 Segmentation (score {best_score:.2f})")
# # ax.axis('off')
# # plt.show()
# #
# # # ----- 6. (Optional) Save mask as PNG -----
# # import numpy as np
# # from PIL import Image
# #
# # mask_img = (mask.astype(np.uint8) * 255)
# # Image.fromarray(mask_img).save("aircraft_mask.png")
# # print("Saved mask to ircraft_mask.png")
from ultralytics import YOLO
import cv2, numpy as np
from pathlib import Path

img_path = "/home/femi/Benchmarking_framework/scripts/pre_processing/overlaysss/scene_000_synthetic_rgb_range.png"
weights  = "yolov8x-seg.pt"
conf_threshold = 0.10    # slightly higher than 0.005 to avoid noise
iou_threshold  = 0.5     # a more typical NMS IoU
run_proj = "detect"
run_name = "aircraft_crops_seg"

model = YOLO(weights)
aircraft_cls = [i for i, n in model.names.items() if "airplane" in n.lower() or "aircraft" in n.lower()]
print("Aircraft class indices:", aircraft_cls)

def run_and_report(classes_filter):
    res = model.predict(
        source=img_path,
        conf=conf_threshold,
        iou=iou_threshold,
        classes=classes_filter,          # None => all classes
        imgsz=1280,                      # bigger for small objects
        retina_masks=True,               # higher-res masks
        save=True,
        project=run_proj,
        name=run_name,
        exist_ok=True,
        verbose=False,
    )
    print(f"Results saved to {run_proj}/{run_name}")

    total_boxes = 0
    total_masks = 0
    for r in res:
        nb = 0 if r.boxes is None else len(r.boxes)
        nm = 0 if (r.masks is None or r.masks.data is None) else len(r.masks.data)
        total_boxes += nb
        total_masks += nm
    print(f"[classes={classes_filter}] boxes={total_boxes}, masks={total_masks}")
    return res

# 1) Try with aircraft-only
results = run_and_report(aircraft_cls if aircraft_cls else None)

# If nothing found, try without filter to confirm model is working on this image
if all((r.boxes is None or len(r.boxes) == 0) for r in results):
    print("No aircraft found — trying without class filter to verify detections…")
    results = run_and_report(None)

# Save segmented crops for whatever was detected in `results`
seg_out_dir = Path(run_proj) / run_name / "seg_crops"
seg_out_dir.mkdir(parents=True, exist_ok=True)

img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
H, W = img_bgr.shape[:2]
saved = 0

for r_idx, r in enumerate(results):
    if r.boxes is None or len(r.boxes) == 0 or r.masks is None:
        continue
    xyxy  = r.boxes.xyxy.cpu().numpy().astype(int)
    cls   = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy().astype(float)
    masks = r.masks.data.cpu().numpy()  # [N, h, w]

    for i, (box, c, s) in enumerate(zip(xyxy, cls, confs)):
        # if you still want only aircraft, uncomment:
        # if aircraft_cls and c not in aircraft_cls: continue

        m = (masks[i] * 255).astype(np.uint8)
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        mask3 = cv2.merge([m, m, m])
        masked = cv2.bitwise_and(img_bgr, mask3)
        crop_bgr   = masked[y1:y2, x1:x2]
        crop_alpha = m[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue

        crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
        crop_bgra[:, :, 3] = crop_alpha
        cls_name = model.names.get(int(c), str(int(c)))
        out_path = seg_out_dir / f"seg_{r_idx}_{i}_{cls_name}_{s:.2f}.png"
        cv2.imwrite(str(out_path), crop_bgra)
        saved += 1

print(f"Saved {saved} segmented crops to {seg_out_dir}")
