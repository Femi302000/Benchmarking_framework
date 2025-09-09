import cv2
import numpy as np
from skimage.morphology import skeletonize, medial_axis
import matplotlib.pyplot as plt

# Load image (grayscale)
img = cv2.imread("/home/femi/Benchmarking_framework/scripts/yolo/seg_outputs/scene_000_seg_crop.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.png not found")

# 1) Make a clean binary mask (tweak threshold + morphology as needed)
_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

# 2) Skeletonize (Zhang–Suen style via skimage)
skel = skeletonize((bw>0)).astype(np.uint8)*255

# (Optional) Medial axis with distance transform (nice for centerlines)
medial, dist = medial_axis((bw>0), return_distance=True)
medial = medial.astype(np.uint8)*255

# Visualize
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("Input"); plt.imshow(img, cmap="gray"); plt.axis("off")
plt.subplot(1,3,2); plt.title("Binary"); plt.imshow(bw, cmap="gray"); plt.axis("off")
plt.subplot(1,3,3); plt.title("Skeleton"); plt.imshow(skel, cmap="gray"); plt.axis("off")
plt.tight_layout(); plt.show()
