import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -------------------------------
# Config
# -------------------------------
RANGE_IMAGE_PATH = "/home/femi/Benchmarking_framework/scripts/pre_processing/3_channel/scene_000_range.png"   # grayscale 16-bit or 8-bit depth/range image
NUM_CLUSTERS = 4                       # number of depth clusters

# -------------------------------
# Load range image
# -------------------------------
                  # number of depth clusters

# -------------------------------
# Load range image
# -------------------------------
img = cv2.imread(RANGE_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Could not read {RANGE_IMAGE_PATH}")

print("Image shape:", img.shape)

# If image has 3 channels, convert to grayscale
if len(img.shape) == 3:
    # Assumes depth stored in one channel (take first, or average)
    depth = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    depth = img

# Normalize to [0,1] for clustering
depth = depth.astype(np.float32)
depth_norm = cv2.normalize(depth, None, 0, 1.0, cv2.NORM_MINMAX)

# Flatten depth values for clustering
h, w = depth.shape
flat_depth = depth_norm.reshape(-1, 1)

# -------------------------------
# KMeans clustering based on depth
# -------------------------------
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init=10)
labels = kmeans.fit_predict(flat_depth)

# Reshape labels to image size
clusters = labels.reshape(h, w)

# -------------------------------
# Visualize
# -------------------------------
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Original Range Image")
plt.imshow(depth, cmap='jet')
plt.colorbar(label="Depth")

plt.subplot(1,2,2)
plt.title(f"Depth Clusters (k={NUM_CLUSTERS})")
plt.imshow(clusters, cmap='tab20')
plt.colorbar(label="Cluster ID")
plt.show()
