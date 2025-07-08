import numpy as np
import pandas as pd

npz_path = "/home/femi/Benchmarking_framework/Data/Machine_learning_dataset/label/HAM_Airport_2024_08_08_movement_a320_ceo_Germany_scene0035_17.595s_filtered.npz"

data = np.load(npz_path, allow_pickle=True)

print("Keys in archive:", list(data.keys()))

points = data["points"]
labels = data["labels"]

if labels.dtype == object:
    labels = np.array([int(x) for x in labels])

print("Points shape:", points.shape)
print("Labels shape:", labels.shape)
print("Unique labels:", np.unique(labels))

df = pd.DataFrame({
    "x": points[:, 0],
    "y": points[:, 1],
    "z": points[:, 2],
    "label": labels
})
print(df.head())
