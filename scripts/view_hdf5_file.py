import h5py
import numpy as np
import matplotlib.pyplot as plt

h5_path = "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/" \
          "HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"

with h5py.File(h5_path, "r") as f:
    frames = list(f.keys())
    print(f"Found {len(frames)} frames, e.g.: {frames[:5]}")

    # Peek at the contents of the first frame
    ts0 = frames[0]
    print(f"Contents of frame {ts0}: {list(f[ts0].keys())}")

    # Now find any dataset in ts0 whose name contains "range"
    ds_names = list(f[ts0].keys())
    range_name = next((n for n in ds_names if 'range' in n.lower()), None)
    ambient_name = next((n for n in ds_names if 'ambient' in n.lower()), None)
    if not range_name or not ambient_name:
        raise RuntimeError(f"Couldn't find both 'range' and 'ambient' datasets in frame {ts0}")

    print(f"Using datasets: range → '{range_name}', ambient → '{ambient_name}'")

    # Load them
    range_img   = f[ts0][range_name][()]    # e.g. shape (H, W)
    ambient_img = f[ts0][ambient_name][()]  # e.g. shape (H, W)

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

im0 = axes[0].imshow(range_img, cmap='viridis')
axes[0].set_title(f"Range (“{range_name}”)")
axes[0].axis('off')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Distance')

im1 = axes[1].imshow(ambient_img, cmap='gray')
axes[1].set_title(f"Ambient (“{ambient_name}”)")
axes[1].axis('off')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Ambient level')

plt.tight_layout()
plt.show()
