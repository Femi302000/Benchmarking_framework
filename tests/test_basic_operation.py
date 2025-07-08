#!/usr/bin/env python3
import numpy as np
import rosbag2_py
from rosbag2_py import StorageFilter, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2, PointField
from scripts.pre_processing.pixel_shift import extract_pixel_shift_by_row_field
import matplotlib.pyplot as plt

# Map ROS PointField datatypes to NumPy
ROS2_TO_NUMPY = {
    PointField.INT8:   ('i1', 1),
    PointField.UINT8:  ('u1', 1),
    PointField.INT16:  ('i2', 2),
    PointField.UINT16: ('u2', 2),
    PointField.INT32:  ('i4', 4),
    PointField.UINT32: ('u4', 4),
    PointField.FLOAT32:('f4', 4),
    PointField.FLOAT64:('f8', 8),
}

def build_dtype_from_fields(fields, point_step, is_bigendian):
    """
    Build a NumPy dtype for a PointCloud2 from its fields[] description.
    """
    names = []
    formats = []
    offsets = []
    for f in fields:
        if f.datatype not in ROS2_TO_NUMPY:
            raise ValueError(f"Unsupported PointField datatype {f.datatype}")
        np_fmt, size = ROS2_TO_NUMPY[f.datatype]
        # if count > 1, represent as a tuple-array
        if f.count != 1:
            np_fmt = (np_fmt, f.count)
        names.append(f.name)
        formats.append(np_fmt)
        offsets.append(f.offset)
    return np.dtype({
        'names':    names,
        'formats':  formats,
        'offsets':  offsets,
        'itemsize': point_step,
        'aligned':  False,
    })

def cloud2_to_array_pure(cloud_msg: PointCloud2) -> np.ndarray:
    """
    Convert a ROS2 PointCloud2 into a structured NumPy array of shape (H, W)
    using only the message’s fields, point_step, and raw data buffer.
    """
    H, W = cloud_msg.height, cloud_msg.width
    dtype = build_dtype_from_fields(cloud_msg.fields,
                                    cloud_msg.point_step,
                                    cloud_msg.is_bigendian)
    # Interpret the raw data buffer directly
    arr1d = np.frombuffer(cloud_msg.data, dtype=dtype)
    # Reshape into (H, W)
    try:
        return arr1d.reshape((H, W))
    except ValueError as e:
        raise RuntimeError(f"Expected {H*W} points but got {arr1d.size}") from e

def destagger(field: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """
    Apply per-row cyclic shifts to destagger a 2D lidar field.
    """
    H, W = field.shape
    out = np.zeros_like(field)
    for u, s in enumerate(shifts):
        out[u, :] = np.roll(field[u, :], s)
    return out

def iterate_clouds_from_bag(bag_path: str, topic: str):
    """
    Generator yielding raw PointCloud2 messages from a ROS2 bag.
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="", output_serialization_format="")
    )
    reader.set_filter(StorageFilter(topics=[topic]))
    while reader.has_next():
        _, raw_buf, _ = reader.read_next()
        yield deserialize_message(raw_buf, PointCloud2)

if __name__ == "__main__":
    # ------- User config -------
    bag_path = "/home/femi/Benchmarking_framework/Data/bag_files/Airbus_Airbus_03_08_2023_tug_towbarless_A321_Neo_08-03-2023-14-59-10"
    topic    = "/main/points"
    # Scale factors for integer fields (adjust per your sensor spec)
    RANGE_SCALE        = 1e-3   # e.g. if 'range' is stored in millimeters
    REFLECTIVITY_SCALE = 1.0    # if 0–255 unitless
    AMBIENT_SCALE      = 1.0
    # ---------------------------

    # 1) Precompute pixel shifts once
    pixel_shifts = extract_pixel_shift_by_row_field(bag_path)

    # 2) Process each PointCloud2
    for idx, cloud_msg in enumerate(iterate_clouds_from_bag(bag_path, topic)):
        # parse all fields into a (H, W) structured array
        cloud_arr = cloud2_to_array_pure(cloud_msg)
        H, W = cloud_arr.shape

        # 3) Extract channels and scale integers → floats
        intensity    = cloud_arr['intensity'].astype(np.float32)
        range_raw    = cloud_arr['range'].astype(np.float32)       * RANGE_SCALE
        reflectivity = cloud_arr['reflectivity'].astype(np.float32)* REFLECTIVITY_SCALE
        ambient      = cloud_arr['ambient'].astype(np.float32)     * AMBIENT_SCALE

        # 4) Destagger each channel
        ds_range     = destagger(range_raw,     pixel_shifts)
        ds_intensity = destagger(intensity,     pixel_shifts)
        ds_reflect   = destagger(reflectivity,  pixel_shifts)
        ds_ambient   = destagger(ambient,       pixel_shifts)

        print(f"[{idx:03d}] Destaggered → shape {H}×{W}")

        # 5) Display each in its own zoomable Matplotlib window
        for field, title, cmap in [
            (ds_range,     'Range (m)',       'viridis'),
            (ds_intensity, 'Intensity',       'inferno'),
            (ds_reflect,   'Reflectivity',    'magma'),
            (ds_ambient,   'Ambient',         'plasma'),
        ]:
            clean = np.nan_to_num(field, nan=0.0)
            vmin, vmax = np.percentile(clean, (1, 99))
            plt.figure(figsize=(6, 6))
            plt.imshow(clean, origin='lower',
                       vmin=vmin, vmax=vmax,
                       cmap=cmap, interpolation='nearest')
            plt.title(f'Frame {idx:03d} – {title}')
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()
