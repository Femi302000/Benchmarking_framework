#!/usr/bin/env python3
"""
scripts/pre_processing/pixel_shift.py

This script will:
 1. Extract the 'pixel_shift_by_row' field for each message on '/main/ouster_info', listing each row index and its value.
 2. Return a dictionary mapping each message timestamp to its pixel_shift_by_row as numpy arrays.
"""

import os
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

# ←— Set your bag folder path here (the directory containing .db3 + metadata.yaml)
BAG_PATH = (
    "/home/femi/Benchmarking_framework/Data/bag_files/"
    "HAM_Airport_2024_08_08_movement_a320_ceo_Germany"
)

# Fully-qualified topic you want to read
TOPIC_OF_INTEREST = '/main/ouster_info'

def extract_pixel_shift_by_row_field(bag_path: str, topic: str) -> np.ndarray | None:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rosidl_runtime_py.utilities import get_message
    from rclpy.serialization import deserialize_message
    import numpy as np

    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_opts = ConverterOptions(input_serialization_format='cdr',
                                      output_serialization_format='cdr')
    reader = SequentialReader()
    reader.open(storage_opts, converter_opts)

    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in topics:
        print(f" topic '{topic}' not found. Available topics:")
        for name in topics:
            print(f"  {name}")
        return None

    # Ouster info type (adjust if different in your bag)
    MsgClass = get_message(topics[topic])

    # Read first message with the field and return its array
    while reader.has_next():
        tname, data, _ = reader.read_next()
        if tname != topic:
            continue
        msg = deserialize_message(data, MsgClass)
        pixel_shift = getattr(msg, 'pixel_shift_by_row', None)
        if pixel_shift is not None:
            return np.array(pixel_shift, dtype=np.int32)

    print(f"No 'pixel_shift_by_row' in any message on {topic}")
    return None



if __name__ == '__main__':
    if not os.path.isdir(BAG_PATH):
        print(f"Error: BAG_PATH '{BAG_PATH}' is not a directory.")
        exit(1)

    shifts = extract_pixel_shift_by_row_field(BAG_PATH)
    print(shifts)
    # shifts is a dict {timestamp: numpy array of pixel_shift values}
    # You can further process shifts here as needed
