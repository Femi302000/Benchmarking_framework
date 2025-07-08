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

def extract_pixel_shift_by_row_field(bag_path: str):
    """
    Reads the bag at `bag_path`, extracts `pixel_shift_by_row` for each message on TOPIC_OF_INTEREST,
    prints each row index and value, and returns a dict mapping timestamps to numpy arrays.
    """
    # Prepare reader
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_opts = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader = SequentialReader()
    reader.open(storage_opts, converter_opts)

    # Map each topic to its msg type
    topic_type_map = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}

    if TOPIC_OF_INTEREST not in topic_type_map:
        print(f"Error: topic '{TOPIC_OF_INTEREST}' not found. Available topics:")
        for name in topic_type_map:
            print(f"  {name}")
        return None

    # Instantiate message class
    msg_type_str = 'ouster_sensor_msgs/msg/OusterSensorInfo'
    MsgClass = get_message(msg_type_str)

    shifts_by_timestamp = {}

    # Read and process messages
    while reader.has_next():
        topic_name, data_bytes, timestamp = reader.read_next()
        if topic_name != TOPIC_OF_INTEREST:
            continue

        msg = deserialize_message(data_bytes, MsgClass)
        pixel_shift = getattr(msg, 'pixel_shift_by_row', None)
        if pixel_shift is None:
            print(f"Message @ {timestamp} has no 'pixel_shift_by_row' field.")
            continue

        # Convert to numpy array
        shift_array = np.array(pixel_shift)
        shifts_by_timestamp[timestamp] = shift_array


    return shift_array


if __name__ == '__main__':
    if not os.path.isdir(BAG_PATH):
        print(f"Error: BAG_PATH '{BAG_PATH}' is not a directory.")
        exit(1)

    shifts = extract_pixel_shift_by_row_field(BAG_PATH)
    print(shifts)
    # shifts is a dict {timestamp: numpy array of pixel_shift values}
    # You can further process shifts here as needed
