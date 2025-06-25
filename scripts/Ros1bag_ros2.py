#!/usr/bin/env python3
"""
convert_ros1_to_ros2.py — manually convert a ROS 1 .bag into a ROS 2 Iron bag folder
by copying raw message data, with zero type‐system lookups.
"""

import shutil
from pathlib import Path

from rosbags.rosbag1 import Reader as Ros1Reader
from rosbags.rosbag2 import Writer as Ros2Writer

# ——— CONFIGURE THESE PATHS ———
SRC = Path("/home/femi/Downloads/"
           "HAM_Airport_2024_08_08_movement_a321_ceo_unknown_2024-08-08T09-48-39.bag")
DST = Path("/home/femi/Evitado/Benchmarking_framework/Data/Bag_Files/"
           "HAM_Airport_2024_08_08_movement_a321_ceo_unknown_2024-08-08T09-48-39")
# ————————————————————————

def main():
    # 1) If the target folder exists, delete it (ROS 2 writer will recreate it)
    if DST.exists():
        print(f"🗑 Removing existing output folder '{DST}'")
        shutil.rmtree(DST)

    # 2) Open the ROS 1 bag for reading...
    with Ros1Reader(SRC) as reader:
        # 3) ...and open a ROS 2 (Iron) bag for writing
        writer = Ros2Writer(
            output_folder=str(DST),
            storage_id=reader.storage_id,
            serialization_format=reader.serialization_format,
            storage_options=reader.storage_options
        )

        # 4) Register every connection (topic/type) present in the ROS 1 bag
        for conn in reader.connections:
            writer.add_connection(conn)

        # 5) Copy every message verbatim
        for conn, data, timestamp in reader.messages():
            writer.write(conn, data, timestamp)

        writer.close()

    print(f"✅ Successfully converted\n  {SRC}\n→ {DST}")

if __name__ == "__main__":
    main()
