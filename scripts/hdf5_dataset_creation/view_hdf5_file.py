#!/usr/bin/env python3
import h5py

def print_h5_structure(group, indent=0):
    """
    Recursively prints the structure of an HDF5 group or file.
    """
    prefix = ' ' * indent
    for attr_key, attr_val in group.attrs.items():
        print(f"{prefix} {attr_key} = {attr_val}")
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            print(f"{prefix}{name}/")
            print_h5_structure(item, indent + 4)
        else:
            shape = item.shape
            dtype = item.dtype
            line = f"{prefix}{name}    [shape={shape}, dtype={dtype}]"

            if 'columns' in item.attrs:
                cols = item.attrs['columns']

                cols = [c.decode('utf-8') if isinstance(c, (bytes, bytearray)) else str(c)
                        for c in cols]
                line += f"    columns={cols}"
            print(line)

if __name__ == "__main__":
    h5_path = (
        "/home/femi/Benchmarking_framework/Data/machine_learning_dataset/"
        "HAM_Airport_2024_08_08_movement_a320_ceo_Germany.h5"
    )

    with h5py.File(h5_path, "r") as f:
        print(f"File: {h5_path}")
        print_h5_structure(f)
