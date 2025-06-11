#!/usr/bin/env python3
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import transforms3d.quaternions as t3q

# ──────────────────────────────────────────────────────────────────────────────
# CDR HELPER CLASSES & FUNCTIONS (from sections 1–4 above)
# ──────────────────────────────────────────────────────────────────────────────
class CdrReader:
    def __init__(self, data: bytes):
        self._data = data
        self._idx = 0
        self._len = len(data)

    def _align(self, alignment: int):
        if alignment <= 1:
            return
        mis = self._idx % alignment
        if mis != 0:
            self._idx += (alignment - mis)

    def read_uint32(self) -> int:
        self._align(4)
        if self._idx + 4 > self._len:
            raise RuntimeError("Buffer overflow on read_uint32")
        val = int.from_bytes(self._data[self._idx:self._idx + 4], 'little')
        self._idx += 4
        return val

    def read_uint64(self) -> int:
        self._align(8)
        if self._idx + 8 > self._len:
            raise RuntimeError("Buffer overflow on read_uint64")
        val = int.from_bytes(self._data[self._idx:self._idx + 8], 'little')
        self._idx += 8
        return val

    def read_double(self) -> float:
        self._align(8)
        if self._idx + 8 > self._len:
            raise RuntimeError("Buffer overflow on read_double")
        val = np.frombuffer(self._data[self._idx:self._idx + 8], dtype='<f8')[0]
        self._idx += 8
        return float(val)

    def read_string(self) -> str:
        self._align(4)
        length = self.read_uint32()
        if length == 0:
            return ''
        if self._idx + length > self._len:
            raise RuntimeError("Buffer overflow on read_string data")
        raw = self._data[self._idx:self._idx + length - 1]  # drop trailing null
        s = raw.decode('utf-8')
        self._idx += length
        self._align(4)
        return s

    def eof(self) -> bool:
        return self._idx >= self._len


def parse_time(reader: CdrReader) -> Tuple[int, int]:
    sec = reader.read_uint32()
    nsec = reader.read_uint32()
    return sec, nsec


def parse_header(reader: CdrReader) -> Tuple[Tuple[int, int], str]:
    stamp = parse_time(reader)
    frame_id = reader.read_string()
    return stamp, frame_id


def parse_vector3(reader: CdrReader) -> Tuple[float, float, float]:
    x = reader.read_double()
    y = reader.read_double()
    z = reader.read_double()
    return x, y, z


def parse_quaternion(reader: CdrReader) -> Tuple[float, float, float, float]:
    x = reader.read_double()
    y = reader.read_double()
    z = reader.read_double()
    w = reader.read_double()
    return x, y, z, w


def parse_transform(reader: CdrReader) -> Tuple[Tuple[float, float, float],
                                                Tuple[float, float, float, float]]:
    t = parse_vector3(reader)
    q = parse_quaternion(reader)
    return t, q


class TransformStampedPure:
    __slots__ = ('parent_frame', 'child_frame',
                 'translation', 'rotation', 'stamp')

    def __init__(self,
                 parent_frame: str,
                 child_frame: str,
                 translation: Tuple[float, float, float],
                 rotation: Tuple[float, float, float, float],
                 stamp: Tuple[int, int]):
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self.translation = translation
        self.rotation = rotation
        self.stamp = stamp

    def __repr__(self):
        return (f"TransformStampedPure("
                f"{self.parent_frame!r} → {self.child_frame!r}, "
                f"t={self.translation}, r={self.rotation}, "
                f"stamp={self.stamp})"
               )


def parse_transform_stamped(reader: CdrReader) -> TransformStampedPure:
    (sec, nsec), frame_id = parse_header(reader)
    child_frame_id = reader.read_string()
    (tx, ty, tz), (qx, qy, qz, qw) = parse_transform(reader)
    return TransformStampedPure(
        parent_frame=frame_id,
        child_frame=child_frame_id,
        translation=(tx, ty, tz),
        rotation=(qx, qy, qz, qw),
        stamp=(sec, nsec)
    )


def parse_tf_message(blob: bytes) -> List[TransformStampedPure]:
    reader = CdrReader(blob)
    count = reader.read_uint32()
    transforms: List[TransformStampedPure] = []
    for _ in range(count):
        ts = parse_transform_stamped(reader)
        transforms.append(ts)
    return transforms


# ──────────────────────────────────────────────────────────────────────────────
# 5) BFS‐BASED CHAIN‐FINDING & MATRIX COMPUTATION (same as before)
# ──────────────────────────────────────────────────────────────────────────────
def find_frame_chain(
    transforms: List[TransformStampedPure],
    source_frame: str,
    target_frame: str
) -> Optional[List[Tuple[str, str, TransformStampedPure, bool]]]:
    neighbors: Dict[str, List[Tuple[str, TransformStampedPure, bool]]] = {}

    for ts in transforms:
        parent = ts.parent_frame
        child  = ts.child_frame
        neighbors.setdefault(parent, [])
        neighbors.setdefault(child, [])
        neighbors[parent].append((child, ts, False))
        neighbors[child].append((parent, ts, True))

    from collections import deque
    queue = deque([(source_frame, [])])
    visited = {source_frame}

    while queue:
        current_frame, path_so_far = queue.popleft()
        if current_frame == target_frame:
            return path_so_far
        for next_frame, ts_msg, inverted_flag in neighbors.get(current_frame, []):
            if next_frame in visited:
                continue
            hop = (current_frame, next_frame, ts_msg, inverted_flag)
            new_path = path_so_far + [hop]
            if next_frame == target_frame:
                return new_path
            visited.add(next_frame)
            queue.append((next_frame, new_path))

    return None


def compute_chain_matrix(
    chain: List[Tuple[str, str, TransformStampedPure, bool]]
) -> np.ndarray:
    M_total = np.eye(4, dtype=float)
    for (_from, _to, ts, inverted) in chain:
        # build 4×4 for this hop
        tx, ty, tz = ts.translation
        qx, qy, qz, qw = ts.rotation
        R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        M_hop = np.eye(4, dtype=float)
        M_hop[0:3, 0:3] = R
        M_hop[0:3, 3] = [tx, ty, tz]
        if inverted:
            M_hop = invert_homogeneous_matrix(M_hop)
        M_total = M_total @ M_hop
    return M_total


def print_tf_tree(transforms: List[TransformStampedPure]) -> None:
    tree: Dict[str, List[str]] = {}
    all_children = set()

    for ts in transforms:
        parent = ts.parent_frame
        child  = ts.child_frame
        tree.setdefault(parent, []).append(child)
        all_children.add(child)
        tree.setdefault(child, [])

    roots = sorted(set(tree.keys()) - all_children)
    if not roots:
        print("No root frames found. (Possible cycle or no transforms at all.)")
        return

    def _print_subtree(frame: str, prefix: str = ""):
        print(prefix + frame)
        for ch in sorted(tree[frame]):
            _print_subtree(ch, prefix + "    ")

    for root in roots:
        _print_subtree(root)
        print("")


# ──────────────────────────────────────────────────────────────────────────────
# 6) SQLITE3 BAG LISTING & RAW READERS (no rosbag2_py)
# ──────────────────────────────────────────────────────────────────────────────
def find_first_db3(bag_folder: str) -> str:
    bag_dir = Path(bag_folder)
    if not bag_dir.is_dir():
        raise RuntimeError(f"Bag folder '{bag_folder}' not found.")
    for file in bag_dir.iterdir():
        if file.suffix == ".db3":
            return str(file)
    raise RuntimeError(f"No .db3 file found in '{bag_folder}'.")


def list_bag_topics(db_path: str) -> Dict[int, Tuple[str, str]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, type FROM topics;")
    rows = cursor.fetchall()
    topic_map: Dict[int, Tuple[str, str]] = {}
    print("Topics in bag:")
    for (topic_id, topic_name, type_name) in rows:
        print(f"  id={topic_id:3d}  →  {topic_name}  (type = '{type_name}')")
        topic_map[topic_id] = (topic_name, type_name)
    conn.close()
    return topic_map


def read_raw_messages(db_path: str, topic_id: int):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC;",
        (topic_id,)
    )
    for (timestamp_ns, raw_blob) in cursor:
        yield (timestamp_ns, raw_blob)
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# 7) MAIN: STATIC-ONLY, DYNAMIC-ONLY, AND COMBINED (pure Python CDR decoding)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    BAG_FOLDER = (
        "/home/femi/Evitado/Benchmarking_framework/Data/Bag_Files/"
        "Airbus_Airbus_03_08_2023_tug_towbarless_A321_Neo_08-03-2023-14-59-10"
    )
    db3_path = find_first_db3(BAG_FOLDER)
    print(f"Using SQLite bag file: {db3_path}\n")

    topic_map = list_bag_topics(db3_path)
    print("")
    name_to_id = {name: tid for (tid, (name, _)) in topic_map.items()}

    if "/tf_static" not in name_to_id:
        print("Warning: '/tf_static' not found in this bag.")
    if "/tf" not in name_to_id:
        print("Warning: '/tf' not found in this bag.")

    # 1) STATIC-ONLY
    static_transforms: List[TransformStampedPure] = []
    seen_static_edges = set()
    if "/tf_static" in name_to_id:
        tfs_id = name_to_id["/tf_static"]
        for (t_ns, raw_blob) in read_raw_messages(db3_path, tfs_id):
            tf_list = parse_tf_message(raw_blob)
            for ts in tf_list:
                edge = (ts.parent_frame, ts.child_frame)
                if edge not in seen_static_edges:
                    seen_static_edges.add(edge)
                    static_transforms.append(ts)

    print("=== Testing STATIC-ONLY TF chain (using /tf_static) ===")
    static_chain = find_frame_chain(static_transforms, "base_link", "main_sensor")
    if static_chain is None:
        print("No static-only transform (base_link → main_sensor) found.\n")
    else:
        print("Found a STATIC-ONLY chain of length", len(static_chain), "hops:")
        for idx, (f_from, f_to, ts, inverted) in enumerate(static_chain, start=1):
            direction = "parent→child" if not inverted else "child→parent"
            print(f"  Hop {idx}: {f_from} → {f_to} ({direction}) [static]")
        M_static = compute_chain_matrix(static_chain)
        print("Static-only 4×4 matrix:")
        print(M_static, "\n")

    # 2) DYNAMIC-ONLY
    dynamic_transforms: List[TransformStampedPure] = []
    seen_dynamic_edges = set()
    first_dynamic_chain_ts = None
    first_dynamic_matrix = None

    if "/tf" in name_to_id:
        tf_id = name_to_id["/tf"]
        for (t_ns, raw_blob) in read_raw_messages(db3_path, tf_id):
            tf_list = parse_tf_message(raw_blob)
            for ts in tf_list:
                edge = (ts.parent_frame, ts.child_frame)
                if edge not in seen_dynamic_edges:
                    seen_dynamic_edges.add(edge)
                    dynamic_transforms.append(ts)

            dyn_chain = find_frame_chain(dynamic_transforms, "base_link", "main_sensor")
            if dyn_chain is not None and first_dynamic_chain_ts is None:
                first_dynamic_chain_ts = t_ns
                first_dynamic_matrix = compute_chain_matrix(dyn_chain)
                break

    print("=== Testing DYNAMIC-ONLY TF chain (using /tf) ===")
    if first_dynamic_chain_ts is None:
        print("No dynamic-only transform (base_link → main_sensor) found.\n")
    else:
        print(f"First dynamic-only chain occurred at timestamp {first_dynamic_chain_ts}:")
        print(first_dynamic_matrix, "\n")

    # 3) COMBINED STATIC+DYNA MIC
    combined_transforms: List[TransformStampedPure] = static_transforms.copy()
    all_edges = set(seen_static_edges)
    chosen_timestamp = None
    chosen_chain = None
    chosen_matrix = None

    if "/tf" in name_to_id:
        tf_id = name_to_id["/tf"]
        for (t_ns, raw_blob) in read_raw_messages(db3_path, tf_id):
            tf_list = parse_tf_message(raw_blob)
            for ts in tf_list:
                edge = (ts.parent_frame, ts.child_frame)
                if edge not in all_edges:
                    all_edges.add(edge)
                    combined_transforms.append(ts)

            comb_chain = find_frame_chain(combined_transforms, "base_link", "main_sensor")
            if comb_chain is not None:
                chosen_timestamp = t_ns
                chosen_chain = comb_chain
                chosen_matrix = compute_chain_matrix(comb_chain)
                break

    print("=== Testing COMBINED (static+dynamic) TF chain ===")
    if chosen_matrix is None:
        print("No combined transform (base_link → main_sensor) found in the entire bag.")
    else:
        print(f"First combined chain at timestamp {chosen_timestamp}:")
        print(chosen_matrix)

        print("\nBreakdown of each hop (static vs dynamic):")
        for idx, (f_from, f_to, ts, inverted) in enumerate(chosen_chain, start=1):
            direction = "parent→child" if not inverted else "child→parent"
            if (f_from, f_to) in seen_static_edges:
                source = "static (/tf_static)"
            else:
                source = "dynamic (/tf)"
            print(f"  Hop {idx}: {f_from} → {f_to} ({direction}), sourced from {source}")

        t = chosen_matrix[0:3, 3]
        R = chosen_matrix[0:3, 0:3]
        qw, qx, qy, qz = t3q.mat2quat(R)
        print(f"\nTranslation (x, y, z): [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
        print(f"Rotation (qx, qy, qz, qw): [{qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}]")

    # 4) Print full TF tree
    print("\n=== Combined TF Tree Over Entire Bag ===")
    print_tf_tree(combined_transforms)


if __name__ == "__main__":
    main()
