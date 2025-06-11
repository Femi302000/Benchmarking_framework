import rosbag2_py
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage
import numpy as np
from geometry_msgs.msg import TransformStamped
from collections import deque
from typing import List, Tuple, Optional, Dict


# ── Helper: build a 3×3 rotation from a quaternion ─────────────────────────────
def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    ww = qw * qw
    xy = qx * qy
    xz = qx * qz
    xw = qx * qw
    yz = qy * qz
    yw = qy * qw
    zw = qz * qw
    R = np.array([
        [ ww + xx - yy - zz,      2 * (xy - zw),      2 * (xz + yw) ],
        [   2 * (xy + zw),    ww - xx + yy - zz,      2 * (yz - xw) ],
        [   2 * (xz - yw),      2 * (yz + xw),    ww - xx - yy + zz ],
    ])
    return R

# ── Helper: turn a TransformStamped into a 4×4 homogeneous matrix ─────────────
def transform_stamped_to_matrix(ts: TransformStamped) -> np.ndarray:
    t = ts.transform.translation
    q = ts.transform.rotation
    R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    M = np.eye(4, dtype=float)
    M[0:3, 0:3] = R
    M[0:3, 3] = [t.x, t.y, t.z]
    return M

# ── Helper: invert a 4×4 homogeneous matrix ────────────────────────────────────
def invert_homogeneous_matrix(M: np.ndarray) -> np.ndarray:
    R = M[0:3, 0:3]
    t = M[0:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    M_inv = np.eye(4, dtype=float)
    M_inv[0:3, 0:3] = R_inv
    M_inv[0:3, 3] = t_inv
    return M_inv

# ── find_frame_chain: BFS over a set of TransformStamped to go source→target ────
def find_frame_chain(
    transforms: List[TransformStamped],
    source_frame: str,
    target_frame: str
) -> Optional[List[Tuple[str, str, TransformStamped, bool]]]:
    neighbors: Dict[str, List[Tuple[str, TransformStamped, bool]]] = {}

    for ts in transforms:
        parent = ts.header.frame_id
        child  = ts.child_frame_id
        neighbors.setdefault(parent, [])
        neighbors.setdefault(child, [])
        # parent→child (inverted=False)
        neighbors[parent].append((child, ts, False))
        # child→parent (inverted=True)
        neighbors[child].append((parent, ts, True))

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

# ── compute_chain_matrix: multiply all hops into one 4×4 matrix ────────────────
def compute_chain_matrix(
    chain: List[Tuple[str, str, TransformStamped, bool]]
) -> np.ndarray:
    M_total = np.eye(4, dtype=float)
    for (from_f, to_f, ts, inverted) in chain:
        M_hop = transform_stamped_to_matrix(ts)
        if inverted:
            M_hop = invert_homogeneous_matrix(M_hop)
        M_total = M_total @ M_hop
    return M_total


# ── Print an indented TF tree for a given list of TransformStamped ─────────────
def print_tf_tree(transforms: List[TransformStamped]) -> None:
    """
    Given a list of TransformStamped, reconstruct the TF tree and print it
    in an indented, human-readable form.
    """
    tree: Dict[str, List[str]] = {}
    all_children = set()

    for ts in transforms:
        parent = ts.header.frame_id
        child  = ts.child_frame_id
        tree.setdefault(parent, []).append(child)
        all_children.add(child)
        tree.setdefault(child, [])

    # root frames = those that never appear as a child
    all_parents = set(tree.keys())
    root_frames = sorted(all_parents - all_children)

    def _print_subtree(frame: str, prefix: str = ""):
        print(prefix + frame)
        for child in sorted(tree[frame]):
            _print_subtree(child, prefix + "    ")

    if not root_frames:
        print("No root frames found. (Possible cycle or no transforms at all.)")
        return

    for root in root_frames:
        _print_subtree(root)
        print("")  # blank line between disconnected subtrees


# ── MAIN ────────────────────────────────────────────────────────────────────────
def main(bag_path: str):
    # -------------------------------------------------------------------
    # 1) FIRST: read _only_ /tf_static and test the "static-only" chain
    # -------------------------------------------------------------------
    static_transforms: List[TransformStamped] = []
    seen_static_edges = set()
    reader_static = rosbag2_py.SequentialReader()
    storage_opts   = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_opts = rosbag2_py.ConverterOptions(input_serialization_format='', output_serialization_format='')
    reader_static.open(storage_opts, converter_opts)

    while reader_static.has_next():
        topic_name, raw_bytes, _ = reader_static.read_next()
        if topic_name != "/tf_static":
            continue
        tf_msg: TFMessage = deserialize_message(raw_bytes, TFMessage)
        for ts in tf_msg.transforms:
            edge = (ts.header.frame_id, ts.child_frame_id)
            if edge not in seen_static_edges:
                seen_static_edges.add(edge)
                static_transforms.append(ts)

    # Right after collecting static transforms, see if they already connect base_link → main_sensor:
    print("=== Testing STATIC-ONLY TF chain (using /tf_static) ===")
    static_chain = find_frame_chain(static_transforms, "base_link", "main_sensor")
    if static_chain is None:
        print("No static-only transform (base_link → main_sensor) found.\n")
    else:
        print("Found a STATIC-ONLY chain of length", len(static_chain), "hops:")
        for idx, (f_from, f_to, ts, inverted) in enumerate(static_chain, start=1):
            dir_str = "parent→child" if not inverted else "child→parent"
            print(f"  Hop {idx}: {f_from} → {f_to} ({dir_str})")
        M_static = compute_chain_matrix(static_chain)
        print("Static-only 4×4 matrix:")
        print(M_static, "\n")

    # -------------------------------------------------------------------
    # 2) NEXT: read /tf (dynamic) until we find a "dynamic-only" chain
    # -------------------------------------------------------------------
    dynamic_transforms: List[TransformStamped] = []
    seen_dynamic_edges = set()
    reader_tf = rosbag2_py.SequentialReader()
    reader_tf.open(storage_opts, converter_opts)

    first_dynamic_chain_ts = None
    first_dynamic_matrix = None

    while reader_tf.has_next():
        topic_name, raw_bytes, timestamp = reader_tf.read_next()
        if topic_name != "/tf":
            continue

        tf_msg: TFMessage = deserialize_message(raw_bytes, TFMessage)
        for ts in tf_msg.transforms:
            edge = (ts.header.frame_id, ts.child_frame_id)
            if edge not in seen_dynamic_edges:
                seen_dynamic_edges.add(edge)
                dynamic_transforms.append(ts)

        # At each new dynamic timestamp, check if dynamic-only transforms suffice:
        dyn_chain = find_frame_chain(dynamic_transforms, "base_link", "main_sensor")
        if (dyn_chain is not None) and (first_dynamic_chain_ts is None):
            # The very first time we see a valid dynamic-only chain, record it
            first_dynamic_chain_ts = timestamp
            first_dynamic_matrix = compute_chain_matrix(dyn_chain)
            break  # stop reading dynamic after finding first dynamic-only path

    print("=== Testing DYNAMIC-ONLY TF chain (using /tf) ===")
    if first_dynamic_chain_ts is None:
        print("No dynamic-only transform (base_link → main_sensor) found.\n")
    else:
        print(f"First dynamic-only chain occurred at timestamp {first_dynamic_chain_ts}:")
        print(first_dynamic_matrix, "\n")

    # -------------------------------------------------------------------
    # 3) FINALLY: combine static + dynamic and pick the earliest possible time
    # -------------------------------------------------------------------
    all_transforms: List[TransformStamped] = static_transforms.copy()
    all_edges = seen_static_edges.copy()
    reader_tf_rewind = rosbag2_py.SequentialReader()
    reader_tf_rewind.open(storage_opts, converter_opts)

    chosen_timestamp = None
    chosen_matrix = None

    while reader_tf_rewind.has_next():
        topic_name, raw_bytes, timestamp = reader_tf_rewind.read_next()
        if topic_name != "/tf":
            continue

        tf_msg: TFMessage = deserialize_message(raw_bytes, TFMessage)
        for ts in tf_msg.transforms:
            edge = (ts.header.frame_id, ts.child_frame_id)
            if edge not in all_edges:
                all_edges.add(edge)
                all_transforms.append(ts)

        # Now check if the combined set—static + whatever dynamics we've seen so far—makes a valid chain:
            comb_chain = find_frame_chain(all_transforms, "base_link", "main_sensor")
            if comb_chain is not None :
                chosen_timestamp = timestamp
                chosen_chain = comb_chain  # save the chain itself, not just the matrix
                chosen_matrix = compute_chain_matrix(comb_chain)
                break

        print("=== Testing COMBINED (static+dynamic) TF chain ===")
        if chosen_matrix is None :
            print("No combined transform (base_link → main_sensor) found in the entire bag.")
        else :
            print(f"First combined chain at timestamp {chosen_timestamp}:")

            # 1) Print the 4×4 matrix as before
            print(chosen_matrix)

            # 2) Now *explain* exactly which hops came from static vs dynamic:
            print("\nBreakdown of each hop in the chain:")
            for idx, (from_f, to_f, ts, inverted) in enumerate(chosen_chain, start=1) :
                direction = "parent→child" if not inverted else "child→parent"

                # Check if this edge was in /tf_static or /tf
                if (from_f, to_f) in seen_static_edges :
                    source = "static (/tf_static)"
                else :
                    source = "dynamic (/tf)"

                print(f"  Hop {idx}: {from_f} → {to_f} ({direction}), sourced from {source}")

            # (Optional) Also print translation + quaternion, as before:
            translation = chosen_matrix[0 :3, 3].tolist()
            R = chosen_matrix[0 :3, 0 :3]
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            if trace > 0.0 :
                s = 0.5 / np.sqrt(trace + 1.0)
                qw = 0.25 / s
                qx = (R[2, 1] - R[1, 2]) * s
                qy = (R[0, 2] - R[2, 0]) * s
                qz = (R[1, 0] - R[0, 1]) * s
            else :
                if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]) :
                    s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                    qw = (R[2, 1] - R[1, 2]) / s
                    qx = 0.25 * s
                    qy = (R[0, 1] + R[1, 0]) / s
                    qz = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2] :
                    s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                    qw = (R[0, 2] - R[2, 0]) / s
                    qx = (R[0, 1] + R[1, 0]) / s
                    qy = 0.25 * s
                    qz = (R[1, 2] + R[2, 1]) / s
                else :
                    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                    qw = (R[1, 0] - R[0, 1]) / s
                    qx = (R[0, 2] + R[2, 0]) / s
                    qy = (R[1, 2] + R[2, 1]) / s
                    qz = 0.25 * s

            print(f"\nTranslation (x, y, z): {translation}")
            print(f"Rotation (qx, qy, qz, qw): [{qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}]")


# ── EDIT ONLY THIS LINE ────────────────────────────────────────────────────────
bag_uri = "/home/femi/Evitado/Benchmarking_framework/Data/Bag_Files/Airbus_Airbus_03_08_2023_tug_towbarless_A321_Neo_08-03-2023-14-59-10"
main(bag_uri)
