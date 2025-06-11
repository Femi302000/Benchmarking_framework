import rosbag2_py
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

import numpy as np
import math

from typing import Dict, Tuple, List, Optional
from collections import deque

def invert_homogeneous_matrix(M: np.ndarray) -> np.ndarray:
    """
    Given a 4×4 homogeneous matrix M = [R  t; 0  1],
    return its inverse M⁻¹ = [Rᵀ  −Rᵀ·t; 0  1].
    """
    R = M[0:3, 0:3]
    t = M[0:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    M_inv = np.eye(4, dtype=float)
    M_inv[0:3, 0:3] = R_inv
    M_inv[0:3, 3] = t_inv
    return M_inv

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Given a quaternion (qx, qy, qz, qw), return the corresponding 3×3 rotation matrix.
    """
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

def transform_stamped_to_matrix(ts: TransformStamped) -> np.ndarray:
    """
    Given a TransformStamped, return its 4×4 homogeneous matrix.
    """
    t = ts.transform.translation
    q = ts.transform.rotation
    R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    M = np.eye(4, dtype=float)
    M[0:3, 0:3] = R
    M[0:3, 3] = [t.x, t.y, t.z]
    return M

def find_frame_chain(
    transforms: List[TransformStamped],
    source_frame: str,
    target_frame: str
) -> Optional[List[Tuple[str, str, TransformStamped, bool]]]:
    """
    Given a list of TransformStamped (each having .header.frame_id and .child_frame_id),
    find a path from source_frame to target_frame. Returns a list of hops:
      (from_frame, to_frame, TransformStamped, inverted_flag)
    or None if no chain exists.
    """
    neighbors: Dict[str, List[Tuple[str, TransformStamped, bool]]] = {}
    for ts in transforms:
        parent = ts.header.frame_id
        child  = ts.child_frame_id
        if parent not in neighbors:
            neighbors[parent] = []
        if child not in neighbors:
            neighbors[child] = []
        neighbors[parent].append((child, ts, False))
        neighbors[child].append((parent, ts, True))

    queue = deque()
    visited = set()
    queue.append((source_frame, []))
    visited.add(source_frame)

    while queue:
        current, path = queue.popleft()
        if current == target_frame:
            return path
        for (nxt, ts_msg, inv) in neighbors.get(current, []):
            if nxt in visited:
                continue
            new_path = path + [(current, nxt, ts_msg, inv)]
            if nxt == target_frame:
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))

    return None

def stamp_to_nanosec(stamp) -> int:
    """
    Convert a builtin_interfaces/Time (stamp.sec, stamp.nanosec) into an integer nanoseconds.
    """
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

def main_offline_tf2_echo(bag_path: str):
    # 1) Open the bag for reading
    reader = rosbag2_py.SequentialReader()
    storage_opts = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_opts = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_opts, converter_opts)

    # 2) Get all topic names
    topics_and_types = reader.get_all_topics_and_types()
    topic_names = [meta.name for meta in topics_and_types]
    print("Topic names in bag (as strings):")
    for name in topic_names:
        print("  •", repr(name))

    # 3) We will maintain a dictionary of "latest transform per (parent,child)".
    #    Key: (parent_frame, child_frame)
    #    Value: TransformStamped with the largest timestamp.
    latest_map: Dict[Tuple[str, str], TransformStamped] = {}

    # Helper to update latest_map (static transforms at time=0 count unless overwritten)
    def maybe_update(ts: TransformStamped):
        key = (ts.header.frame_id, ts.child_frame_id)
        nano = stamp_to_nanosec(ts.header.stamp)
        if key not in latest_map:
            latest_map[key] = ts
        else:
            existing = latest_map[key]
            if stamp_to_nanosec(existing.header.stamp) < nano:
                latest_map[key] = ts

    # 4) Read BOTH /tf_static and /tf in one pass.
    while reader.has_next():
        topic, raw_bytes, _ = reader.read_next()
        if topic not in ['/tf_static', '/tf']:
            continue

        tf_msg: TFMessage = deserialize_message(raw_bytes, TFMessage)
        for ts in tf_msg.transforms:
            # For a static transform (from /tf_static), header.stamp is often zero.
            # For dynamic (/tf), stamp is actual ROS time.
            maybe_update(ts)

    # 5) Now latest_map contains exactly one TransformStamped per edge (parent→child),
    #    chosen to be the one with the highest timestamp in the bag.
    all_transforms = list(latest_map.values())

    # 6) Print out what tf2_echo would show for: base_link_tug → main_sensor
    chain = find_frame_chain(all_transforms, "base_link", "main_sensor")
    if chain is None:
        print("No connection found between base_link_tug and main_sensor.")
        return

    # 7) Print the hops
    print("=== Hops in chain (using latest transforms) ===")
    for idx, (f, t, ts_msg, inv) in enumerate(chain, start=1):
        direction = "parent→child" if not inv else "child→parent"
        ts_stamp = ts_msg.header.stamp
        print(f"  Hop {idx}: {f} → {t}  ({direction}),  stamp = {ts_stamp.sec}.{ts_stamp.nanosec:09d}")

    # 8) For each hop, compute and print its 4×4 matrix, translation, quaternion, RPY
    print("\n=== Individual hop matrices ===")
    for idx, (f, t, ts_msg, inv) in enumerate(chain, start=1):
        M_hop = transform_stamped_to_matrix(ts_msg)
        if inv:
            M_hop = invert_homogeneous_matrix(M_hop)

        print(f"\nHop {idx} matrix ({f} → {t}, inverted={inv}):")
        print(M_hop)

        tx, ty, tz = M_hop[0,3], M_hop[1,3], M_hop[2,3]
        # If not inverted, quaternion == ts_msg.transform.rotation
        if not inv:
            qx = ts_msg.transform.rotation.x
            qy = ts_msg.transform.rotation.y
            qz = ts_msg.transform.rotation.z
            qw = ts_msg.transform.rotation.w
        else:
            # Recompute quaternion from inverted rotation block
            R_inv = M_hop[0:3, 0:3]
            trace = R_inv[0, 0] + R_inv[1, 1] + R_inv[2, 2]
            if trace > 0.0:
                s = 0.5 / math.sqrt(trace + 1.0)
                qw = 0.25 / s
                qx = (R_inv[2,1] - R_inv[1,2]) * s
                qy = (R_inv[0,2] - R_inv[2,0]) * s
                qz = (R_inv[1,0] - R_inv[0,1]) * s
            else:
                if (R_inv[0,0] > R_inv[1,1]) and (R_inv[0,0] > R_inv[2,2]):
                    s = 2.0 * math.sqrt(1.0 + R_inv[0,0] - R_inv[1,1] - R_inv[2,2])
                    qw = (R_inv[2,1] - R_inv[1,2]) / s
                    qx = 0.25 * s
                    qy = (R_inv[0,1] + R_inv[1,0]) / s
                    qz = (R_inv[0,2] + R_inv[2,0]) / s
                elif R_inv[1,1] > R_inv[2,2]:
                    s = 2.0 * math.sqrt(1.0 + R_inv[1,1] - R_inv[0,0] - R_inv[2,2])
                    qw = (R_inv[0,2] - R_inv[2,0]) / s
                    qx = (R_inv[0,1] + R_inv[1,0]) / s
                    qy = 0.25 * s
                    qz = (R_inv[1,2] + R_inv[2,1]) / s
                else:
                    s = 2.0 * math.sqrt(1.0 + R_inv[2,2] - R_inv[0,0] - R_inv[1,1])
                    qw = (R_inv[1,0] - R_inv[0,1]) / s
                    qx = (R_inv[0,2] + R_inv[2,0]) / s
                    qy = (R_inv[1,2] + R_inv[2,1]) / s
                    qz = 0.25 * s

        print(f"    Translation: x={tx:.6f}, y={ty:.6f}, z={tz:.6f}")
        print(f"    Quaternion: [qx={qx:.6f}, qy={qy:.6f}, qz={qz:.6f}, qw={qw:.6f}]")

        # RPY (rad and deg)
        sinr_cosp = 2.0 * (qw*qx + qy*qz)
        cosr_cosp = 1.0 - 2.0*(qx*qx + qy*qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw*qy - qz*qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi/2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (qw*qz + qx*qy)
        cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        print(f"    RPY (rad):  [{roll:.3f}, {pitch:.3f}, {yaw:.3f}]")
        print(f"    RPY (deg):  [{math.degrees(roll):.3f}, {math.degrees(pitch):.3f}, {math.degrees(yaw):.3f}]")

    # 9) Compute the “common” timestamp for the final transform:
    #    It is the minimum nanoseconds among all hops in the chain,
    #    because the final transform is only valid at the earliest time
    #    at which each used edge is available.
    hop_nanos = [stamp_to_nanosec(ts_msg.header.stamp) for (_, _, ts_msg, _) in chain]
    common_nano = min(hop_nanos)
    common_sec = common_nano // 1_000_000_000
    common_rem = common_nano % 1_000_000_000
    print(f"\nFinal transform is valid at timestamp: {common_sec}.{common_rem:09d}")

    # 10) Compute the cumulative 4×4 from base_link_tug → main_sensor
    M_cumul = np.eye(4, dtype=float)
    for (_, _, ts_msg, inv) in chain:
        M_hop = transform_stamped_to_matrix(ts_msg)
        if inv:
            M_hop = invert_homogeneous_matrix(M_hop)
        M_cumul = M_cumul @ M_hop

    print("\n=== Transform from 'base_link' to 'main_sensor' (cumulative) ===")
    print(M_cumul)

    # 11) Finally decompose M_cumul into translation & quaternion, and print
    tx, ty, tz = M_cumul[0,3], M_cumul[1,3], M_cumul[2,3]
    R = M_cumul[0:3, 0:3]
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s

    print("\nTranslation (x, y, z):", [tx, ty, tz])
    print("Rotation (qx, qy, qz, qw):", [qx, qy, qz, qw])


if __name__ == '__main__':
    # ── EDIT ONLY THIS PATH ────────────────────────────────────────────────────────
    bag_uri = "/home/femi/Evitado/Benchmarking_framework/Data/Bag_Files/Airbus_Airbus_03_08_2023_tug_towbarless_A321_Neo_08-03-2023-14-59-10"
    main_offline_tf2_echo(bag_uri)
