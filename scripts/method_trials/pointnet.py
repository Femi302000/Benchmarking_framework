import os
import json
import random
from pathlib import Path

import numpy as np
import open3d as o3d
import kagglehub
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# CONFIG
# ----------------------------
MODE = "infer"   # "train" or "infer"
CUSTOM_PCD_PATH = "/home/femi/Benchmarking_framework/scripts/pre_processing/outputs/scene_000_red_bbox_points.pcd"

NPOINTS = 4096
EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-3
SAVE_PATH = "pointnet_aircraft.pt"

AIRPLANE_CATEGORY = "02691156"  # airplane
PARTS = ["body", "engine", "tail", "wing"]
PART_TO_ID = {p: i for i, p in enumerate(PARTS)}  # body=0, engine=1, tail=2, wing=3

# ----------------------------
# Utils
# ----------------------------
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def normalize_pc(pc):
    centroid = pc.mean(axis=0, keepdims=True)
    pc = pc - centroid
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc = pc / (scale + 1e-9)
    return pc

def farthest_point_sampling(x, n_samples):
    N = x.shape[0]
    if N <= n_samples:
        idx = np.arange(N)
        if N < n_samples:
            pad = np.random.choice(idx, n_samples - N, replace=True)
            idx = np.concatenate([idx, pad])
        return idx
    idxs = np.zeros(n_samples, dtype=np.int64)
    dists = np.ones(N) * 1e10
    idxs[0] = np.random.randint(0, N)
    last = x[idxs[0]]
    for i in range(1, n_samples):
        d = np.sum((x - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        idxs[i] = np.argmax(dists)
        last = x[idxs[i]]
    return idxs

def augment(pc):
    pc = pc + np.clip(0.01 * np.random.randn(*pc.shape), -0.03, 0.03)
    theta = np.random.uniform(0, 2 * np.pi)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0, 0, 1]])
    return pc @ Rz.T

# ----------------------------
# Dataset Loader for Airplane
# ----------------------------
class ShapeNetAirplane(Dataset):
    def __init__(self, split="train", npoints=4096, augment_xyz=True):
        self.npoints = npoints
        self.augment_xyz = augment_xyz

        # Download (cached after first time)
        dataset_root = Path(kagglehub.dataset_download("majdouline20/shapenetpart-dataset"))

        # Correct airplane path under PartAnnotation
        cat_base = dataset_root / "PartAnnotation" / AIRPLANE_CATEGORY
        pts_dir = cat_base / "points"
        perpart_labels_base = cat_base / "points_label"
        expert_labels_dir = cat_base / "expert_verified" / "points_label"

        if not pts_dir.exists():
            raise RuntimeError(f"Airplane points folder not found at {pts_dir}")
        if not perpart_labels_base.exists() and not expert_labels_dir.exists():
            raise RuntimeError(
                f"No labels directory found at {perpart_labels_base} or {expert_labels_dir}"
            )

        self.pts_dir = pts_dir
        self.perpart_dirs = {p: perpart_labels_base / p for p in PARTS}
        self.expert_dir = expert_labels_dir  # files like <sid>.seg (single-file labels)

        # Preprocessed storage (keep next to dataset cache)
        self.proc_dir = dataset_root / "airplane_npz"
        self.proc_dir.mkdir(exist_ok=True)
        self.index_file = self.proc_dir / "index.json"

        if not self.index_file.exists():
            self._preprocess_all()

        with open(self.index_file, "r") as f:
            index = json.load(f)
        if split not in index:
            raise RuntimeError(f"Index file missing split '{split}'. Available: {list(index.keys())}")

        # Keep only npz that actually exist
        files = []
        for sid in index[split]:
            npz_path = self.proc_dir / f"{sid}.npz"
            if npz_path.exists():
                files.append(npz_path)

        self.files = files
        if len(self.files) == 0:
            raise RuntimeError(f"No {split} samples found. Check dataset / preprocessing.")

    def _load_labels_expert(self, sid, n_points):
        """Try expert_verified/points_label/<sid>.seg (per-point integer labels)."""
        if not self.expert_dir.exists():
            return None
        seg_path = self.expert_dir / f"{sid}.seg"
        if not seg_path.exists():
            return None
        lbl = np.loadtxt(seg_path).astype(np.int64)
        if lbl.ndim != 1 or lbl.shape[0] != n_points:
            return None
        # Normalize labels to start from 0 (they might be 1..4)
        lbl = lbl - lbl.min()
        return lbl

    def _load_labels_perpart(self, sid, n_points):
        """Fallback: merge binary masks from points_label/<part>/<sid>.seg."""
        any_found = False
        labels = np.zeros(n_points, dtype=np.int64)
        for part_name, part_id in PART_TO_ID.items():
            seg_path = self.perpart_dirs[part_name] / f"{sid}.seg"
            if seg_path.exists():
                mask = np.loadtxt(seg_path).astype(np.int64)
                if mask.ndim != 1 or mask.shape[0] != n_points:
                    return None
                labels[mask == 1] = part_id
                any_found = True
        return labels if any_found else None

    def _preprocess_all(self):
        print("Preprocessing Airplane category from PartAnnotation/02691156 ...")
        train_ids, val_ids = [], []
        all_pts_files = sorted(self.pts_dir.glob("*.pts"))

        kept = 0
        skipped_missing = 0
        skipped_shape = 0

        for pts_file in tqdm(all_pts_files):
            sid = pts_file.stem
            try:
                pts = np.loadtxt(pts_file).astype(np.float32)
            except Exception:
                skipped_shape += 1
                continue

            n_points = pts.shape[0]

            # 1) Prefer expert labels (single .seg with per-point class ids)
            labels = self._load_labels_expert(sid, n_points)

            # 2) Fallback to merging per-part binary masks
            if labels is None:
                labels = self._load_labels_perpart(sid, n_points)

            if labels is None:
                skipped_missing += 1
                continue

            # Normalize + FPS
            pts = normalize_pc(pts)
            sel = farthest_point_sampling(pts, self.npoints)
            pts_s = pts[sel]
            lbl_s = labels[sel]

            # Save
            np.savez(self.proc_dir / f"{sid}.npz", points=pts_s, labels=lbl_s)
            kept += 1

            if np.random.rand() < 0.8:
                train_ids.append(sid)
            else:
                val_ids.append(sid)

        with open(self.index_file, "w") as f:
            json.dump({"train": train_ids, "val": val_ids}, f)

        print(f"Preprocess summary: kept={kept}, skipped_missing_labels={skipped_missing}, "
              f"skipped_bad_points={skipped_shape}")
        print(f"Train={len(train_ids)}, Val={len(val_ids)}")
        if kept == 0:
            raise RuntimeError(
                "No samples were preprocessed. Check that expert labels exist at "
                f"{self.expert_dir} or per-part labels at {self.perpart_dirs['body'].parent}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        pts = arr["points"].astype(np.float32)
        lbl = arr["labels"].astype(np.int64)
        if self.augment_xyz:
            pts = augment(pts).astype(np.float32)
        return torch.from_numpy(pts), torch.from_numpy(lbl)

# ----------------------------
# PointNet Model
# ----------------------------
class STN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=False)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        init = torch.eye(3).repeat(B,1,1).to(x.device)
        mat = self.fc3(x).view(-1,3,3) + init
        return mat

class PointNetSeg(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3,64,1);   self.bn1  = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64,128,1); self.bn2  = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128,1024,1); self.bn3 = nn.BatchNorm1d(1024)

        # 128 (local from conv2) + 1024 (global) = 1152
        self.conv_f1 = nn.Conv1d(1152,512,1); self.bnf1 = nn.BatchNorm1d(512)
        self.conv_f2 = nn.Conv1d(512,256,1);  self.bnf2 = nn.BatchNorm1d(256)
        self.conv_f3 = nn.Conv1d(256,128,1);  self.bnf3 = nn.BatchNorm1d(128)
        self.conv_out = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        # x: (B,N,3)
        x = x.transpose(2,1)                  # (B,3,N)
        trans = self.stn(x)
        x = torch.bmm(trans, x)               # aligned (B,3,N)

        x = F.relu(self.bn1(self.conv1(x)))   # (B,64,N)
        x = F.relu(self.bn2(self.conv2(x)))   # (B,128,N)
        x_local = F.relu(self.bn3(self.conv3(x)))   # (B,1024,N)

        x_global = torch.max(x_local, 2, keepdim=True)[0]  # (B,1024,1)
        x_global_exp = x_global.repeat(1, 1, x.size(2))    # (B,1024,N)

        # concat 128-ch local (from conv2) with 1024-ch global -> 1152
        x_cat = torch.cat([x, x_global_exp], 1)            # (B,1152,N)

        x = F.relu(self.bnf1(self.conv_f1(x_cat)))
        x = F.relu(self.bnf2(self.conv_f2(x)))
        x = F.relu(self.bnf3(self.conv_f3(x)))
        x = self.conv_out(x)                                # (B,num_classes,N)
        return x.transpose(2,1)                             # (B,N,num_classes)


# ----------------------------
# Training / Evaluation / Inference
# ----------------------------
def train_one_epoch(model, loader, opt, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for pts, lbl in tqdm(loader, desc="train", leave=False):
        pts, lbl = pts.to(device), lbl.to(device)
        logits = model(pts)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lbl.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        pred = logits.argmax(-1)
        correct += (pred == lbl).sum().item()
        total += lbl.numel()
        loss_sum += loss.item()
    return loss_sum/len(loader), correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for pts, lbl in tqdm(loader, desc="eval", leave=False):
        pts, lbl = pts.to(device), lbl.to(device)
        logits = model(pts)
        pred = logits.argmax(-1)
        correct += (pred == lbl).sum().item()
        total += lbl.numel()
    return correct/total

@torch.no_grad()
def infer_and_visualize(model, path, device, npoints=4096):
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[0] == 0:
        raise ValueError("Point cloud has 0 points.")
    pts = normalize_pc(pts)
    sel = farthest_point_sampling(pts, npoints)
    pts_s = pts[sel]
    t = torch.from_numpy(pts_s).unsqueeze(0).to(device)
    logits = model(t).squeeze(0).cpu().numpy()
    pred = np.argmax(logits, axis=1)
    cmap = np.array([
        [0.9, 0.4, 0.3], # class 0
        [0.3, 0.7, 0.9], # class 1
        [0.5, 0.9, 0.5], # class 2
        [0.8, 0.8, 0.3], # class 3
    ], dtype=np.float32)
    colors = cmap[pred % 4]
    vis = o3d.geometry.PointCloud()
    vis.points = o3d.utility.Vector3dVector(pts_s.astype(np.float64))
    vis.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.visualization.draw_geometries([vis])

# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODE == "infer":
        model = PointNetSeg(num_classes=4).to(device)
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        model.eval()
        infer_and_visualize(model, CUSTOM_PCD_PATH, device, NPOINTS)
        return

    train_set = ShapeNetAirplane(split="train", npoints=NPOINTS, augment_xyz=True)
    val_set   = ShapeNetAirplane(split="val",   npoints=NPOINTS, augment_xyz=False)

    print(f"Loaded airplane dataset: train={len(train_set)} samples, val={len(val_set)} samples")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = PointNetSeg(num_classes=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_acc, best_epoch = 0.0, -1
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}: loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), SAVE_PATH)
    print(f"Best val acc: {best_acc:.4f} @ epoch {best_epoch}")

if __name__ == "__main__":
    main()
