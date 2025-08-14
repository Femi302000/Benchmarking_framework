#!/usr/bin/env python3
import argparse, os, random
from pathlib import Path
import numpy as np
import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# =========================================================
# Utils
# =========================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pairwise_sq_dists(A, B):
    # A: (N,d) B: (K,d) -> (N,K)
    A2 = (A*A).sum(-1, keepdim=True)
    B2 = (B*B).sum(-1, keepdim=True).T
    AB = A @ B.T
    return A2 + B2 - 2*AB


def sinkhorn(K, r, c, n_iters=50, eps=1e-8):
    # K: (N,K)>0, r:(N,), c:(K,)
    u = torch.ones_like(r); v = torch.ones_like(c)
    for _ in range(n_iters):
        Ku = K @ v
        u = r / (Ku + eps)
        KTu = K.T @ u
        v = c / (KTu + eps)
    return torch.diag(u) @ K @ torch.diag(v)


def soft_cross_entropy(logp, soft_targets):
    # logp:(N,K) log-softmax, soft_targets:(N,K) rows sum to 1
    return -(soft_targets * logp).sum(dim=1).mean()


# =========================================================
# Geometry features (normals, curvature, FPFH)
# =========================================================
def estimate_normals_and_curvature(pcd, radius=0.15, max_nn=50, curv_knn=30):
    if len(pcd.points) == 0:
        return np.zeros((0,3), np.float32), np.zeros((0,), np.float32), pcd
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    # optional orientation for stability
    try:
        pcd.orient_normals_consistent_tangent_plane(k=curv_knn)
    except Exception:
        pass
    pts = np.asarray(pcd.points, dtype=np.float32)
    kd = o3d.geometry.KDTreeFlann(pcd)
    curv = np.zeros((pts.shape[0],), dtype=np.float32)
    for i in range(pts.shape[0]):
        _, idx, _ = kd.search_knn_vector_3d(pts[i], curv_knn)
        if len(idx) < 3:
            continue
        C = np.cov(pts[idx].T)
        ew, _ = np.linalg.eigh(C)
        ew = np.clip(ew, 1e-12, None)
        curv[i] = float(ew.min() / ew.sum())
    nr = np.asarray(pcd.normals, dtype=np.float32)
    return nr, curv, pcd


def compute_fpfh(pcd, radius=0.25, max_nn=80):
    if len(pcd.points) == 0:
        return np.zeros((0,33), dtype=np.float32)
    # requires normals present
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return np.asarray(fpfh.data, dtype=np.float32).T  # (N,33)


# =========================================================
# Prefilter (XY-DBSCAN + z-level helpers)
# =========================================================
def generate_keypoints_meta(
    pcd: o3d.geometry.PointCloud,
    eps_xy: float = 0.35,
    min_xy: int = 40,
    z_bin_size: float = 0.5,
    require_level_pair: bool = True,
):
    P = np.asarray(pcd.points)
    if P.size == 0:
        return {"labels": np.array([]), "cluster_ids": [], "cluster_to_level": {},
                "kept_levels": set(), "z_bin_size": z_bin_size}
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    XY = P[:, :2]
    labels = DBSCAN(eps=eps_xy, min_samples=min_xy).fit_predict(XY)
    cluster_ids = [lab for lab in np.unique(labels) if lab != -1]

    if z_bin_size <= 0:
        raise ValueError("z_bin_size must be > 0")
    cl_mean_z = {lab: Z[labels == lab].mean() for lab in cluster_ids}
    cl_level = {lab: int(np.floor(cl_mean_z[lab] / z_bin_size)) for lab in cluster_ids}

    level_to_cluster_count = {}
    for lab in cluster_ids:
        lev = cl_level[lab]
        level_to_cluster_count[lev] = level_to_cluster_count.get(lev, 0) + 1

    kept_levels = set(level_to_cluster_count.keys())
    if require_level_pair:
        kept_levels = {lev for lev, cnt in level_to_cluster_count.items() if cnt >= 2}

    return {
        "labels": labels,
        "cluster_ids": cluster_ids,
        "cluster_to_level": cl_level,
        "kept_levels": kept_levels,
        "z_bin_size": z_bin_size,
    }


def prefilter_points_with_xy_dbscan(
    pcd: o3d.geometry.PointCloud,
    eps_xy: float = 0.35,
    min_xy: int = 40,
    z_bin_size: float = 0.5,
    keep_only_kept_levels: bool = True,
    remove_noise_cluster: bool = True,
) -> np.ndarray:
    P = np.asarray(pcd.points)
    if P.size == 0: return P
    meta = generate_keypoints_meta(
        pcd, eps_xy=eps_xy, min_xy=min_xy, z_bin_size=z_bin_size,
        require_level_pair=keep_only_kept_levels
    )
    labels = meta["labels"]
    kept_lvls = meta["kept_levels"]
    cl2lvl = meta["cluster_to_level"]

    mask = np.ones(P.shape[0], dtype=bool)
    if remove_noise_cluster:
        mask &= (labels != -1)

    if keep_only_kept_levels and len(kept_lvls) > 0:
        lvl_ok = np.zeros_like(mask)
        pos = np.where(labels != -1)[0]
        for lab in np.unique(labels[pos]):
            lvl = cl2lvl[lab]
            if lvl in kept_lvls:
                lvl_ok[labels == lab] = True
        mask &= lvl_ok

    if not mask.any():  # fallback
        return P
    return P[mask]


# =========================================================
# Dataset
# =========================================================
class PcdDataset(Dataset):
    def __init__(self, data_dir, n_points=20000, voxel_size=None,
                 use_prefilter=True, pf_eps_xy=0.35, pf_min_xy=40, pf_z_bin=0.5,
                 pf_keep_levels=True, pf_drop_noise=True, add_normals=False):
        self.paths = sorted([str(p) for p in Path(data_dir).glob("*.pcd")])
        assert len(self.paths) > 0, f"No .pcd found in {data_dir}"
        self.n_points = n_points
        self.voxel_size = voxel_size
        self.use_prefilter = use_prefilter
        self.pf_eps_xy = pf_eps_xy
        self.pf_min_xy = pf_min_xy
        self.pf_z_bin = pf_z_bin
        self.pf_keep_levels = pf_keep_levels
        self.pf_drop_noise = pf_drop_noise
        self.add_normals = add_normals

    def __len__(self): return len(self.paths)

    def _load_o3d(self, path):
        pcd = o3d.io.read_point_cloud(path)
        if self.voxel_size:
            pcd = pcd.voxel_down_sample(self.voxel_size)

        if self.use_prefilter:
            pts_filtered = prefilter_points_with_xy_dbscan(
                pcd,
                eps_xy=self.pf_eps_xy,
                min_xy=self.pf_min_xy,
                z_bin_size=self.pf_z_bin,
                keep_only_kept_levels=self.pf_keep_levels,
                remove_noise_cluster=self.pf_drop_noise,
            )
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_filtered))

        pts = np.asarray(pcd.points, dtype=np.float32)

        if self.add_normals and len(pcd.points) > 0:
            nr, curv, pcd = estimate_normals_and_curvature(pcd, radius=0.15, max_nn=50, curv_knn=30)
            fpfh = compute_fpfh(pcd, radius=0.25, max_nn=80)  # (N,33)
            arr = np.concatenate([pts, nr, curv[:, None], fpfh], axis=1).astype(np.float32)  # (N,40)
        else:
            arr = pts

        return arr

    def _normalize(self, arr):
        pts = arr[:, :3]
        c = pts.mean(axis=0, keepdims=True)
        pts = pts - c
        scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
        pts = pts / scale
        arr[:, :3] = pts
        return arr

    def _resample(self, arr):
        N = arr.shape[0]
        if N == 0: raise ValueError("Empty point cloud after prefilter!")
        if N >= self.n_points:
            idx = np.random.choice(N, self.n_points, replace=False)
        else:
            extra = np.random.choice(N, self.n_points - N, replace=True)
            idx = np.concatenate([np.arange(N), extra])
        return arr[idx]

    def __getitem__(self, idx):
        arr = self._load_o3d(self.paths[idx])    # (N,3) or (N,40)
        arr = self._normalize(arr)
        arr = self._resample(arr)
        return torch.from_numpy(arr)


# =========================================================
# Model
# =========================================================
class PointNetEncoder(nn.Module):
    def __init__(self, out_dim=256, in_dim=3):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(True),
            nn.Linear(128, 256), nn.ReLU(True),
            nn.Linear(256, out_dim)
        )
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):  # x: (B,P,in_dim)
        B, P, _ = x.shape
        y = self.mlp1(x)          # (B,P,D)
        y = y.reshape(B*P, -1)
        y = self.bn(y)
        y = y.reshape(B, P, -1)
        return y


class ClassHead(nn.Module):
    def __init__(self, in_dim, k):
        super().__init__()
        self.lin = nn.Linear(in_dim, k)

    def forward(self, feat):  # (B,P,D) -> (B,P,K)
        return self.lin(feat)


# =========================================================
# Training (anti-collapse + smoothness + cosine feature distance)
# =========================================================
def train_softclu(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ds = PcdDataset(
        args.data_dir, n_points=args.points, voxel_size=args.voxel,
        use_prefilter=args.prefilter,
        pf_eps_xy=args.pf_eps_xy, pf_min_xy=args.pf_min_xy, pf_z_bin=args.pf_z_bin,
        pf_keep_levels=args.pf_keep_levels, pf_drop_noise=args.pf_drop_noise,
        add_normals=args.add_normals
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    in_dim = 3
    if args.add_normals:
        in_dim = 3 + 3 + 1 + 33  # xyz + normals + curvature + FPFH = 40

    enc = PointNetEncoder(out_dim=args.feat_dim, in_dim=in_dim).to(device)
    head = ClassHead(args.feat_dim, args.k).to(device)

    # diverse start
    nn.init.orthogonal_(head.lin.weight)
    nn.init.zeros_(head.lin.bias)

    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()),
                           lr=args.lr, weight_decay=1e-4)

    print(f"Training on {len(ds)} clouds | K={args.k} P={args.points} feat={args.feat_dim} "
          f"prefilter={args.prefilter} add_normals={args.add_normals}")

    enc.train(); head.train()

    mass_prior = None
    ema_momentum = 0.9
    uniform_c = None

    for epoch in range(1, args.epochs + 1):
        # gentler anneal
        t = epoch / max(1, args.epochs)
        tau = max(0.10, args.tau * (1.0 - 0.7 * t))        # e.g., 0.5 -> 0.10
        alpha = args.alpha * (1.0 - 0.4 * t)               # e.g., 0.6 -> 0.36

        losses = []
        for arr in dl:
            arr = arr.to(device).float()  # (1,P,in_dim)
            pts = arr[..., :3]

            feat = enc(arr)                      # (1,P,D)
            logits = head(feat)                  # (1,P,K)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp().squeeze(0)   # (P,K)
            f = feat.squeeze(0)                  # (P,D)
            x = pts.squeeze(0)                   # (P,3)
            Pn = x.shape[0]
            Knum = probs.shape[1]
            if uniform_c is None:
                uniform_c = torch.full((Knum,), 1.0 / Knum, device=device)

            # soft centroids (detach)
            assign = probs / (probs.sum(dim=0, keepdim=True) + 1e-8)
            C_f = (assign.T @ f).detach()                 # (K,D)
            C_x = (assign.T @ x).detach()                 # (K,3)

            # geometry distance
            Dx = pairwise_sq_dists(x, C_x)                # (P,K)

            # cosine feature distance
            fn = F.normalize(f, dim=1)
            Cfn = F.normalize(C_f, dim=1)
            Df = 1.0 - (fn @ Cfn.T)                       # (P,K), in [0,2]

            # stabilized blended cost
            Cmat = alpha * Dx + (1 - alpha) * Df
            Cmat = (Cmat - Cmat.mean()) / (Cmat.std() + 1e-6)
            Kmat = torch.exp(-Cmat / tau)

            # column prior (warmup uniform, then smoothed EMA pulled to uniform)
            if epoch <= 10:
                c = uniform_c
            else:
                hist = probs.mean(dim=0)  # (K,)
                if mass_prior is None:
                    mass_prior = hist.detach()
                else:
                    mass_prior = ema_momentum * mass_prior + (1 - ema_momentum) * hist.detach()
                c = 0.9 * (mass_prior / (mass_prior.sum() + 1e-8)) + 0.1 * uniform_c
                c = torch.clamp(c, min=1e-3)
                c = c / c.sum()

            r = torch.full((Pn,), 1.0 / Pn, device=device)

            with torch.no_grad():
                gamma = sinkhorn(Kmat, r, c, n_iters=args.sinkhorn)

            # main OT-matching loss
            loss_main = soft_cross_entropy(log_probs.squeeze(0), gamma)

            # orthogonality on classifier weights
            W = head.lin.weight
            ortho = ((W @ W.T) - torch.eye(W.shape[0], device=device)).pow(2).mean()

            # diversity (use many clusters): minimize -H(mean_probs)
            mean_probs = probs.mean(dim=0)
            div_loss = torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            # instance entropy: discourage ultra-peaky per-point assignments early
            inst_ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            inst_loss = -inst_ent

            # neighbor-consistency (kNN smoothness) loss
            with torch.no_grad():
                pts_np = x.detach().cpu().numpy()
                nn_graph = NearestNeighbors(n_neighbors=12).fit(pts_np)
                knn_idx = torch.from_numpy(nn_graph.kneighbors(pts_np, return_distance=False)).to(x.device)  # (P,12)

            nbr_probs = probs[knn_idx]                  # (P,12,K)
            Plog = torch.log(probs + 1e-8)
            nbr_logp = torch.log(nbr_probs + 1e-8)
            kl1 = (probs.unsqueeze(1) * (torch.log(probs.unsqueeze(1)+1e-8) - nbr_logp)).sum(-1)
            kl2 = (nbr_probs * (torch.log(nbr_probs+1e-8) - Plog.unsqueeze(1))).sum(-1)
            smooth_loss = 0.5 * (kl1 + kl2).mean()

            loss = loss_main + args.ortho_w * ortho + 0.05 * div_loss + 0.02 * inst_loss + 0.02 * smooth_loss

            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))

        print(f"[Epoch {epoch:03d}] loss={np.mean(losses):.4f}  tau={tau:.3f} alpha={alpha:.3f}")

    return enc.eval(), head.eval(), device


# =========================================================
# Inference, postprocess, save/view
# =========================================================
def colorize_by_cluster(pts_np, labels, k, seed=123):
    rng = np.random.default_rng(seed)
    palette = rng.random((k, 3))
    colors = palette[labels % k]
    return np.clip(colors, 0, 1)


def smooth_labels_knn(pts_sel, labels, k_neighbors=10):
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(pts_sel)
    idx_knn = nbrs.kneighbors(pts_sel, return_distance=False)
    smoothed = labels.copy()
    for i in range(labels.shape[0]):
        neigh = labels[idx_knn[i]]
        vals, cnts = np.unique(neigh, return_counts=True)
        smoothed[i] = vals[np.argmax(cnts)]
    return smoothed


def remove_tiny_components(pts_sel, labels, k, eps=0.02, min_samples=5, min_keep=50):
    labels_ref = labels.copy()
    for cid in range(k):
        mask = (labels_ref == cid)
        if mask.sum() == 0: continue
        coords = pts_sel[mask]
        if coords.shape[0] < min_keep:
            labels_ref[mask] = -1
            continue
        sub = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        comp_labels, comp_counts = np.unique(sub.labels_, return_counts=True)
        if comp_labels.size > 1:
            keep = comp_labels[np.argmax(comp_counts)]
            drop_mask = (sub.labels_ != keep)
            drop_idx = np.where(mask)[0][drop_mask]
            labels_ref[drop_idx] = -1
    # reassign -1 to modal neighbor label
    bad = np.where(labels_ref == -1)[0]
    if bad.size > 0:
        nn_all = NearestNeighbors(n_neighbors=10).fit(pts_sel)
        knn = nn_all.kneighbors(pts_sel[bad], return_distance=False)
        fill = []
        for row in knn:
            neigh = labels_ref[row]
            neigh = neigh[neigh != -1]
            if neigh.size == 0:
                fill.append(0)
            else:
                u, c = np.unique(neigh, return_counts=True)
                fill.append(u[np.argmax(c)])
        labels_ref[bad] = np.array(fill)
    return labels_ref


def run_inference_and_save(enc, head, device, input_pcd, args):
    # load original cloud
    pcd = o3d.io.read_point_cloud(input_pcd)
    if args.voxel is not None:
        pcd = pcd.voxel_down_sample(args.voxel)
    pts = np.asarray(pcd.points).astype(np.float32)
    if pts.shape[0] == 0:
        raise ValueError("Empty PCD at inference.")

    # normalize like training
    c = pts.mean(0, keepdims=True); s = np.max(np.linalg.norm(pts - c, axis=1)) + 1e-6
    pts_norm = (pts - c) / s

    # resample to fixed count for prediction
    N = pts_norm.shape[0]
    Pfix = args.points
    if N >= Pfix:
        sel_idx = np.random.choice(N, Pfix, replace=False)
    else:
        extra = np.random.choice(N, Pfix - N, replace=True)
        sel_idx = np.concatenate([np.arange(N), extra])
    pts_sel = pts_norm[sel_idx]

    # build inference features similarly (normals + FPFH, curvature optional)
    arr_sel = pts_sel
    if args.add_normals:
        pcd_sel = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_sel))
        nr, curv, pcd_sel = estimate_normals_and_curvature(pcd_sel, radius=0.15, max_nn=50, curv_knn=30)
        fpfh = compute_fpfh(pcd_sel, radius=0.25, max_nn=80)
        arr_sel = np.concatenate([pts_sel, nr, curv[:, None], fpfh], axis=1).astype(np.float32)

    with torch.no_grad():
        arr_t = torch.from_numpy(arr_sel).unsqueeze(0).to(device)
        feat = enc(arr_t).squeeze(0)                      # (P,D)
        logits = head(feat)                               # (P,K)
        labels = logits.argmax(dim=-1).cpu().numpy()      # (P,)

    # post-process: smoothing + tiny components
    labels = smooth_labels_knn(pts_sel, labels, k_neighbors=10)
    labels = remove_tiny_components(pts_sel, labels, k=args.k, eps=0.02, min_samples=5, min_keep=50)

    # propagate to all original points via NN in normalized space
    nn_all = NearestNeighbors(n_neighbors=1).fit(pts_sel)
    nn_idx = nn_all.kneighbors(pts_norm, return_distance=False).squeeze(1)
    labels_all = labels[nn_idx]

    # color and save
    colored = o3d.geometry.PointCloud()
    colored.points = o3d.utility.Vector3dVector(pts)  # original scale
    colors = colorize_by_cluster(pts, labels_all, args.k)
    colored.colors = o3d.utility.Vector3dVector(colors)

    # Print usage to verify spread
    u, cnts = np.unique(labels_all, return_counts=True)
    print("cluster usage:", dict(zip(u.tolist(), cnts.tolist())))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(args.out, colored)
    print(f"Saved clustered point cloud -> {args.out}")

    if args.show:
        o3d.visualization.draw_geometries([colored],
                                          window_name="SoftClu (features + smoothness)",
                                          width=1280, height=800,
                                          point_show_normal=False)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/femi/Benchmarking_framework/scripts/data_dir", help="folder with .pcd files")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--points", type=int, default=20000)
    parser.add_argument("--voxel", type=float, default=None)
    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--sinkhorn", type=int, default=120)
    parser.add_argument("--ortho_w", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")

    # prefilter controls
    parser.add_argument("--prefilter", action="store_true",
                        help="use XY-DBSCAN + z-level filtering before training")
    parser.add_argument("--pf_eps_xy", type=float, default=0.35)
    parser.add_argument("--pf_min_xy", type=int, default=40)
    parser.add_argument("--pf_z_bin", type=float, default=0.5)
    parser.add_argument("--pf_keep_levels", action="store_true",
                        help="keep only z-levels with >=2 clusters")
    parser.add_argument("--pf_drop_noise", action="store_true", help="drop DBSCAN noise (-1)")
    parser.add_argument("--add_normals", action="store_true",
                        help="estimate normals + curvature + FPFH as input features")

    # io
    parser.add_argument("--test_pcd", type=str, default=None)
    parser.add_argument("--out", type=str, default="outputs/pred_clusters.ply")
    parser.add_argument("--show", action="store_true")

    # sensible defaults ON
    parser.set_defaults(prefilter=True, pf_keep_levels=True, pf_drop_noise=True, add_normals=True)

    args = parser.parse_args()

    set_seed(123)
    enc, head, device = train_softclu(args)

    test_pcd = args.test_pcd
    if test_pcd is None:
        cand = sorted([str(p) for p in Path(args.data_dir).glob("*.pcd")])
        test_pcd = cand[0]
        print(f"No --test_pcd given, using {test_pcd}")

    run_inference_and_save(enc, head, device, test_pcd, args)


if __name__ == "__main__":
    main()
