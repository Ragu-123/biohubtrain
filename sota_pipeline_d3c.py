#!/usr/bin/env python3

"""

=================================================================================

🏆 CZ BIOHUB CELL TRACKING CHALLENGE — SOTA DUAL-GPU ENSEMBLE SUBMISSION PIPELINE

=================================================================================

Hardware: 2x Tesla T4 GPUs (cuda:0 & cuda:1) parallel multiprocessing

Input Pack: /kaggle/input/notebooks/ragunathravi/forantigravity/biohub-cell-tracking-solution.zip

Runtime: ~3.8 minutes across all 4 test volumes

=================================================================================

"""



import os

import sys



# Critical configuration for tracksdata & polars ABI compatibility

os.environ.setdefault("POLARS_PREFER_PKG", "32")



import time

import zipfile

import multiprocessing as mp

from pathlib import Path



import numpy as np

import polars as pl

import torch

import torch.nn.functional as F

import zarr



# --- 1. RESOLVE & UNPACK SOLUTION PACK ---

def resolve_solution_pack():

    zip_candidates = [

        Path("/kaggle/input/notebooks/ragunathravi/forantigravity/biohub-cell-tracking-solution.zip"),

        Path("/kaggle/input/forantigravity/biohub-cell-tracking-solution.zip"),

        Path("/kaggle/input/biohub-cell-tracking-solution/biohub-cell-tracking-solution.zip"),

        Path("/kaggle/working/biohub-cell-tracking-solution.zip"),

    ]

    dir_candidates = [

        Path("/kaggle/input/datasets/ragunathravi/forcompbiohub"),

        Path("/kaggle/input/forcompbiohub"),

        Path("/kaggle/working"),

        Path("/kaggle/working/support_pack"),

        Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),

        Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),

        Path("/tmp/biohub-cell-tracking-solution"),

        Path("/kaggle/input/notebooks/ragunathravi/forantigravity/biohub-cell-tracking-solution"),

        Path("/kaggle/input/forantigravity/biohub-cell-tracking-solution"),

        Path("/kaggle/input/biohub-cell-tracking-solution"),

        Path("/kaggle/working/biohub-cell-tracking-solution"),

    ]



    # Check direct directories

    for dc in dir_candidates:

        if (dc / "weights").exists() and (dc / "repo").exists():

            return dc

        # Also check if dc contains repo directly

        if (dc / "weights").exists() or (dc / "repo").exists():

            return dc



    # Unpack zip if found

    for zc in zip_candidates:

        if zc.exists():

            extract_dir = Path("/tmp/biohub-cell-tracking-solution")

            extract_dir.mkdir(parents=True, exist_ok=True)

            print(f"Unpacking {zc} -> /tmp ...")

            with zipfile.ZipFile(zc, "r") as zf:

                zf.extractall("/tmp")

            if (extract_dir / "weights").exists() and (extract_dir / "repo").exists():

                return extract_dir

            sub = extract_dir / "biohub-cell-tracking-solution"

            if (sub / "weights").exists() and (sub / "repo").exists():

                return sub

            if (Path("/tmp/weights")).exists() and (Path("/tmp/repo")).exists():

                return Path("/tmp")

            return extract_dir



    # Fallback search in /kaggle subdirectories (bounded to avoid slow recursive crawl)

    for base in [Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp")]:

        if not base.exists():

            continue

        try:

            for child in base.iterdir():

                if not child.is_dir() or "competitions" in child.name:

                    continue

                if (child / "weights").exists() and (child / "repo").exists():

                    return child

                try:

                    for sub in child.iterdir():

                        if sub.is_dir() and (sub / "weights").exists() and (sub / "repo").exists():

                            return sub

                except (PermissionError, OSError):

                    continue

        except (PermissionError, OSError):

            continue



    return Path("/kaggle/working")



SOLUTION_ROOT = resolve_solution_pack()

print(f"Using Solution Root: {SOLUTION_ROOT}")



# Register repo source paths

for candidate_root in [

    Path("/kaggle/input/datasets/ragunathravi/forcompbiohub"),

    Path("/kaggle/input/forcompbiohub"),

    SOLUTION_ROOT,

    Path("/kaggle/working"),

    Path("/kaggle/working/support_pack"),

]:

    for sub in ["repo/src", "repo/scripts", "src", "scripts"]:

        p = str(candidate_root / sub)

        if Path(p).exists() and p not in sys.path:

            sys.path.insert(0, p)



if "/kaggle/working" not in sys.path:

    sys.path.insert(0, "/kaggle/working")

if str(Path(__file__).resolve().parent) not in sys.path:

    sys.path.insert(0, str(Path(__file__).resolve().parent))



import contextlib

import logging

# Silence tracksdata Gurobi license check and warning traceback completely

logging.getLogger("tracksdata").setLevel(logging.ERROR)

logging.raiseExceptions = False



from biohub_tracking.io import open_dataset

from predict_unet_transformer import load_model, _load_frame, pool_kernel_from_um, _detect_cells_pooled

from train_unet_transformer import extract_pos_features, _POS_EMBED_DIM

import ilpy

import tracksdata as td

from tracksdata.solvers import _ilp_solver

from postprocess_clean import filter_output_graph



# Direct SCIP Solver backend: bypasses Gurobi check and eliminates traceback completely

def _direct_scip_solve(self):

    if self._count == 0:

        raise ValueError("Empty ILPSolver model, there is nothing to solve.")

    if len(self._edge_vars) == 0:

        raise ValueError("No edges found in the graph, there is nothing to solve.")



    solver = ilpy.Solver(

        num_variables=self._count,

        default_variable_type=ilpy.VariableType.Binary,

        preference=ilpy.Preference.Scip,

    )

    solver.set_num_threads(self.num_threads)

    solver.set_objective(self._objective)

    solver.set_constraints(self._constraints)

    solver.set_optimality_gap(self.gap)

    if self.timeout is not None:

        solver.set_timeout(self.timeout)

    solution = solver.solve()

    if solution is None:

        raise RuntimeError("Failed to solve the ILP problem with SCIP solver.")

    return solution



_ilp_solver.ILPSolver._solve = _direct_scip_solve





# Resolve model weights

def first_file(paths):

    for p in paths:

        if Path(p).exists():

            return Path(p)

    return Path(paths[0])



PRIMARY_WEIGHTS = first_file([

    Path("/kaggle/input/datasets/ragunathravi/forcompbiohub/weights/unet_transformer/split_0/edge_predictor_best.pth"),

    Path("/kaggle/input/forcompbiohub/weights/unet_transformer/split_0/edge_predictor_best.pth"),

    SOLUTION_ROOT / "weights/unet_transformer/split_0/edge_predictor_best.pth",

    Path("/kaggle/working/weights/unet_transformer/split_0/edge_predictor_best.pth"),

    Path("/kaggle/working/support_pack/weights/unet_transformer/split_0/edge_predictor_best.pth"),

    Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1/weights/unet_transformer/split_0/edge_predictor_best.pth"),

    Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/weights/unet_transformer/split_0/edge_predictor_best.pth"),

])



def resolve_seed_weights():

    candidates = [

        Path("/kaggle/input/datasets/ragunathravi/forcompbiohub/secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth"),

        Path("/kaggle/input/forcompbiohub/secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth"),

        Path("/kaggle/input/datasets/ragunathravi/forcompbiohub/weights/unet_transformer/split_1/edge_predictor_best.pth"),

        Path("/kaggle/input/forcompbiohub/weights/unet_transformer/split_1/edge_predictor_best.pth"),

        Path("/kaggle/input/datasets/ragunathravi/forcompbiohub/secondary_seed_weights/edge_predictor_best.pth"),

        Path("/kaggle/input/forcompbiohub/secondary_seed_weights/edge_predictor_best.pth"),

        Path("/kaggle/input/datasets/ragunathravi/forcompbiohub/secondary_seed_weights/weights/unet_transformer/split_0/edge_predictor_best.pth"),

        Path("/kaggle/input/forcompbiohub/secondary_seed_weights/weights/unet_transformer/split_0/edge_predictor_best.pth"),

        SOLUTION_ROOT / "weights/unet_transformer/split_1/edge_predictor_best.pth",

        Path("/kaggle/working/weights/unet_transformer/split_1/edge_predictor_best.pth"),

        Path("/kaggle/working/secondary_seed_weights/weights/unet_transformer/split_0/edge_predictor_best.pth"),

        Path("/kaggle/working/secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth"),

        Path("/kaggle/working/secondary_seed/weights/unet_transformer/split_0/edge_predictor_best.pth"),

        SOLUTION_ROOT / "weights/seed_314159/split_0/edge_predictor_best.pth",

        Path("/kaggle/working/seed_314159/weights/unet_transformer/split_0/edge_predictor_best.pth"),

        Path("/kaggle/input/biohub-temporal-unet3d-seed314159-v1/weights/unet_transformer/split_0/edge_predictor_best.pth"),

        Path("/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/weights/unet_transformer/split_0/edge_predictor_best.pth"),

    ]

    for c in candidates:

        if c.exists():

            return c

    for base in [Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp")]:

        if not base.exists():

            continue

        try:

            for child in base.iterdir():

                if not child.is_dir() or "competitions" in child.name:

                    continue

                for w in child.glob("**/edge_predictor_best.pth"):

                    if w.exists() and w != PRIMARY_WEIGHTS:

                        return w

        except (PermissionError, OSError):

            continue

    return PRIMARY_WEIGHTS



SEED_WEIGHTS = resolve_seed_weights()



def find_test_dir():

    candidates = [

        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/test"),

        Path("/kaggle/input/biohub-cell-tracking-during-development/test"),

        Path("/kaggle/working/test"),

    ]

    for c in candidates:

        if c.exists() and len(list(c.glob("*.zarr"))) > 0:

            return c

    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:

        if not base.exists():

            continue

        try:

            for child in base.iterdir():

                test_sub = child / "test"

                if test_sub.is_dir() and len(list(test_sub.glob("*.zarr"))) > 0:

                    return test_sub

        except (PermissionError, OSError):

            continue

    return candidates[0]



TEST_DIR = (Path(os.environ["SOTA_TEST_DIR"]) if os.environ.get("SOTA_TEST_DIR") else find_test_dir())

OUTPUT_CSV = Path("/kaggle/working/submission.csv")



print(f"Primary Weights : {PRIMARY_WEIGHTS} (exists: {PRIMARY_WEIGHTS.exists()})")

print(f"Seed Weights    : {SEED_WEIGHTS} (exists: {SEED_WEIGHTS.exists()})")

print(f"Test Directory  : {TEST_DIR} (exists: {TEST_DIR.exists()})")



# Optimal Hyperparameters (Calibrated against FN Error Audit)

DET_THRESHOLD        = 0.96875

POOL_KERNEL_UM       = 3.0

EDGE_STRONG_THRESH   = 0.40

EDGE_MIN_THRESH      = 0.10

EDGE_TOPK_PARENTS    = 5

EDGE_MAX_DISTANCE_UM = 15.0



ILP_EDGE_WEIGHT      = -1.0

ILP_APPEAR_WEIGHT    = 0.0

ILP_DISAPPEAR_WEIGHT = 2.0

ILP_DIVISION_WEIGHT  = 0.60





def detect_and_refine_peaks(prob_map: torch.Tensor, t: int, threshold: float, pool_k: tuple) -> np.ndarray:

    """

    3D Continuous Sub-Voxel Peak Detection & Parabolic Fitting.

    Eliminates centroid quantization lattice errors across anisotropic Z and 4x downsampled XY.

    """

    # Normalize tensor shape to exactly (1, 1, Z, Y, X)

    while prob_map.ndim < 5:

        prob_map = prob_map.unsqueeze(0)

    while prob_map.ndim > 5:

        prob_map = prob_map.squeeze(0)



    pad = tuple(k // 2 for k in pool_k)

    pooled = F.max_pool3d(prob_map, pool_k, stride=1, padding=pad)

    is_peak = (prob_map == pooled) & (prob_map > threshold)

    peak_idx = torch.nonzero(is_peak[0, 0])

    if peak_idx.shape[0] == 0:

        return np.empty((0, 4), dtype=np.float32)



    # 3D Continuous Sub-Voxel Parabolic Refinement

    prob_3d = prob_map[0, 0]

    prob_pad = F.pad(prob_3d.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]

    pz = peak_idx[:, 0] + 1

    py = peak_idx[:, 1] + 1

    px = peak_idx[:, 2] + 1

    v0 = prob_pad[pz, py, px]



    # X axis

    vx_m = prob_pad[pz, py, px - 1]

    vx_p = prob_pad[pz, py, px + 1]

    denom_x = 2.0 * (vx_m - 2.0 * v0 + vx_p)

    delta_x = torch.where(denom_x < -1e-6, (vx_m - vx_p) / denom_x, torch.zeros_like(v0))

    delta_x = torch.clamp(delta_x, -0.5, 0.5)



    # Y axis

    vy_m = prob_pad[pz, py - 1, px]

    vy_p = prob_pad[pz, py + 1, px]

    denom_y = 2.0 * (vy_m - 2.0 * v0 + vy_p)

    delta_y = torch.where(denom_y < -1e-6, (vy_m - vy_p) / denom_y, torch.zeros_like(v0))

    delta_y = torch.clamp(delta_y, -0.5, 0.5)



    # Z axis

    vz_m = prob_pad[pz - 1, py, px]

    vz_p = prob_pad[pz + 1, py, px]

    denom_z = 2.0 * (vz_m - 2.0 * v0 + vz_p)

    delta_z = torch.where(denom_z < -1e-6, (vz_m - vz_p) / denom_z, torch.zeros_like(v0))

    delta_z = torch.clamp(delta_z, -0.5, 0.5)



    delta = torch.stack([delta_z, delta_y, delta_x], dim=-1)

    refined = (peak_idx.float() + delta).cpu().numpy()

    t_col = np.full((len(refined), 1), t, dtype=np.float32)

    return np.hstack([t_col, refined])





def compute_fwd_and_harmonic(

    model, f_src, f_tgt,

    p_coords_src_ds, p_coords_tgt_ds,

    p_pos_src, p_pos_tgt,

    p_mask_src, p_mask_tgt,

    w_rev: float = 0.15,

):

    """

    Computes forward and reverse edge logits with distribution moment alignment

    and asymmetric harmonic mean soft-veto (w_rev=0.15).

    """

    fwd_logits = model.predict_edges(

        f_src, f_tgt,

        p_coords_src_ds, p_coords_tgt_ds,

        p_pos_src, p_pos_tgt,

        p_mask_src, p_mask_tgt,

    )

    rev_logits_native = model.predict_edges(

        f_tgt, f_src,

        p_coords_tgt_ds, p_coords_src_ds,

        p_pos_tgt, p_pos_src,

        p_mask_tgt, p_mask_src,

    )

    rev_logits = rev_logits_native.transpose(1, 2)



    # Moment alignment: scale reverse distribution to forward scale

    fwd_mean, fwd_std = fwd_logits.mean(), fwd_logits.std().clamp_min(1e-4)

    rev_mean, rev_std = rev_logits.mean(), rev_logits.std().clamp_min(1e-4)

    rev_aligned = (rev_logits - rev_mean) * (fwd_std / rev_std).clamp(0.5, 2.0) + fwd_mean



    prob_fwd = torch.softmax(fwd_logits[0].float(), dim=0).clamp_min(1e-8)

    prob_rev = torch.softmax(rev_aligned[0].float(), dim=0).clamp_min(1e-8)



    # Asymmetric harmonic mean

    p_harm = 1.0 / ((1.0 - w_rev) / prob_fwd + w_rev / prob_rev)

    p_harm = p_harm / p_harm.sum(dim=0, keepdim=True).clamp_min(1e-8)

    return p_harm.cpu().numpy()





def apply_kinematics(

    cand_list: list[tuple[int, int, float, float]],

    coords_dict: dict[int, np.ndarray],

    predecessor_map: dict[int, int],

    lambda_v: float = 0.60,

    sigma_kine: float = 4.5,

    gamma_align: float = 0.35,

    beta_kine: float = 0.40,

):

    """

    Modulates candidate edge probabilities using continuous velocity momentum deflection

    and directional cosine alignment in physical space.

    """

    modulated = []

    for gi, gj, p, dist in cand_list:

        pos_i = coords_dict[gi]

        pos_j = coords_dict[gj]

        disp = pos_j - pos_i

        disp_norm = float(np.linalg.norm(disp))



        pred_id = predecessor_map.get(gi)

        if pred_id is not None and pred_id in coords_dict and disp_norm > 1e-4:

            pos_prev = coords_dict[pred_id]

            v_prev = pos_i - pos_prev

            v_norm = float(np.linalg.norm(v_prev))

            if v_norm > 1e-4:

                pred_pos = pos_i + lambda_v * v_prev

                motion_resid = float(np.linalg.norm(pos_j - pred_pos))

                cos_theta = float(np.dot(disp, v_prev) / (disp_norm * v_norm))

                cos_factor = ((1.0 + np.clip(cos_theta, -1.0, 1.0)) / 2.0) ** gamma_align

                kine_mult = float(np.exp(-(motion_resid ** 2) / (2.0 * (sigma_kine ** 2))) * cos_factor)

            else:

                kine_mult = 1.0

        else:

            kine_mult = 1.0



        p_mod = float(p * ((1.0 - beta_kine) + beta_kine * kine_mult))

        modulated.append((gi, gj, p_mod, dist))

    return modulated





@torch.no_grad()

def process_single_volume(ds_path: Path, device: torch.device, m0, m1, window_size: int, downsample: tuple):

    t0 = time.time()

    stem = ds_path.stem

    print(f"[{device}] Starting inference on {stem}...", flush=True)



    ds = open_dataset(ds_path, normalize=False, load_image=False, downsample=downsample)

    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]

    q_low = float(ds.quantiles["0.001"])

    q_high = float(ds.quantiles["0.999"])

    

    T = ds.image_shape[0]

    image_shape = (T,) + ds.image_shape[1:]

    target_shape = list(image_shape[1:])

    scale = np.array(ds.scale, dtype=np.float32)

    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))

    raw_voxel_size = np.asarray(voxel_size, dtype=np.float64) / np.asarray(downsample, dtype=np.float64)

    pool_k = pool_kernel_from_um(POOL_KERNEL_UM, voxel_size)

    

    ds_arr_np = np.array(downsample, dtype=np.float32)

    ds_arr_t = torch.from_numpy(ds_arr_np).to(device)



    stride = max(window_size - 1, 1)

    window_starts = list(range(0, T - window_size + 1, stride))

    if not window_starts or window_starts[-1] + window_size < T:

        last = max(T - window_size, 0)

        if not window_starts or last != window_starts[-1]:

            window_starts.append(last)



    seen_frames = set()

    seen_pairs = set()

    coord_lists = []

    coord_offset = {}

    global_node_count = 0

    candidate_edges = []



    for ws in window_starts:

        frame_indices = list(range(ws, ws + window_size))

        imgs = torch.stack([_load_frame(zarr_arr, t, target_shape, downsample) for t in frame_indices])

        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0).to(device)



        out0, det0 = m0.encode(imgs)

        out1, det1 = m1.encode(imgs)



        # 4-fold Flip-XY TTA

        for dims in [(-1,), (-2,), (-2, -1)]:

            imgs_flip = imgs.flip(dims)

            _, d0_flip = m0.encode(imgs_flip)

            _, d1_flip = m1.encode(imgs_flip)

            for f in range(window_size):

                det0[f] = det0[f] + d0_flip[f].flip(dims)

                det1[f] = det1[f] + d1_flip[f].flip(dims)

            del imgs_flip, d0_flip, d1_flip



        det_fused = [(det0[f] + det1[f]) / 8.0 for f in range(window_size)]

        del imgs



        for f_idx, t in enumerate(frame_indices):

            if t not in seen_frames:

                arr = detect_and_refine_peaks(det_fused[f_idx][0], t, DET_THRESHOLD, pool_k)

                coord_offset[t] = (global_node_count, global_node_count + len(arr))

                global_node_count += len(arr)

                coord_lists.append(arr)

                seen_frames.add(t)



        coords_so_far = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.float32)



        for f_idx in range(window_size - 1):

            t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]

            if (t_src, t_tgt) in seen_pairs:

                continue

            seen_pairs.add((t_src, t_tgt))

            if t_src not in coord_offset or t_tgt not in coord_offset:

                continue

            s_src, e_src = coord_offset[t_src]

            s_tgt, e_tgt = coord_offset[t_tgt]

            if e_src == s_src or e_tgt == s_tgt:

                continue



            c_src = coords_so_far[s_src:e_src]

            c_tgt = coords_so_far[s_tgt:e_tgt]

            n_src, n_tgt = len(c_src), len(c_tgt)



            p_coords_src = torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(device)

            p_coords_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)

            c_src_rel = c_src.copy()

            c_src_rel[:, 0] = f_idx

            c_tgt_rel = c_tgt.copy()

            c_tgt_rel[:, 0] = f_idx + 1

            window_shape = (window_size,) + image_shape[1:]

            p_pos_src = torch.from_numpy(extract_pos_features(c_src_rel, window_shape)).unsqueeze(0).to(device)

            p_pos_tgt = torch.from_numpy(extract_pos_features(c_tgt_rel, window_shape)).unsqueeze(0).to(device)

            p_mask_src = torch.ones(1, n_src, dtype=torch.bool, device=device)

            p_mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=device)



            feat0_src = m0._index_features(out0[:, f_idx], p_coords_src, p_mask_src)

            feat0_tgt = m0._index_features(out0[:, f_idx + 1], p_coords_tgt, p_mask_tgt)

            p0 = compute_fwd_and_harmonic(

                m0, feat0_src, feat0_tgt,

                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,

                p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt, w_rev=0.15

            )



            feat1_src = m1._index_features(out1[:, f_idx], p_coords_src, p_mask_src)

            feat1_tgt = m1._index_features(out1[:, f_idx + 1], p_coords_tgt, p_mask_tgt)

            p1 = compute_fwd_and_harmonic(

                m1, feat1_src, feat1_tgt,

                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,

                p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt, w_rev=0.15

            )



            p_ens = 0.50 * p0 + 0.50 * p1



            # Candidate edge generation for ILP

            cand_pairs = set()

            strong = np.argwhere(p_ens >= EDGE_STRONG_THRESH)

            for si, tj in strong:

                cand_pairs.add((int(si), int(tj)))

            top_k = min(EDGE_TOPK_PARENTS, n_src)

            for tj in range(n_tgt):

                col_p = p_ens[:, tj]

                top_sources = np.argpartition(col_p, -top_k)[-top_k:] if top_k < n_src else np.arange(n_src)

                for si in top_sources:

                    if float(col_p[si]) >= EDGE_MIN_THRESH:

                        cand_pairs.add((int(si), int(tj)))



            c_src_xyz = c_src[:, 1:].astype(np.float32)

            c_tgt_xyz = c_tgt[:, 1:].astype(np.float32)



            for si, tj in cand_pairs:

                dist = float(np.linalg.norm((c_src_xyz[si] - c_tgt_xyz[tj]) * raw_voxel_size))

                if dist <= EDGE_MAX_DISTANCE_UM:

                    candidate_edges.append((s_src + si, s_tgt + tj, float(p_ens[si, tj]), dist))



        del out0, out1



    coords_orig = coords_so_far.astype(np.float64)

    coords_orig[:, 1:] = coords_orig[:, 1:] * ds_arr_np



    # Apply Kinematic Momentum Prior in physical space

    coords_phys = {

        i: coords_orig[i, 1:] * scale

        for i in range(len(coords_orig))

    }

    s_edges = sorted(candidate_edges, key=lambda x: (coords_orig[x[0], 0], -x[2]))

    pred_map, used = {}, set()

    for gi, gj, p, d in s_edges:

        if gj not in used and p >= 0.40:

            pred_map[gj] = gi

            used.add(gj)

    candidate_edges = apply_kinematics(candidate_edges, coords_phys, pred_map)

    # --- D3C-EXP: dump candidate edges + node coords (um) for division calibration ---
    try:
        _fd = os.environ.get('FORKDUMP_DIR', '/kaggle/working/forkdumps')
        os.makedirs(_fd, exist_ok=True)
        _cand = np.asarray(candidate_edges, dtype=np.float64)  # (E,4): src,tgt,prob,dist
        _um = np.stack([coords_phys[i] for i in range(len(coords_orig))])
        _coords_um = np.concatenate([coords_orig[:, :1], _um], axis=1)  # (N,4): t,z,y,x in um
        np.savez_compressed(os.path.join(_fd, f'{stem}_d3c.npz'),
                            cand=_cand, coords_um=_coords_um,
                            scale=np.asarray(scale, dtype=np.float64))
        print(f'[dump] {stem}: {_cand.shape[0]} cand edges, {_coords_um.shape[0]} nodes', flush=True)
    except Exception as _e:
        print(f'[dump] FAILED {stem}: {_e}', flush=True)
    # --- end D3C-EXP ---



    print(f"[{device}] {stem}: {len(coords_orig)} raw nodes, {len(candidate_edges)} candidate edges. Running global ILP...", flush=True)



    ilp_graph = td.graph.InMemoryGraph()

    for k in ["z", "y", "x"]:

        ilp_graph.add_node_attr_key(k, pl.Float64, 0.0)

    ilp_nids = ilp_graph.bulk_add_nodes([

        {"t": int(c[0]), "z": float(c[1]), "y": float(c[2]), "x": float(c[3])}

        for c in coords_orig

    ])

    ilp_graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)

    ilp_graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)

    ilp_graph.bulk_add_edges([

        {

            "source_id": ilp_nids[gi],

            "target_id": ilp_nids[gj],

            "edge_prob": p,

            "edge_dist": d,

        }

        for gi, gj, p, d in candidate_edges

    ])



    t_ilp = time.time()

    solver = td.solvers.ILPSolver(

        edge_weight=ILP_EDGE_WEIGHT * td.EdgeAttr("edge_prob"),

        appearance_weight=ILP_APPEAR_WEIGHT,

        disappearance_weight=ILP_DISAPPEAR_WEIGHT,

        division_weight=ILP_DIVISION_WEIGHT,

        num_threads=2,

    )

    with contextlib.redirect_stdout(None):

        solved_graph = solver.solve(ilp_graph)

    if hasattr(solved_graph, "detach"):

        solved_graph = solved_graph.detach()

    print(f"[{device}] {stem}: Global SCIP ILP solved in {time.time() - t_ilp:.1f}s.", flush=True)



    nodes_by_id = {}

    for r in solved_graph.node_attrs().iter_rows(named=True):

        nid = int(r["node_id"])

        nodes_by_id[nid] = {

            "node_id": nid,

            "t": int(r["t"]),

            "z": float(r["z"]),

            "y": float(r["y"]),

            "x": float(r["x"]),

        }

    raw_edges = []

    for r in solved_graph.edge_attrs().iter_rows(named=True):

        raw_edges.append({

            "source_id": int(r["source_id"]),

            "target_id": int(r["target_id"]),

            "edge_prob": float(r.get("edge_prob", 0.9)),

            "distance_um": float(r.get("edge_dist", 0.0)),

        })



    filt_nodes, filt_edges, stats = filter_output_graph(nodes_by_id, raw_edges, dataset=stem)

    dt = time.time() - t0

    pruned = len(nodes_by_id) - len(filt_nodes)

    print(f"[{device}] Finished {stem} in {dt:.1f}s: {len(filt_nodes)} nodes, {len(filt_edges)} edges (pruned {pruned} noisy nodes, recovered {stats.get('gap_closed_single', 0) + stats.get('gap2_recovered', 0)} gap edges).", flush=True)

    return stem, filt_nodes, filt_edges





def gpu_worker(gpu_id: int, volume_paths: list[Path], return_dict):

    device = torch.device(f"cuda:{gpu_id}")

    print(f"Worker for {device} initialized with {len(volume_paths)} volume(s).", flush=True)



    m0, window_size, downsample = load_model(PRIMARY_WEIGHTS, device)

    m1, _, _ = load_model(SEED_WEIGHTS, device)



    worker_results = []

    for vp in volume_paths:

        res = process_single_volume(vp, device, m0, m1, window_size, downsample)

        worker_results.append(res)



    return_dict[gpu_id] = worker_results





def build_submission_dataframe(all_results):

    print("\nCompiling final submission dataframe...", flush=True)

    all_results = sorted(all_results, key=lambda x: x[0])

    dfs = []

    seen_stems = set()

    for stem, filt_nodes, filt_edges in all_results:

        if stem in seen_stems:

            print(f"Skipping duplicate result for {stem}", flush=True)

            continue

        seen_stems.add(stem)

        node_id_map = {}

        node_records = []

        # Sort nodes deterministically by frame time then original ID

        sorted_nodes = sorted(filt_nodes.items(), key=lambda item: (int(item[1]["t"]), item[0]))

        for new_id, (old_id, node) in enumerate(sorted_nodes, start=1):

            node_id_map[old_id] = new_id

            node_records.append({

                "dataset": stem,

                "row_type": "node",

                "node_id": new_id,

                "t": int(node["t"]),

                "z": max(0, int(round(float(node["z"])))),

                "y": max(0, int(round(float(node["y"])))),

                "x": max(0, int(round(float(node["x"])))),

                "source_id": -1,

                "target_id": -1,

            })



        edge_records = []

        for edge in filt_edges:

            src = node_id_map.get(int(edge["source_id"]))

            tgt = node_id_map.get(int(edge["target_id"]))

            if src is not None and tgt is not None:

                edge_records.append({

                    "dataset": stem,

                    "row_type": "edge",

                    "node_id": -1,

                    "t": -1,

                    "z": -1,

                    "y": -1,

                    "x": -1,

                    "source_id": src,

                    "target_id": tgt,

                })

        # Sort edges deterministically by source then target

        edge_records = sorted(edge_records, key=lambda e: (e["source_id"], e["target_id"]))



        vol_nodes_df = pl.DataFrame(node_records, schema={

            "dataset": pl.Utf8, "row_type": pl.Utf8, "node_id": pl.Int64,

            "t": pl.Int64, "z": pl.Int64, "y": pl.Int64, "x": pl.Int64,

            "source_id": pl.Int64, "target_id": pl.Int64,

        })

        vol_edges_df = pl.DataFrame(edge_records, schema={

            "dataset": pl.Utf8, "row_type": pl.Utf8, "node_id": pl.Int64,

            "t": pl.Int64, "z": pl.Int64, "y": pl.Int64, "x": pl.Int64,

            "source_id": pl.Int64, "target_id": pl.Int64,

        })

        dfs.append(pl.concat([vol_nodes_df, vol_edges_df]))



    if not dfs:

        # Fallback empty dataframe matching schema

        return pl.DataFrame({

            "id": pl.Series(dtype=pl.Int64),

            "dataset": pl.Series(dtype=pl.Utf8),

            "row_type": pl.Series(dtype=pl.Utf8),

            "node_id": pl.Series(dtype=pl.Int64),

            "t": pl.Series(dtype=pl.Int64),

            "z": pl.Series(dtype=pl.Int64),

            "y": pl.Series(dtype=pl.Int64),

            "x": pl.Series(dtype=pl.Int64),

            "source_id": pl.Series(dtype=pl.Int64),

            "target_id": pl.Series(dtype=pl.Int64),

        })



    final_df = pl.concat(dfs)

    final_df = final_df.with_columns(pl.arange(0, final_df.height).alias("id"))

    final_df = final_df.select(["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])

    return final_df





def main():

    print("=================================================================")

    print("🚀 LAUNCHING SOTA DUAL-GPU SUBMISSION PIPELINE")

    print("=================================================================")

    start_time = time.time()



    all_zarrs = sorted(list(TEST_DIR.glob("*.zarr")))

    print(f"Found {len(all_zarrs)} test volume(s): {[z.name for z in all_zarrs]}")



    n_gpus = torch.cuda.device_count()

    print(f"Available GPUs: {n_gpus}")



    if n_gpus >= 2 and len(all_zarrs) >= 2:

        gpu0_vols = all_zarrs[::2]

        gpu1_vols = all_zarrs[1::2]

        print(f"cuda:0 tasks ({len(gpu0_vols)}): {[z.name for z in gpu0_vols]}")

        print(f"cuda:1 tasks ({len(gpu1_vols)}): {[z.name for z in gpu1_vols]}")



        manager = mp.Manager()

        return_dict = manager.dict()



        p0 = mp.Process(target=gpu_worker, args=(0, gpu0_vols, return_dict))

        p1 = mp.Process(target=gpu_worker, args=(1, gpu1_vols, return_dict))



        p0.start()

        p1.start()

        p0.join()

        p1.join()



        if p0.exitcode != 0 or p1.exitcode != 0:

            raise RuntimeError(f"GPU Worker process failed! Exit codes: GPU0={p0.exitcode}, GPU1={p1.exitcode}")



        all_results = list(return_dict.get(0, [])) + list(return_dict.get(1, []))

        if len(all_results) != len(all_zarrs):

            raise RuntimeError(f"Expected {len(all_zarrs)} test volume results, got {len(all_results)}!")

    else:

        print("Running on single GPU or single volume...")

        manager = mp.Manager()

        return_dict = manager.dict()

        gpu_worker(0, all_zarrs, return_dict)

        all_results = list(return_dict.get(0, []))

        if len(all_results) != len(all_zarrs):

            raise RuntimeError(f"Expected {len(all_zarrs)} test volume results, got {len(all_results)}!")



    sub_df = build_submission_dataframe(all_results)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    sub_df.write_csv(OUTPUT_CSV)

    elapsed = time.time() - start_time



    print("\n=================================================================")

    print(f"✅ SUBMISSION GENERATED SUCCESSFULLY in {elapsed:.1f}s ({elapsed/60:.2f} mins)!")

    print(f"Path: {OUTPUT_CSV}")

    print(f"Total Rows: {sub_df.height:,}")

    print(f"Summary by row type: {sub_df['row_type'].value_counts().to_dicts()}")

    print(f"Summary by dataset: {sub_df['dataset'].value_counts().to_dicts()}")

    print("=================================================================")





if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    main()

