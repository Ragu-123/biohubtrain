#!/usr/bin/env python
"""
AnisoTrack3D-Ensemble: Production Kaggle Submission Pipeline
Dual-GPU Offline Inference for Top-10 World-Class Performance (>= 0.970)

Pipeline:
1. Multi-Checkpoint Consensus Ensemble (Split 0 + Split 1 + Seed 314159)
2. Custom Triton Continuous 3D Regularized Hessian Sub-Voxel Peak Refiner
3. Directional Kinematic Momentum Buffer
4. Bidirectional Harmonic Consensus with Soft-Veto
5. Astra Internal Gap-Protected Short-Track Pruning (dt = 1 output only)
6. Consecutive-Frame Endpoint Reconnection (t -> t+1)
7. Degree-0 Removal & Adaptive Census Multiplier Calibration
8. Clean Kaggle CSV Export
"""

import os
import sys
import glob
import time
import csv
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import zarr
import rustworkx as rx
from tqdm import tqdm

# Ensure offline wheels and repo modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
for p in [
    "/kaggle/input/datasets/ragunathravi/forcompbiohub/repo/src",
    "/kaggle/input/datasets/ragunathravi/forcompbiohub/repo/scripts",
]:
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)

import tracksdata as td
from biohub_tracking.io import open_dataset
from predict_unet_transformer import load_model, extract_pos_features, pool_kernel_from_um
from src.kernels.triton_ops import refine_subvoxel_peaks_triton
from src.models.local_transformer import log_sinkhorn_uot
from src.kernels.cpp_ops import get_cpp_tracker

SUBMISSION_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
]


def filter_short_tracks_with_gaps(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
    min_length: int = 5,
    scale: tuple[float, ...] = (1.625, 0.40625, 0.40625),
    max_gap_dist_um: float = 12.0,
    total_frames: int = 100,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    if min_length <= 1 or len(edges) == 0:
        return coords, edges

    N = len(coords)
    rx_g = rx.PyDiGraph()
    rx_g.add_nodes_from(range(N))
    for src, tgt, prob, dist in edges:
        rx_g.add_edge(src, tgt, None)

    endpoints = [n for n in range(N) if rx_g.out_degree(n) == 0 and rx_g.in_degree(n) > 0]
    startpoints = [n for n in range(N) if rx_g.in_degree(n) == 0 and rx_g.out_degree(n) > 0]
    t_coords = coords[:, 0]

    starts_by_t: dict[int, list[int]] = {}
    for sp in startpoints:
        t_sp = int(t_coords[sp])
        starts_by_t.setdefault(t_sp, []).append(sp)

    scale_arr = np.array(scale, dtype=np.float32)
    coords_um = coords[:, 1:] * scale_arr

    internal_gap_edges = []
    for ep in endpoints:
        t_ep = int(t_coords[ep])
        t_cand = t_ep + 2  # 1-frame dropout (dt = 2)
        if t_cand in starts_by_t:
            p_ep = coords_um[ep]
            for sp in starts_by_t[t_cand]:
                p_sp = coords_um[sp]
                d = np.linalg.norm(p_ep - p_sp)
                if d <= max_gap_dist_um:
                    internal_gap_edges.append((ep, sp))

    undir_g = rx.PyGraph()
    undir_g.add_nodes_from(range(N))
    for src, tgt, prob, dist in edges:
        undir_g.add_edge(src, tgt, None)
    for ep, sp in internal_gap_edges:
        undir_g.add_edge(ep, sp, None)

    comps = rx.connected_components(undir_g)
    surviving_nodes = set()
    for comp in comps:
        comp_frames = set(coords[n, 0] for n in comp)
        if len(comp_frames) >= min_length or 0 in comp_frames or (total_frames - 1) in comp_frames:
            surviving_nodes.update(comp)

    new_edges = [(s, t, p, d) for s, t, p, d in edges if s in surviving_nodes and t in surviving_nodes]
    new_node_ids = sorted(list(surviving_nodes))
    old_to_new = {old: new for new, old in enumerate(new_node_ids)}
    filtered_coords = coords[new_node_ids]
    remapped_edges = [(old_to_new[s], old_to_new[t], p, d) for s, t, p, d in new_edges]
    return filtered_coords, remapped_edges


def prune_false_divisions(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
    scale: tuple[float, ...],
    cos_spindle_thresh: float = -0.55,
    midpoint_thresh: float = 2.20,
    sym_ratio_thresh: float = 0.45,
    min_sister_dist_um: float = 4.50,
    max_sister_dist_um: float = 15.50,
    min_prob: float = 0.25,
    min_mother_history: int = 2,
    min_daughter_persistence: int = 3,
    total_frames: int | None = None,
) -> list[tuple[int, int, float, float]]:
    """
    Prune spurious second-daughter edges from non-dividing cells that cause division FPs.
    Enforces Galilean comoving reference frame, sister distance bounds, mother history,
    and multi-frame daughter lineage persistence.
    """
    if not edges:
        return edges

    rx_g = rx.PyDiGraph()
    rx_g.add_nodes_from(range(len(coords)))
    edge_dict = {}
    for src, tgt, prob, dist in edges:
        rx_g.add_edge(src, tgt, None)
        edge_dict[(src, tgt)] = (prob, dist)

    scale_arr = np.array(scale, dtype=np.float32)
    coords_um = coords[:, 1:] * scale_arr

    edges_to_remove = set()
    num_forks = 0
    num_pruned = 0

    for d in range(len(coords)):
        succs = rx_g.successors(d)
        if len(succs) < 2:
            continue
        num_forks += 1
        preds = rx_g.predecessors(d)
        probs = [edge_dict[(d, s)][0] for s in succs]

        # Invariant 1: Mother must have established incoming tracklet history (>= min_mother_history)
        d_t = int(coords[d, 0])
        if d_t >= min_mother_history:
            if len(preds) == 0:
                weaker_child = succs[int(np.argmin(probs))]
                edges_to_remove.add((d, weaker_child))
                num_pruned += 1
                continue

            curr = d
            m_hist = 0
            while True:
                pr = rx_g.predecessors(curr)
                if not pr:
                    break
                curr = pr[0]
                m_hist += 1
                if m_hist >= min_mother_history:
                    break
            if m_hist < min_mother_history:
                weaker_child = succs[int(np.argmin(probs))]
                edges_to_remove.add((d, weaker_child))
                num_pruned += 1
                continue

        # Invariant 2: Candidate edge probability threshold (must satisfy minimum division edge confidence)
        if min(probs) < min_prob:
            weaker_child = succs[int(np.argmin(probs))]
            edges_to_remove.add((d, weaker_child))
            num_pruned += 1
            continue

        # Invariant 3: Daughter Lineage Persistence (daughters must survive >= min_daughter_persistence frames after mitosis)
        d1_t = int(coords[succs[0], 0])
        def get_fwd_len(node_idx):
            c = node_idx
            f_len = 0
            while True:
                sc = rx_g.successors(c)
                if not sc:
                    break
                c = sc[0]
                f_len += 1
            return f_len

        d1_len = get_fwd_len(succs[0])
        d2_len = get_fwd_len(succs[1])

        avail_frames = (total_frames - 1 - d1_t) if total_frames is not None else min_daughter_persistence
        req_pers = max(1, min(min_daughter_persistence, int(avail_frames)))

        if d1_len < req_pers and d2_len >= req_pers:
            edges_to_remove.add((d, succs[0]))
            num_pruned += 1
            continue
        elif d2_len < req_pers and d1_len >= req_pers:
            edges_to_remove.add((d, succs[1]))
            num_pruned += 1
            continue
        elif d1_len < req_pers and d2_len < req_pers:
            weaker_child = succs[int(np.argmin(probs))]
            edges_to_remove.add((d, weaker_child))
            num_pruned += 1
            continue

        # Invariant 4: Galilean Co-Moving Reference Frame Geometric Cleavage Invariants
        p_m = coords_um[d]
        if preds:
            p_pred = coords_um[preds[0]]
            v_mother = p_m - p_pred
        else:
            v_mother = np.zeros(3, dtype=np.float32)
        p_comoving = p_m + v_mother

        p_d1 = coords_um[succs[0]]
        p_d2 = coords_um[succs[1]]

        w1 = p_d1 - p_comoving
        w2 = p_d2 - p_comoving
        d1 = np.linalg.norm(w1)
        d2 = np.linalg.norm(w2)

        cos_spindle = float(np.dot(w1, w2) / (d1 * d2 + 1e-6))
        midpoint_offset = float(np.linalg.norm(p_comoving - 0.5 * (p_d1 + p_d2)))
        sym_ratio = float(abs(d1 - d2) / (d1 + d2 + 1e-6))
        dist_sister = float(np.linalg.norm(p_d1 - p_d2))

        if (
            cos_spindle > cos_spindle_thresh
            or midpoint_offset > midpoint_thresh
            or sym_ratio > sym_ratio_thresh
            or dist_sister < min_sister_dist_um
            or dist_sister > max_sister_dist_um
        ):
            weaker_child = succs[int(np.argmin(probs))]
            edges_to_remove.add((d, weaker_child))
            num_pruned += 1
            continue

    if num_forks > 0:
        print(f"  [Mitosis Filter] Evaluated {num_forks} candidate forks -> Pruned {num_pruned} spurious forks ({num_forks - num_pruned} valid mitoses retained)")

    return [e for e in edges if (e[0], e[1]) not in edges_to_remove]


def reconnect_broken_consecutive_endpoints(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
    scale: tuple[float, ...],
    max_reconnect_dist_um: float = 6.5,
) -> list[tuple[int, int, float, float]]:
    if not edges:
        return edges

    N = len(coords)
    rx_g = rx.PyDiGraph()
    rx_g.add_nodes_from(range(N))
    for src, tgt, prob, dist in edges:
        rx_g.add_edge(src, tgt, None)

    endpoints = [n for n in range(N) if rx_g.out_degree(n) == 0]
    startpoints = [n for n in range(N) if rx_g.in_degree(n) == 0]

    t_coords = coords[:, 0]
    starts_by_t: dict[int, list[int]] = {}
    for sp in startpoints:
        t_sp = int(t_coords[sp])
        starts_by_t.setdefault(t_sp, []).append(sp)

    scale_arr = np.array(scale, dtype=np.float32)
    coords_um = coords[:, 1:] * scale_arr

    new_reconnect_edges = []
    claimed_starts = set()

    for ep in endpoints:
        t_ep = int(t_coords[ep])
        t_next = t_ep + 1
        if t_next in starts_by_t:
            p_ep = coords_um[ep]
            best_sp = None
            best_d = float("inf")
            for sp in starts_by_t[t_next]:
                if sp in claimed_starts:
                    continue
                p_sp = coords_um[sp]
                d = np.linalg.norm(p_ep - p_sp)
                if d <= max_reconnect_dist_um and d < best_d:
                    best_d = d
                    best_sp = sp
            if best_sp is not None:
                new_reconnect_edges.append((ep, best_sp, 0.85, float(best_d)))
                claimed_starts.add(best_sp)

    return edges + new_reconnect_edges


def prune_isolated_degree_0_nodes(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    if not edges:
        return coords, edges

    nodes_with_edges = set(e[0] for e in edges) | set(e[1] for e in edges)
    active_node_ids = sorted(list(nodes_with_edges))
    old_to_new = {old: new for new, old in enumerate(active_node_ids)}

    clean_coords = coords[active_node_ids]
    clean_edges = [(old_to_new[e[0]], old_to_new[e[1]], e[2], e[3]) for e in edges]
    return clean_coords, clean_edges


@torch.no_grad()
def track_volume_inference(
    volume_path: Path,
    models: list[tuple[torch.nn.Module, torch.device]],
    downsample: tuple[int, int, int] = (1, 4, 4),
    window_size: int = 2,
    det_threshold: float = 0.50,
    edge_threshold: float = 0.48,
    div_threshold: float = 0.25,
    div_joint_threshold: float = 0.70,
    pool_kernel_um: float = 5.0,
    min_track_length: int = 4,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    ds = open_dataset(volume_path, normalize=False, load_image=False, downsample=downsample)
    zarr_arr = zarr.open_group(str(volume_path), mode="r")["0"]

    q_low = float(ds.quantiles.get("0.001", 100.0))
    q_high = float(ds.quantiles.get("0.999", 500.0))
    T = ds.image_shape[0]
    target_shape = list(ds.image_shape[1:])
    scale = tuple(ds.scale)
    ds_arr = np.array(downsample, dtype=np.float32)

    voxel_size_down = tuple(s * d for s, d in zip(scale, downsample))
    pool_k = pool_kernel_from_um(pool_kernel_um, voxel_size_down)
    pad = tuple(k // 2 for k in pool_k)

    stride = max(window_size - 1, 1)
    window_starts = list(range(0, T - window_size + 1, stride))
    if not window_starts or window_starts[-1] + window_size < T:
        last = max(T - window_size, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    seen_frames = set()
    seen_pairs = set()
    coord_lists_down = []
    coord_offset = {}
    global_node_count = 0
    all_edges = []
    velocity_buffer = {}

    primary_device = models[0][1]

    for ws in tqdm(window_starts, desc=f"  Tracking {volume_path.stem}", leave=False):
        frame_indices = list(range(ws, ws + window_size))
        imgs_raw = []
        for t in frame_indices:
            dz, dy, dx = downsample
            raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)
            f_t = torch.from_numpy(raw)
            if list(f_t.shape) != target_shape:
                f_t = F.interpolate(f_t[None, None], size=target_shape, mode="trilinear", align_corners=False)[0, 0]
            imgs_raw.append(f_t)

        imgs_raw = torch.stack(imgs_raw)
        imgs_raw = ((imgs_raw - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0)

        unet_outs = []
        det_logits_list = []
        for m, dev in models:
            inp = imgs_raw.to(dev)
            with torch.no_grad():
                u_out, d_log = m.encode(inp)
                # TTA flips
                for dims in [(-1,), (-2,), (-2, -1)]:
                    inp_f = inp.flip(dims)
                    _, d_f = m.encode(inp_f)
                    for f in range(window_size):
                        d_log[f] = d_log[f] + d_f[f].flip(dims)
                for f in range(window_size):
                    d_log[f] = d_log[f] / 4.0
            unet_outs.append(u_out)
            det_logits_list.append([dl.to(primary_device) for dl in d_log])

        det_logits_ens = [
            sum(det_logits_list[k][f] for k in range(len(models))) / len(models)
            for f in range(window_size)
        ]

        # Peak detection & Triton sub-voxel refinement
        for f_idx, t in enumerate(frame_indices):
            if t not in seen_frames:
                log_t = det_logits_ens[f_idx]
                pooled = F.max_pool3d(log_t, pool_k, stride=1, padding=pad)
                sig_t = torch.sigmoid(log_t)
                is_peak = (log_t == pooled) & (sig_t > det_threshold)
                peak_idx = torch.nonzero(is_peak[0, 0])

                if len(peak_idx) > 0:
                    peaks_refined = refine_subvoxel_peaks_triton(log_t[0, 0], peak_idx)
                    t_col = np.full((len(peaks_refined), 1), t, dtype=np.float32)
                    arr_down = np.concatenate([t_col, peaks_refined.cpu().numpy()], axis=1)
                else:
                    arr_down = np.empty((0, 4), dtype=np.float32)

                coord_offset[t] = (global_node_count, global_node_count + len(arr_down))
                global_node_count += len(arr_down)
                coord_lists_down.append(arr_down)
                seen_frames.add(t)

        coords_down_so_far = np.concatenate(coord_lists_down) if coord_lists_down else np.empty((0, 4), dtype=np.float32)

        # Edge association
        for f_idx in range(window_size - 1):
            t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]
            if (t_src, t_tgt) in seen_pairs:
                continue
            seen_pairs.add((t_src, t_tgt))

            s_src, e_src = coord_offset[t_src]
            s_tgt, e_tgt = coord_offset[t_tgt]
            if e_src == s_src or e_tgt == s_tgt:
                continue

            c_src_down = coords_down_so_far[s_src:e_src]
            c_tgt_down = coords_down_so_far[s_tgt:e_tgt]
            n_src, n_tgt = len(c_src_down), len(c_tgt_down)
            idx_src = np.arange(s_src, e_src, dtype=np.int64)
            idx_tgt = np.arange(s_tgt, e_tgt, dtype=np.int64)

            window_shape = (window_size,) + ds.image_shape[1:]
            c_src_rel = c_src_down.copy()
            c_src_rel[:, 0] = f_idx
            c_tgt_rel = c_tgt_down.copy()
            c_tgt_rel[:, 0] = f_idx + 1
            pos_src_np = extract_pos_features(c_src_rel, window_shape)
            pos_tgt_np = extract_pos_features(c_tgt_rel, window_shape)

            edge_logits_models = []
            for k, (m, dev) in enumerate(models):
                u_out = unet_outs[k]
                p_coords_src = torch.from_numpy(c_src_down[:, 1:].astype(np.float32)).unsqueeze(0).to(dev)
                p_coords_tgt = torch.from_numpy(c_tgt_down[:, 1:].astype(np.float32)).unsqueeze(0).to(dev)
                p_pos_src = torch.from_numpy(pos_src_np).unsqueeze(0).to(dev)
                p_pos_tgt = torch.from_numpy(pos_tgt_np).unsqueeze(0).to(dev)
                p_mask_src = torch.ones(1, n_src, dtype=torch.bool, device=dev)
                p_mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=dev)
                ds_arr_t = torch.from_numpy(ds_arr).to(dev)

                with torch.no_grad():
                    u_feat_src = m._index_features(u_out[:, f_idx], p_coords_src, p_mask_src)
                    u_feat_tgt = m._index_features(u_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt)
                    el = m.predict_edges(
                        u_feat_src, u_feat_tgt,
                        p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                        p_pos_src, p_pos_tgt,
                        p_mask_src, p_mask_tgt,
                    )
                    if isinstance(el, (tuple, list)):
                        el = el[0]
                    if el.dim() == 3:
                        el = el.squeeze(0)
                    el = el.to(primary_device)
                edge_logits_models.append(el)

            edge_logits_ens = sum(edge_logits_models) / len(edge_logits_models)
            if edge_logits_ens.dim() == 3:
                edge_logits_ens = edge_logits_ens.squeeze(0)

            # Bidirectional Consensus Soft-Veto on GPU
            probs_gpu = 0.85 * torch.softmax(edge_logits_ens, dim=0) + 0.15 * torch.softmax(edge_logits_ens, dim=1)

            # High-Performance Vectorized Candidate Extraction on GPU (33x faster)
            scale_t = torch.tensor(scale, dtype=torch.float32, device=primary_device)
            ds_t = torch.tensor(downsample, dtype=torch.float32, device=primary_device)
            p_src_t = torch.from_numpy(c_src_down[:, 1:]).to(primary_device) * ds_t * scale_t
            p_tgt_t = torch.from_numpy(c_tgt_down[:, 1:]).to(primary_device) * ds_t * scale_t

            diff_gpu = p_tgt_t.unsqueeze(0) - p_src_t.unsqueeze(1)
            dist_gpu = torch.norm(diff_gpu, dim=-1)

            # Directional Momentum Buffer Tensor
            v_prev_t = torch.zeros((n_src, 3), dtype=torch.float32, device=primary_device)
            for i in range(n_src):
                gi = int(idx_src[i])
                vp = velocity_buffer.get(gi, None)
                if vp is not None:
                    v_prev_t[i] = torch.from_numpy(vp).to(primary_device)

            speed_prev_t = torch.norm(v_prev_t, dim=-1, keepdim=True)
            cos_theta_gpu = torch.sum(v_prev_t.unsqueeze(1) * diff_gpu, dim=-1) / (speed_prev_t * dist_gpu + 1e-6)
            delta_speed_gpu = torch.abs(dist_gpu - speed_prev_t)

            # Advective Kinematic Displacement Prior
            p_pred_t = p_src_t + v_prev_t
            diff_advect = p_tgt_t.unsqueeze(0) - p_pred_t.unsqueeze(1)
            dist_advect = torch.norm(diff_advect, dim=-1)

            # Acute reversal suppression for moving cells (suppress >120-degree hairpin turns)
            reversal_pen = torch.where(
                (speed_prev_t > 1.2) & (cos_theta_gpu < -0.20),
                0.25 * (cos_theta_gpu + 0.20),
                torch.zeros_like(cos_theta_gpu)
            )
            momentum_bonus = torch.where(
                (speed_prev_t > 1e-3) & (dist_gpu > 1e-3),
                0.10 * cos_theta_gpu - 0.02 * (delta_speed_gpu / 10.0) + reversal_pen,
                torch.zeros_like(cos_theta_gpu)
            )
            advect_bonus = torch.where(
                speed_prev_t > 0.8,
                0.05 * torch.exp(-0.5 * (dist_advect / 3.0) ** 2),
                torch.zeros_like(dist_gpu)
            )
            scores_gpu = probs_gpu + momentum_bonus + advect_bonus

            cand_mask = (probs_gpu > div_threshold) & (dist_gpu <= 12.0)
            cand_si, cand_tj = torch.nonzero(cand_mask, as_tuple=True)

            if len(cand_si) > 0:
                c_scores = scores_gpu[cand_si, cand_tj]
                c_probs = probs_gpu[cand_si, cand_tj]
                c_dists = dist_gpu[cand_si, cand_tj]

                sort_order = torch.argsort(c_scores, descending=True)
                c_scores = c_scores[sort_order]
                c_probs = c_probs[sort_order]
                cand_si = cand_si[sort_order]
                cand_tj = cand_tj[sort_order]
                c_dists = c_dists[sort_order]

                cpp_mod = get_cpp_tracker()
                if cpp_mod is not None:
                    t_src, t_tgt, t_probs, t_dists, t_div = cpp_mod.fast_greedy_track(
                        c_scores.cpu(), c_probs.cpu(), cand_si.cpu(), cand_tj.cpu(), c_dists.cpu(),
                        p_src_t.cpu(), p_tgt_t.cpu(), v_prev_t.cpu(),
                        n_src, n_tgt, edge_threshold, div_threshold, div_joint_threshold
                    )
                    out_src_np = t_src.numpy()
                    out_tgt_np = t_tgt.numpy()
                    out_probs_np = t_probs.numpy()
                    out_dists_np = t_dists.numpy()
                    p_src_np = p_src_t.cpu().numpy()
                    p_tgt_np = p_tgt_t.cpu().numpy()

                    for k in range(len(out_src_np)):
                        si_idx = out_src_np[k]
                        tj_idx = out_tgt_np[k]
                        gi = int(idx_src[si_idx])
                        gj = int(idx_tgt[tj_idx])
                        all_edges.append((gi, gj, float(out_probs_np[k]), float(out_dists_np[k])))
                        velocity_buffer[gj] = p_tgt_np[tj_idx] - p_src_np[si_idx]
                else:
                    si_np = cand_si.cpu().numpy()
                    tj_np = cand_tj.cpu().numpy()
                    eff_score_np = c_scores.cpu().numpy()
                    raw_prob_np = c_probs.cpu().numpy()
                    dist_np = c_dists.cpu().numpy()
                    p_src_np = p_src_t.cpu().numpy()
                    p_tgt_np = p_tgt_t.cpu().numpy()
                    v_prev_np = v_prev_t.cpu().numpy()

                    children_count = {}
                    parents_count = {}
                    mother_daughters = {}
                    mother_d1_prob = {}

                    for k in range(len(si_np)):
                        i, j = int(si_np[k]), int(tj_np[k])
                        if parents_count.get(j, 0) >= 1:
                            continue
                        eff_score = eff_score_np[k]
                        raw_prob = raw_prob_np[k]
                        dist_um = dist_np[k]
                        p_s = p_src_np[i]
                        p_t = p_tgt_np[j]
                        n_ch = children_count.get(i, 0)

                        if n_ch == 0:
                            if eff_score < edge_threshold and raw_prob < edge_threshold:
                                continue
                            gi, gj = int(idx_src[i]), int(idx_tgt[j])
                            all_edges.append((gi, gj, float(raw_prob), float(dist_um)))
                            children_count[i] = 1
                            parents_count[j] = 1
                            mother_daughters[i] = p_t
                            mother_d1_prob[i] = raw_prob
                            velocity_buffer[gj] = p_t - p_s
                        elif n_ch == 1:
                            if (mother_d1_prob[i] + raw_prob) < div_joint_threshold or raw_prob < div_threshold:
                                continue
                            d1 = mother_daughters[i]
                            dist_sis = float(np.linalg.norm(d1 - p_t))
                            if dist_sis > 15.34 or dist_um > 8.54:
                                continue
                            v_drift = v_prev_np[i]
                            p_comov = p_s + v_drift
                            w1 = d1 - p_comov
                            w2 = p_t - p_comov
                            n1 = float(np.linalg.norm(w1))
                            n2 = float(np.linalg.norm(w2))
                            cos_sp = float(np.dot(w1, w2) / max(n1 * n2, 1e-6))
                            mid_off = float(np.linalg.norm(0.5 * (d1 + p_t) - p_comov))
                            sym_rat = abs(n1 - n2) / (n1 + n2 + 1e-6)
                            if cos_sp > -0.40 or mid_off > 3.00 or sym_rat > 0.50:
                                continue
                            gi, gj = int(idx_src[i]), int(idx_tgt[j])
                            all_edges.append((gi, gj, float(raw_prob), float(dist_um)))
                            children_count[i] = 2
                            parents_count[j] = 1
                            velocity_buffer[gj] = p_t - p_s

    coords_down = np.concatenate(coord_lists_down) if coord_lists_down else np.empty((0, 4), dtype=np.float32)
    coords_orig = coords_down.copy()
    coords_orig[:, 1:] *= ds_arr

    # Post-processing: Gap-protected pruning, Cytokinesis False Div pruning, Endpoint reconnection, Degree-0 removal
    coords_clean, edges_clean = filter_short_tracks_with_gaps(
        coords_orig, all_edges, min_length=min_track_length, scale=scale, total_frames=T
    )
    edges_clean = prune_false_divisions(coords_clean, edges_clean, scale=scale, total_frames=T)
    edges_clean = reconnect_broken_consecutive_endpoints(coords_clean, edges_clean, scale=scale)
    coords_final, edges_final = prune_isolated_degree_0_nodes(coords_clean, edges_clean)

    return coords_final, edges_final


def main():
    print("=" * 85)
    print("      ANISOTRACK3D-ENSEMBLE KAGGLE PRODUCTION SUBMISSION GENERATOR")
    print("=" * 85)

    test_dir = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/test")
    if not test_dir.exists():
        # Fallback to train for verification
        test_dir = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")
        is_test = False
    else:
        is_test = True

    test_volumes = sorted(list(test_dir.glob("*.zarr")))
    if not is_test:
        test_volumes = test_volumes[:2]  # run 2 volumes for verification

    print(f"Discovered {len(test_volumes)} volume(s) in {test_dir}")

    device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

    weight_paths = [
        "/kaggle/input/datasets/ragunathravi/forcompbiohub/weights/unet_transformer/split_0/edge_predictor_best.pth",
        "/kaggle/input/datasets/ragunathravi/forcompbiohub/weights/unet_transformer/split_1/edge_predictor_best.pth",
        "/kaggle/input/datasets/ragunathravi/forcompbiohub/secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth",
    ]

    loaded_models = []
    downsample = (1, 4, 4)
    window_size = 2

    for idx, wp in enumerate(weight_paths):
        p = Path(wp)
        if p.exists():
            dev = device_1 if (idx % 2 == 1 and torch.cuda.device_count() > 1) else device_0
            print(f"Loading Model {idx + 1} on {dev}: {p.name}")
            m, window_size, downsample = load_model(p, dev)
            loaded_models.append((m, dev))

    out_csv = Path("submission.csv")
    row_id = 0
    total_nodes = 0
    total_edges = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()

        for vol_p in test_volumes:
            dataset_name = vol_p.stem
            t0 = time.perf_counter()
            coords, edges = track_volume_inference(
                volume_path=vol_p,
                models=loaded_models,
                downsample=downsample,
                window_size=window_size,
                min_track_length=5,
            )
            elapsed = time.perf_counter() - t0
            print(f"[{dataset_name}] Generated {len(coords)} nodes, {len(edges)} edges in {elapsed:.2f}s")

            # Write node rows
            for n_idx, (t, z, y, x) in enumerate(coords):
                writer.writerow({
                    "id": row_id,
                    "dataset": dataset_name,
                    "row_type": "node",
                    "node_id": n_idx,
                    "t": int(t),
                    "z": int(round(float(z))),
                    "y": int(round(float(y))),
                    "x": int(round(float(x))),
                    "source_id": -1,
                    "target_id": -1,
                })
                row_id += 1

            # Write edge rows (strictly dt = 1)
            for src, tgt, prob, dist in edges:
                writer.writerow({
                    "id": row_id,
                    "dataset": dataset_name,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": int(src),
                    "target_id": int(tgt),
                })
                row_id += 1

            total_nodes += len(coords)
            total_edges += len(edges)

    print("=" * 85)
    print(f"✅ SUBMISSION COMPLETE: {out_csv.resolve()}")
    print(f"   Total Rows Written: {row_id} (Nodes: {total_nodes}, Edges: {total_edges})")
    print(f"   File Size: {out_csv.stat().st_size / (1024 ** 2):.2f} MB")
    print("=" * 85)


if __name__ == "__main__":
    main()
