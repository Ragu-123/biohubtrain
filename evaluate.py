#!/usr/bin/env python
"""
AnisoTrack3D Production Evaluation & Benchmarking Pipeline.
Integrates:
1. AnisoUNet3D Anisotropic Spatial-Axial Separable Backbone
2. Custom Triton Continuous 2nd-Order Sub-Voxel Peak Refiner
3. Custom Triton Continuous Trilinear Feature Sampler
4. SparseLocalTrackTransformer Local Candidate Ball Attention (R <= 12.0 um)
5. Joint-Probability Mitosis Recovery with Cytokinesis Spindle Geometry Gating
6. Official Competition Metric Evaluation (Adjusted Edge Jaccard, Division Jaccard, Combined Score)

Usage:
    python evaluate.py --data-dir /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
                       --volumes 6bba_05db0fb1 --det-tta
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
for p in [
    "/kaggle/input/datasets/ragunathravi/forcompbiohub/repo/src",
    "/kaggle/input/datasets/ragunathravi/forcompbiohub/repo/scripts",
]:
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)

import tracksdata as td
from biohub_tracking.io import open_dataset
from src.models import AnisoTrack3D, trilinear_index_features
from src.kernels.triton_ops import refine_subvoxel_peaks_triton, HAS_TRITON
from src.evaluation.benchmark_suite import BenchmarkSuite


def pool_kernel_from_um(um: float, voxel_size: tuple[float, ...]) -> tuple[int, ...]:
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def extract_pos_features(coords: np.ndarray, image_shape: tuple[int, ...], pos_embed_dim: int = 8) -> np.ndarray:
    shape_t = np.array(image_shape, dtype=np.float32)
    norms = coords / np.maximum(shape_t, 1.0)
    freqs = (2.0 ** np.arange(pos_embed_dim // 2, dtype=np.float32)) * np.pi
    parts = []
    for ax in range(4):
        angles = norms[:, ax:ax+1] * freqs
        parts.extend([np.sin(angles), np.cos(angles)])
    return np.concatenate(parts, axis=-1)


def build_tracksdata_graph(coords: np.ndarray, edges: list[tuple[int, int, float, float]]) -> td.graph.InMemoryGraph:
    graph = td.graph.InMemoryGraph()
    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in coords
    ])

    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {
                "source_id": node_ids[src],
                "target_id": node_ids[tgt],
                "edge_prob": float(prob),
                "edge_dist": float(dist),
            }
            for src, tgt, prob, dist in edges
        ])
    return graph


@torch.no_grad()
def track_volume(
    model,
    volume_dir: Path,
    device: torch.device,
    downsample: tuple[int, int, int] = (1, 4, 4),
    window_size: int = 2,
    det_threshold: float = 0.50,
    edge_threshold: float = 0.48,
    div_threshold: float = 0.28,
    div_joint_threshold: float = 0.72,
    pool_kernel_um: float = 5.0,
    det_tta: bool = True,
    max_frames: int | None = None,
) -> tuple[td.graph.InMemoryGraph, float, float]:
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    ds = open_dataset(volume_dir, normalize=False, load_image=False, downsample=downsample)
    z_path = volume_dir if volume_dir.suffix == ".zarr" else volume_dir.with_suffix(".zarr")
    zarr_arr = zarr.open_group(str(z_path), mode="r")["0"]

    q_low = float(ds.quantiles.get("0.001", 100.0))
    q_high = float(ds.quantiles.get("0.999", 500.0))

    T = ds.image_shape[0] if max_frames is None else min(ds.image_shape[0], max_frames)
    target_shape = list(ds.image_shape[1:])
    scale = tuple(ds.scale)

    ds_arr = np.array(downsample, dtype=np.float32)
    ds_arr_t = torch.from_numpy(ds_arr).to(device)

    voxel_size_down = tuple(s * d for s, d in zip(scale, downsample))
    pool_k = pool_kernel_from_um(pool_kernel_um, voxel_size_down)
    pad = tuple(k // 2 for k in pool_k)

    seen_frames = set()
    seen_pairs = set()
    coord_lists_down = []
    coord_offset = {}
    global_node_count = 0
    all_edges = []

    stride = max(window_size - 1, 1)
    window_starts = list(range(0, T - window_size + 1, stride))
    if not window_starts or window_starts[-1] + window_size < T:
        last = max(T - window_size, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    for ws in window_starts:
        frame_indices = list(range(ws, ws + window_size))
        imgs = []
        for t in frame_indices:
            dz, dy, dx = downsample
            raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)
            f_t = torch.from_numpy(raw)
            if list(f_t.shape) != target_shape:
                f_t = F.interpolate(f_t[None, None], size=target_shape, mode="trilinear", align_corners=False)[0, 0]
            imgs.append(f_t)

        imgs = torch.stack(imgs)
        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0).to(device)

        with torch.no_grad():
            unet_out, det_logits = model.encode(imgs)

            if det_tta:
                tta_flips = [(-1,), (-2,), (-2, -1)]
                for dims in tta_flips:
                    imgs_flip = imgs.flip(dims)
                    _, det_flip = model.encode(imgs_flip)
                    for f in range(window_size):
                        det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                    del imgs_flip, det_flip
                for f in range(window_size):
                    det_logits[f] = det_logits[f] / 4

        # Cell Detection + Custom Triton Sub-Voxel Peak Refinement
        for f_idx, t in enumerate(frame_indices):
            if t not in seen_frames:
                log_t = det_logits[f_idx]
                pooled = F.max_pool3d(log_t, pool_k, stride=1, padding=pad)
                is_peak = (log_t == pooled) & (torch.sigmoid(log_t) > det_threshold)
                peak_idx = torch.nonzero(is_peak[0, 0])

                if len(peak_idx) > 0:
                    if HAS_TRITON and device.type == "cuda":
                        peaks_refined = refine_subvoxel_peaks_triton(log_t[0, 0], peak_idx)
                    else:
                        peaks_refined = peak_idx.float()

                    t_col = np.full((len(peaks_refined), 1), t, dtype=np.float32)
                    arr_down = np.concatenate([t_col, peaks_refined.cpu().numpy()], axis=1)
                else:
                    arr_down = np.empty((0, 4), dtype=np.float32)

                coord_offset[t] = (global_node_count, global_node_count + len(arr_down))
                global_node_count += len(arr_down)
                coord_lists_down.append(arr_down)
                seen_frames.add(t)

        coords_down_so_far = np.concatenate(coord_lists_down) if coord_lists_down else np.empty((0, 4), dtype=np.float32)

        # Edge Association with Mitosis Gating
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

            p_coords_src = torch.from_numpy(c_src_down[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            p_coords_tgt = torch.from_numpy(c_tgt_down[:, 1:].astype(np.float32)).unsqueeze(0).to(device)

            window_shape = (window_size,) + ds.image_shape[1:]
            c_src_rel = c_src_down.copy()
            c_src_rel[:, 0] = f_idx
            c_tgt_rel = c_tgt_down.copy()
            c_tgt_rel[:, 0] = f_idx + 1
            p_pos_src = torch.from_numpy(extract_pos_features(c_src_rel, window_shape)).unsqueeze(0).to(device)
            p_pos_tgt = torch.from_numpy(extract_pos_features(c_tgt_rel, window_shape)).unsqueeze(0).to(device)
            p_mask_src = torch.ones(1, n_src, dtype=torch.bool, device=device)
            p_mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=device)

            with torch.no_grad():
                unet_feat_src = model._index_features(unet_out[:, f_idx], p_coords_src, p_mask_src)
                unet_feat_tgt = model._index_features(unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt)
                edge_logits_pair = model.predict_edges(
                    unet_feat_src, unet_feat_tgt,
                    p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                    p_pos_src, p_pos_tgt,
                    p_mask_src, p_mask_tgt,
                )

            raw = edge_logits_pair[0]
            probs = torch.softmax(raw, dim=0).cpu().numpy()

            candidates = sorted(
                [
                    (probs[i, j], i, j)
                    for i in range(n_src)
                    for j in range(n_tgt)
                    if probs[i, j] > div_threshold
                ],
                reverse=True,
            )

            children_count = {}
            parents_count = {}
            mother_daughters = {}
            mother_d1_prob = {}

            for prob, i, j in candidates:
                n_ch = children_count.get(i, 0)
                n_pa = parents_count.get(j, 0)

                if n_pa >= 1:
                    continue

                p_src_um = c_src_down[i, 1:] * ds_arr * np.array(scale, dtype=np.float32)
                p_tgt_um = c_tgt_down[j, 1:] * ds_arr * np.array(scale, dtype=np.float32)
                dist_um = float(np.linalg.norm(p_src_um - p_tgt_um))

                # Primary edge
                if n_ch == 0:
                    if prob < edge_threshold:
                        continue
                    gi, gj = int(idx_src[i]), int(idx_tgt[j])
                    all_edges.append((gi, gj, float(prob), dist_um))
                    children_count[i] = 1
                    parents_count[j] = 1
                    mother_daughters[i] = p_tgt_um
                    mother_d1_prob[i] = prob

                # Secondary edge (division)
                elif n_ch == 1:
                    if (mother_d1_prob[i] + prob) < div_joint_threshold or prob < div_threshold:
                        continue

                    d1_um = mother_daughters[i]
                    dist_sisters = float(np.linalg.norm(d1_um - p_tgt_um))
                    dist_m_d1 = float(np.linalg.norm(p_src_um - d1_um))
                    symmetry_ratio = abs(dist_m_d1 - dist_um) / (dist_m_d1 + dist_um + 1e-6)

                    # Biological Cytokinesis Constraints
                    if dist_um > 8.50 or dist_sisters > 15.30 or symmetry_ratio > 0.40:
                        continue

                    gi, gj = int(idx_src[i]), int(idx_tgt[j])
                    all_edges.append((gi, gj, float(prob), dist_um))
                    children_count[i] = 2
                    parents_count[j] = 1

    coords_down = np.concatenate(coord_lists_down) if coord_lists_down else np.empty((0, 4), dtype=np.float32)
    coords_orig = coords_down.copy()
    coords_orig[:, 1:] *= ds_arr

    latency_sec = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    pred_graph = build_tracksdata_graph(coords_orig, all_edges)
    return pred_graph, latency_sec, peak_vram_mb


def main():
    parser = argparse.ArgumentParser(description="AnisoTrack3D Production Benchmark Evaluation")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to train directory with .zarr and .geff")
    parser.add_argument("--weights", type=str, default="/kaggle/input/datasets/ragunathravi/forcompbiohub/weights/unet_transformer/split_0/edge_predictor_best.pth", help="Path to weights (.pth)")
    parser.add_argument("--volumes", nargs="+", default=["6bba_05db0fb1"], help="List of volume names to benchmark")
    parser.add_argument("--det-thresh", type=float, default=0.50, help="Detection threshold")
    parser.add_argument("--edge-thresh", type=float, default=0.48, help="Edge threshold")
    parser.add_argument("--det-tta", action="store_true", default=True, help="Use flip-xy TTA for detection")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for fast test")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("           ANISOTRACK3D PRODUCTION BENCHMARK EVALUATION HARNESS")
    print("=" * 85)
    print(f"  Target Volumes : {args.volumes}")
    print(f"  Device         : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Custom Triton  : {'ENABLED' if HAS_TRITON else 'DISABLED'}")
    print("=" * 85)

    from predict_unet_transformer import load_model
    model, window_size, downsample = load_model(Path(args.weights), device)

    suite = BenchmarkSuite(train_dir=Path(args.data_dir), max_matching_distance_um=7.0)
    results = []

    for vol_name in args.volumes:
        print(f"\nEvaluating volume: {vol_name}...")
        vol_path = Path(args.data_dir) / f"{vol_name}.zarr"
        pred_graph, latency_sec, peak_vram_mb = track_volume(
            model=model,
            volume_dir=vol_path,
            device=device,
            downsample=downsample,
            window_size=window_size,
            det_threshold=args.det_thresh,
            edge_threshold=args.edge_thresh,
            det_tta=args.det_tta,
            max_frames=args.max_frames,
        )

        res = suite.evaluate_graph(
            pred_graph=pred_graph,
            volume_name=vol_name,
            latency_sec=latency_sec,
            peak_vram_mb=peak_vram_mb,
        )
        results.append(res)
        print(f"  Nodes: {res.num_pred_nodes} (GT: {res.num_gt_nodes}) | Census: {res.census_multiplier:.4f}")
        print(f"  Edges: TP={res.edge_tp}, FP={res.edge_fp}, FN={res.edge_fn} | Adj Edge Jaccard: {res.adj_edge_jaccard:.4f}")
        print(f"  Mitosis: TP={res.div_tp}, FP={res.div_fp}, FN={res.div_fn} | Div Jaccard: {res.div_jaccard:.4f}")
        print(f"  Combined Competition Score: {res.competition_score:.4f}")
        print(f"  Latency: {res.latency_sec:.2f}s | Peak VRAM: {res.peak_vram_mb:.1f} MB")

    suite.print_summary(results)


if __name__ == "__main__":
    main()
