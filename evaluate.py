#!/usr/bin/env python
"""
AnisoTrack3D Evaluation & Benchmarking Pipeline.
Runs end-to-end 3D cell tracking on validation/test movies using:
1. AnisoUNet3D anisotropic backbone
2. Custom Triton Continuous Sub-Voxel Peak Refiner
3. Custom Triton Trilinear Feature Sampler
4. SparseLocalTrackTransformer local candidate ball cross-attention
5. BenchmarkSuite official competition metric evaluation (Adjusted Edge Jaccard, Division Jaccard, Combined Score)

Usage:
    python evaluate.py --data-dir /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
                       --checkpoint checkpoints/anisotrack3d_epoch_1.pth \
                       --volumes 6bba_05db0fb1
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

# Add support paths
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


def build_tracksdata_graph(coords: np.ndarray, edges: list[tuple[int, int, float, float]]) -> td.graph.InMemoryGraph:
    """Builds InMemoryGraph with sub-voxel floating point coordinates."""
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
def track_volume_anisotrack(
    model: AnisoTrack3D,
    volume_dir: Path,
    device: torch.device,
    downsample: tuple[int, int, int] = (1, 4, 4),
    det_threshold: float = 0.40,
    edge_threshold: float = 0.35,
    r_max_um: float = 12.0,
    pool_kernel_um: float = 4.5,
    max_frames: int | None = None,
) -> tuple[td.graph.InMemoryGraph, float, float]:
    """
    Executes full continuous 3D tracking inference on a single volume.
    Returns: (pred_graph, latency_sec, peak_vram_mb)
    """
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    ds = open_dataset(volume_dir, normalize=False, load_image=False, downsample=downsample)
    z_path = volume_dir if volume_dir.suffix == ".zarr" else volume_dir.with_suffix(".zarr")
    zarr_arr = zarr.open_group(str(z_path), mode="r")["0"]

    q_low = float(ds.quantiles.get("0.001", 100.0))
    q_high = float(ds.quantiles.get("0.999", 500.0))

    T = ds.image_shape[0] if max_frames is None else min(ds.image_shape[0], max_frames)
    scale = tuple(ds.scale)

    ds_tensor = torch.tensor(downsample, device=device, dtype=torch.float32)
    scale_tensor = torch.tensor(scale, device=device, dtype=torch.float32)

    voxel_size_down = tuple(s * d for s, d in zip(scale, downsample))
    pool_k = pool_kernel_from_um(pool_kernel_um, voxel_size_down)
    pad = tuple(k // 2 for k in pool_k)

    frame_coords = []
    frame_feats = []
    global_coords = []
    all_edges = []
    coord_offset = {}
    current_node_idx = 0

    dz, dy, dx = downsample

    # Phase 1: Sequential Frame Feature Extraction & Sub-Voxel Peak Detection
    for t in range(T):
        raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)
        img = torch.from_numpy((raw - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        img = img.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, Z, Y, X)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            f_map, det_logit, sub_delta = model.unet._forward_single_frame(img)

        # Local maxima peak detection
        pooled = F.max_pool3d(det_logit, pool_k, stride=1, padding=pad)
        is_peak = (det_logit == pooled) & (torch.sigmoid(det_logit) > det_threshold)
        peak_idx = torch.nonzero(is_peak[0, 0]) # (N, 3) in [z, y, x]

        if peak_idx.shape[0] == 0:
            frame_coords.append(torch.empty((0, 3), device=device))
            frame_feats.append(torch.empty((0, model.unet.out_channels), device=device))
            coord_offset[t] = (current_node_idx, current_node_idx)
            continue

        # Continuous Sub-Voxel Peak Refinement via Custom Triton Kernel
        if HAS_TRITON and device.type == "cuda":
            peaks_refined = refine_subvoxel_peaks_triton(det_logit[0, 0], peak_idx)
        else:
            peaks_refined = peak_idx.float()

        # Differentiable Trilinear Feature Sampling via Custom Triton Kernel
        node_feats = trilinear_index_features(f_map[0], peaks_refined)

        # Scale continuous coordinates to full original resolution
        peaks_orig_res = peaks_refined * ds_tensor

        # Save for tracking
        frame_coords.append(peaks_orig_res)
        frame_feats.append(node_feats)

        n_nodes = len(peaks_orig_res)
        coord_offset[t] = (current_node_idx, current_node_idx + n_nodes)
        current_node_idx += n_nodes

        # Pack into global coords array [t, z, y, x]
        t_col = np.full((n_nodes, 1), t, dtype=np.float32)
        c_np = np.concatenate([t_col, peaks_orig_res.cpu().numpy()], axis=1)
        global_coords.append(c_np)

    stacked_coords = np.concatenate(global_coords, axis=0) if global_coords else np.empty((0, 4), dtype=np.float32)

    # Phase 2: Consecutive Frame Local Association & Mitosis Scoring
    for t in range(T - 1):
        s_src, e_src = coord_offset[t]
        s_tgt, e_tgt = coord_offset[t + 1]
        n_src = e_src - s_src
        n_tgt = e_tgt - s_tgt

        if n_src == 0 or n_tgt == 0:
            continue

        c_src_vox = frame_coords[t]
        c_tgt_vox = frame_coords[t + 1]
        f_src = frame_feats[t]
        f_tgt = frame_feats[t + 1]

        # Physical coordinates in microns
        c_src_um = c_src_vox * scale_tensor
        c_tgt_um = c_tgt_vox * scale_tensor

        # Local Candidate Ball Cross-Attention
        with torch.amp.autocast('cuda', dtype=torch.float16):
            edge_logits, cand_mask = model.predict_edges(f_src, c_src_um, f_tgt, c_tgt_um)

        # Column-wise Softmax over candidate sources
        probs = torch.softmax(edge_logits.float(), dim=0).cpu().numpy()

        # Extract sorted candidate associations
        cand_list = [
            (probs[i, j], i, j)
            for i in range(n_src)
            for j in range(n_tgt)
            if probs[i, j] > edge_threshold and cand_mask[i, j].item()
        ]
        cand_list.sort(key=lambda x: x[0], reverse=True)

        # Greedy Bipartite Matching with Division Capacity = 2
        children_count = {}
        parents_count = {}

        for prob, i, j in cand_list:
            n_ch = children_count.get(i, 0)
            n_pa = parents_count.get(j, 0)

            # Cell biology constraints:
            # Maximum 2 daughters per mother cell (mitosis)
            # Maximum 1 parent per daughter cell
            if n_ch >= 2 or n_pa >= 1:
                continue

            gi = s_src + i
            gj = s_tgt + j
            p_src_um = stacked_coords[gi, 1:] * np.array(scale, dtype=np.float32)
            p_tgt_um = stacked_coords[gj, 1:] * np.array(scale, dtype=np.float32)
            dist_um = float(np.linalg.norm(p_src_um - p_tgt_um))

            all_edges.append((gi, gj, float(prob), dist_um))
            children_count[i] = n_ch + 1
            parents_count[j] = n_pa + 1

    latency_sec = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    pred_graph = build_tracksdata_graph(stacked_coords, all_edges)
    return pred_graph, latency_sec, peak_vram_mb


def main():
    parser = argparse.ArgumentParser(description="AnisoTrack3D Benchmark Evaluation")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to train directory with .zarr and .geff")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--volumes", nargs="+", default=["6bba_05db0fb1"], help="List of volume names to benchmark")
    parser.add_argument("--det-thresh", type=float, default=0.40, help="Detection threshold")
    parser.add_argument("--edge-thresh", type=float, default=0.35, help="Edge threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for fast benchmark")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 82)
    print("               ANISOTRACK3D RIGOROUS BENCHMARK EVALUATION")
    print("=" * 82)
    print(f"  Target Volumes : {args.volumes}")
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Device         : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Custom Triton  : {'ENABLED' if HAS_TRITON else 'DISABLED (fallback)'}")
    print("=" * 82)

    # Initialize model
    model = AnisoTrack3D(
        unet_out_channels=32,
        unet_layers=[32, 64, 128],
        transformer_d_model=64,
        depthwise=True,
    ).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint weights from {args.checkpoint}...")
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.unet.load_state_dict(state["unet_state_dict"])
        model.transformer.load_state_dict(state["transformer_state_dict"])
        print(f"Successfully restored model from epoch {state.get('epoch', 'N/A')}")
    else:
        print("No checkpoint provided or found; evaluating initialized model architecture.")

    model.eval()

    suite = BenchmarkSuite(train_dir=Path(args.data_dir), max_matching_distance_um=7.0)
    results = []

    for vol_name in args.volumes:
        print(f"\nEvaluating volume: {vol_name}...")
        vol_path = Path(args.data_dir) / f"{vol_name}.zarr"
        pred_graph, latency_sec, peak_vram_mb = track_volume_anisotrack(
            model=model,
            volume_dir=vol_path,
            device=device,
            det_threshold=args.det_thresh,
            edge_threshold=args.edge_thresh,
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
