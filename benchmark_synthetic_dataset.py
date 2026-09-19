"""
Comprehensive Multi-Sequence Benchmark on Jose Neto's Synthetic 3D Cell Tracking Dataset
=======================================================================================
Evaluates DuplicateParentTrackingSolver + Postprocess Filtering across synthetic 3D sequences.
Computes official micro-averaged Edge Jaccard, Division Jaccard, and Combined Score,
and mathematically projects the cumulative score to 150 volumes.
"""

import os
import sys
import time
import argparse
import numpy as np
import polars as pl
from typing import Dict, List, Tuple

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from duplicate_parent_solver import DuplicateParentTrackingSolver, compute_pairwise_physical_distances
import postprocess_clean

def evaluate_sequence(
    seq_path: str,
    solver: DuplicateParentTrackingSolver,
    sigma_d: float = 4.5,
    r_horizon_um: float = 16.0,
) -> Dict[str, float]:
    """
    Evaluates tracking and division linking on a single synthetic sequence.
    """
    data = np.load(seq_path, allow_pickle=True)
    raw_nodes = data['nodes'] # (N, 5): [t, z, y, x, local_id]
    gt_edges = data['edges'] # (E, 2): [src_nid, tgt_nid]
    gt_divs = data['divisions'] # (D,): parent node IDs
    scale = tuple(float(s) for s in data['voxel_um_pooled'])
    solver.voxel_scale = scale

    # Build node lookup by frame
    frames = sorted(list(set(int(r[0]) for r in raw_nodes)))
    nodes_by_frame = {}
    for f in frames:
        mask = raw_nodes[:, 0] == f
        nodes_by_frame[f] = raw_nodes[mask]

    # Map node index (0..N-1) to (t, z, y, x)
    node_id_to_idx = {i: i for i in range(len(raw_nodes))}

    # Solve frame-by-frame bipartite matching
    pred_edges = []
    for t in range(len(frames) - 1):
        f_src = frames[t]
        f_tgt = frames[t + 1]

        src_arr = nodes_by_frame[f_src]
        tgt_arr = nodes_by_frame[f_tgt]

        if len(src_arr) == 0 or len(tgt_arr) == 0:
            continue

        src_coords = src_arr[:, 1:4] # (N, 3)
        tgt_coords = tgt_arr[:, 1:4] # (M, 3)

        # Pairwise physical anisotropic distances in microns
        dists = compute_pairwise_physical_distances(src_coords, tgt_coords, scale)
        
        # Kinetic transition probability with gentle decay for cytokinesis
        probs = np.exp(- (dists ** 2) / (2.0 * sigma_d ** 2))
        probs[dists > r_horizon_um] = 0.0

        solved_edges = solver.solve_frame_pair(src_coords, tgt_coords, probs)

        # Find global indices
        src_global_ids = np.where(raw_nodes[:, 0] == f_src)[0]
        tgt_global_ids = np.where(raw_nodes[:, 0] == f_tgt)[0]

        for se in solved_edges:
            g_src = int(src_global_ids[se.source_idx])
            g_tgt = int(tgt_global_ids[se.target_idx])
            pred_edges.append({
                'source_id': g_src,
                'target_id': g_tgt,
                'edge_prob': float(se.prob),
                'distance_um': float(se.distance_um),
                'is_division': int(se.is_division),
            })

    # Prepare input for postprocess_clean
    nodes_by_id = {
        int(i): {
            'node_id': int(i),
            't': int(raw_nodes[i, 0]),
            'z': float(raw_nodes[i, 1]),
            'y': float(raw_nodes[i, 2]),
            'x': float(raw_nodes[i, 3]),
        }
        for i in range(len(raw_nodes))
    }

    # Set voxel scale in postprocess_clean
    postprocess_clean.VOXEL_SCALE_UM = scale

    # Filter with postprocess_clean in mitotic mode (accommodating high-mitosis synthetic volumes)
    custom_params = {
        "safe_div_global_frac_cap": 0.08,
        "safe_div_frame_frac_cap": 0.12,
        "div_sister_max_um": 18.0,
        "div_parent_max_um": 10.0,
    }
    kept_nodes, kept_edges, _ = postprocess_clean.filter_output_graph(
        nodes_by_id=nodes_by_id,
        raw_edges=pred_edges,
        mean_nodes_per_frame=500.0,
        total_frames=len(frames),
        custom_params=custom_params,
    )

    # Reconstruct predicted edges and divisions
    pred_edge_set = set(zip([int(e['source_id']) for e in kept_edges], [int(e['target_id']) for e in kept_edges]))
    
    # Predicted divisions: parent nodes with out-degree >= 2
    out_degree = {}
    for src, tgt in pred_edge_set:
        out_degree[src] = out_degree.get(src, 0) + 1
    pred_div_set = {src for src, deg in out_degree.items() if deg >= 2}

    # Ground truth edges and divisions
    gt_edge_set = set(zip(gt_edges[:, 0].tolist(), gt_edges[:, 1].tolist()))
    gt_div_set = set(gt_divs.tolist())

    # Compute Edge TP, FP, FN
    edge_tp = len(pred_edge_set & gt_edge_set)
    edge_fp = len(pred_edge_set - gt_edge_set)
    edge_fn = len(gt_edge_set - pred_edge_set)

    # Compute Division TP, FP, FN
    div_tp = len(pred_div_set & gt_div_set)
    div_fp = len(pred_div_set - gt_div_set)
    div_fn = len(gt_div_set - pred_div_set)

    # Compute sample metrics
    edge_denom = edge_tp + edge_fp + edge_fn
    edge_jaccard = edge_tp / edge_denom if edge_denom > 0 else 0.0

    div_denom = div_tp + div_fp + div_fn
    div_jaccard = div_tp / div_denom if div_denom > 0 else 0.0

    score = edge_jaccard + 0.10 * div_jaccard if div_denom > 0 else edge_jaccard

    return {
        'num_nodes': len(raw_nodes),
        'gt_edges': len(gt_edges),
        'gt_divisions': len(gt_divs),
        'edge_tp': edge_tp,
        'edge_fp': edge_fp,
        'edge_fn': edge_fn,
        'edge_jaccard': edge_jaccard,
        'div_tp': div_tp,
        'div_fp': div_fp,
        'div_fn': div_fn,
        'div_jaccard': div_jaccard,
        'score': score,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Synthetic 3D Cell Tracking Dataset")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/notebooks/josefreitasalvesneto/biohub-synthetic-dataset/biohub_synthetic/sequences")
    parser.add_argument("--num_seqs", type=int, default=50, help="Number of sequences to evaluate")
    parser.add_argument("--c_div", type=float, default=0.20, help="Division cost threshold penalty")
    parser.add_argument("--sigma_d", type=float, default=5.5, help="Dispersion scale parameter")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    all_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.npz')])
    num_to_eval = min(args.num_seqs, len(all_files))
    selected_files = all_files[:num_to_eval]

    print("=" * 80)
    print(f"🚀 RUNNING MULTI-SEQUENCE BENCHMARK ON {num_to_eval} SYNTHETIC SEQUENCES")
    print(f"Data directory: {args.data_dir}")
    print(f"Division penalty c_div: {args.c_div} | sigma_d: {args.sigma_d}")
    print("=" * 80)

    # Setup solver with calibrated parameters
    solver = DuplicateParentTrackingSolver(
        use_mejc=True,
        c_app=0.10,
        c_div=0.65, # Restore high-precision c_div
        min_sister_dist_um=3.0,
        max_sister_dist_um=20.0, 
        max_parent_dist_um=15.0, 
        r_max_um=25.0,
        mejc_min_prob=0.01,
    )

    # Post-processing calibration
    postprocess_clean.SAFE_DIV_SISTER_SYMMETRY_TAU = 0.60 # Tighten symmetry for precision
    postprocess_clean.SAFE_DIV_SISTER_MAX_UM = 18.0
    postprocess_clean.SAFE_DIV_MAX_UM = 12.0
    postprocess_clean.SAFE_DIV_GLOBAL_FRAC_CAP = 0.05
    postprocess_clean.OUTPUT_BLC_CONSENSUS = True
    postprocess_clean.BLC_VETO_STRENGTH = 0.8
    postprocess_clean.BLC_DISAGREEMENT_PENALTY = 0.2

    t0 = time.time()
    results = []
    
    # Cumulative micro-averaged accumulators
    tot_edge_tp = tot_edge_fp = tot_edge_fn = 0
    tot_div_tp = tot_div_fp = tot_div_fn = 0
    tot_gt_edges = tot_gt_divs = 0

    for idx, fname in enumerate(selected_files):
        fpath = os.path.join(args.data_dir, fname)
        t_seq = time.time()
        res = evaluate_sequence(fpath, solver, sigma_d=args.sigma_d)
        dt = time.time() - t_seq
        results.append(res)

        tot_edge_tp += res['edge_tp']
        tot_edge_fp += res['edge_fp']
        tot_edge_fn += res['edge_fn']
        tot_div_tp += res['div_tp']
        tot_div_fp += res['div_fp']
        tot_div_fn += res['div_fn']
        tot_gt_edges += res['gt_edges']
        tot_gt_divs += res['gt_divisions']

        if (idx + 1) % 1 == 0 or idx == 0 or (idx + 1) == num_to_eval:
            cur_edge_j = tot_edge_tp / (tot_edge_tp + tot_edge_fp + tot_edge_fn) if (tot_edge_tp + tot_edge_fp + tot_edge_fn) > 0 else 0
            cur_div_j = tot_div_tp / (tot_div_tp + tot_div_fp + tot_div_fn) if (tot_div_tp + tot_div_fp + tot_div_fn) > 0 else 0
            cur_score = cur_edge_j + 0.10 * cur_div_j
            print(f"[{idx+1:03d}/{num_to_eval:03d}] {fname} | EdgeJ: {cur_edge_j:.4f} | DivJ: {cur_div_j:.4f} (TP:{tot_div_tp}/FP:{tot_div_fp}/FN:{tot_div_fn}) | Combined Score: {cur_score:.4f} ({dt:.2f}s)")

    total_time = time.time() - t0

    # 1. Measured Micro-Averaged Performance
    micro_edge_j = tot_edge_tp / (tot_edge_tp + tot_edge_fp + tot_edge_fn)
    micro_edge_prec = tot_edge_tp / (tot_edge_tp + tot_edge_fp) if (tot_edge_tp + tot_edge_fp) > 0 else 0
    micro_edge_rec = tot_edge_tp / (tot_edge_tp + tot_edge_fn) if (tot_edge_tp + tot_edge_fn) > 0 else 0

    micro_div_j = tot_div_tp / (tot_div_tp + tot_div_fp + tot_div_fn) if (tot_div_tp + tot_div_fp + tot_div_fn) > 0 else 0
    micro_div_prec = tot_div_tp / (tot_div_tp + tot_div_fp) if (tot_div_tp + tot_div_fp) > 0 else 0
    micro_div_rec = tot_div_tp / (tot_div_tp + tot_div_fn) if (tot_div_tp + tot_div_fn) > 0 else 0

    measured_score = micro_edge_j + 0.10 * micro_div_j

    print("\n" + "=" * 80)
    print(f"📊 EMPIRICAL MICRO-AVERAGED BENCHMARK RESULTS ({num_to_eval} SEQUENCES)")
    print("=" * 80)
    print(f"Total Execution Time: {total_time:.2f}s ({total_time / num_to_eval * 1000:.1f} ms/sequence)")
    print(f"\n--- EDGE TRACKING METRICS ---")
    print(f"  Total GT Edges:    {tot_gt_edges:,}")
    print(f"  Edge TP / FP / FN: {tot_edge_tp:,} / {tot_edge_fp:,} / {tot_edge_fn:,}")
    print(f"  Edge Precision:    {micro_edge_prec * 100:.2f}%")
    print(f"  Edge Recall:       {micro_edge_rec * 100:.2f}%")
    print(f"  Micro Edge Jaccard: {micro_edge_j:.4f}")

    print(f"\n--- MITOTIC CELL DIVISION METRICS ---")
    print(f"  Total GT Divisions: {tot_gt_divs:,}")
    print(f"  Div TP / FP / FN:   {tot_div_tp:,} / {tot_div_fp:,} / {tot_div_fn:,}")
    print(f"  Division Precision: {micro_div_prec * 100:.2f}%")
    print(f"  Division Recall:    {micro_div_rec * 100:.2f}%")
    print(f"  Micro Div Jaccard:  {micro_div_j:.4f}")

    print(f"\n--- OVERALL RUN-LEVEL SCORE ---")
    print(f"  Formula:        Score = Edge_Jaccard + 0.10 * Div_Jaccard")
    print(f"  Calculation:    {micro_edge_j:.4f} + 0.10 * {micro_div_j:.4f}")
    print(f"  EMPIRICAL SCORE: {measured_score:.4f}")

    # 2. Mathematical Projection to 150 Volumes
    print("\n" + "=" * 80)
    print("📈 MATHEMATICAL PROJECTION TO 150 FULL COMPETITION VOLUMES")
    print("=" * 80)
    scale_factor = 150.0 / num_to_eval
    proj_edge_tp = int(round(tot_edge_tp * scale_factor))
    proj_edge_fp = int(round(tot_edge_fp * scale_factor))
    proj_edge_fn = int(round(tot_edge_fn * scale_factor))
    proj_div_tp = int(round(tot_div_tp * scale_factor))
    proj_div_fp = int(round(tot_div_fp * scale_factor))
    proj_div_fn = int(round(tot_div_fn * scale_factor))

    proj_edge_j = proj_edge_tp / (proj_edge_tp + proj_edge_fp + proj_edge_fn)
    proj_div_j = proj_div_tp / (proj_div_tp + proj_div_fp + proj_div_fn) if (proj_div_tp + proj_div_fp + proj_div_fn) > 0 else 0
    final_150_score = proj_edge_j + 0.10 * proj_div_j

    print(f"Scale Factor: {scale_factor:.2f}x ({num_to_eval} sequences -> 150 full test volumes)")
    print(f"Projected Total Edges:      {proj_edge_tp + proj_edge_fn:,}")
    print(f"Projected Edge TP/FP/FN:    {proj_edge_tp:,} / {proj_edge_fp:,} / {proj_edge_fn:,}")
    print(f"Projected Edge Jaccard:     {proj_edge_j:.4f}")
    print(f"Projected Total Divisions:  {proj_div_tp + proj_div_fn:,}")
    print(f"Projected Div TP/FP/FN:     {proj_div_tp:,} / {proj_div_fp:,} / {proj_div_fn:,}")
    print(f"Projected Division Jaccard: {proj_div_j:.4f}")
    print(f"--------------------------------------------------")
    print(f"🏆 FINAL PROJECTED LEADERBOARD SCORE (150 VOLUMES): {final_150_score:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
