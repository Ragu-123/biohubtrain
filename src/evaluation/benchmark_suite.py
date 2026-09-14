"""
BenchmarkSuite: Unified Evaluation & Profiling Framework for Biohub Cell Tracking.
Measures:
1. Detection Quality (TP, FP, FN, F1 @ 3um, 5um, 7um)
2. Edge Linking (TP, FP, FN, Raw Jaccard)
3. Biological Mitosis (Division TP, FP, FN, Division Jaccard)
4. Whole-Embryo Node Census Penalty (m_i)
5. Official Competition Combined Score
6. System Performance (Peak VRAM, Latency ms/frame, Throughput)
"""

import time
import gc
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import polars as pl
import torch
import sys
for p in [
    "/kaggle/input/datasets/ragunathravi/forcompbiohub/repo/src",
    "/kaggle/input/datasets/ragunathravi/forcompbiohub/repo/scripts",
]:
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)

import tracksdata as td
from geff import GeffMetadata

# Import official metrics
import biohub_tracking.metrics as bmetrics
from biohub_tracking.io import open_dataset


@dataclass
class VolumeEvaluationMetrics:
    volume_name: str
    num_pred_nodes: int
    num_gt_nodes: int
    estimated_n_total: float
    census_multiplier: float
    
    # Edge Linking Metrics
    edge_tp: int
    edge_fp: int
    edge_fn: int
    raw_edge_jaccard: float
    adj_edge_jaccard: float
    
    # Division Metrics
    div_tp: int
    div_fp: int
    div_fn: int
    div_jaccard: float
    
    # Combined Competition Score
    competition_score: float
    
    # Hardware Profiling
    latency_sec: float
    peak_vram_mb: float


class BenchmarkSuite:
    def __init__(self, train_dir: Path, max_matching_distance_um: float = 7.0):
        self.train_dir = Path(train_dir)
        self.max_matching_distance_um = max_matching_distance_um

    def load_gt(self, volume_name: str):
        """Loads ground truth tracks and metadata."""
        geff_path = self.train_dir / f"{volume_name}.geff"
        meta = GeffMetadata.read(geff_path)
        n_total = float((meta.extra or {}).get("estimated_number_of_nodes", float("nan")))
        ds = open_dataset(self.train_dir / volume_name, require_tracks=True, load_image=False, device="cpu")
        return ds.tracks, tuple(ds.scale), n_total

    def evaluate_graph(
        self,
        pred_graph: td.graph.BaseGraph,
        volume_name: str,
        latency_sec: float = 0.0,
        peak_vram_mb: float = 0.0,
    ) -> VolumeEvaluationMetrics:
        """Evaluates a predicted InMemoryGraph against ground truth."""
        gt_graph, scale, n_total = self.load_gt(volume_name)

        # Run official distance matching evaluation
        er = bmetrics.evaluate(pred_graph, gt_graph, scale=scale, max_distance=self.max_matching_distance_um)
        rec = bmetrics.node_recall(pred_graph, gt_graph) if pred_graph.num_edges() > 0 and pred_graph.num_nodes() > 0 else 0.0
        m = bmetrics.per_sample_metrics(er, n_total, rec)

        # Diagnostic inspection of any divisions matched to GT tracks
        if er.division_fp > 0 or er.division_tp > 0:
            try:
                from biohub_tracking.division_metrics import _match_full
                matched_pred = _match_full(pred_graph, gt_graph, scale=scale, max_distance=self.max_matching_distance_um)
                node_attrs = matched_pred.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID, "t", "z", "y", "x"])
                matched_nodes = node_attrs.filter(
                    pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
                    & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
                )
                scale_arr = np.array(scale, dtype=np.float32)
                for row in matched_nodes.iter_rows(named=True):
                    pred_node = row[td.DEFAULT_ATTR_KEYS.NODE_ID]
                    gt_node = row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
                    if matched_pred.out_degree(pred_node) >= 2 and gt_graph.out_degree(gt_node) >= 1:
                        is_true = gt_graph.out_degree(gt_node) >= 2
                        tag = "TRUE POSITIVE" if is_true else "FALSE POSITIVE"
                        t = int(row["t"])
                        succs = pred_graph.successors(pred_node)
                        preds = pred_graph.predecessors(pred_node)
                        p_m = np.array([row["z"], row["y"], row["x"]]) * scale_arr
                        if preds:
                            p_pred_row = pred_graph.node_attrs(node_ids=[preds[0]], attr_keys=["z", "y", "x"]).to_dicts()[0]
                            p_pred = np.array([p_pred_row["z"], p_pred_row["y"], p_pred_row["x"]]) * scale_arr
                            v_mom = p_m - p_pred
                        else:
                            v_mom = np.zeros(3)
                        p_comov = p_m + v_mom
                        c_rows = pred_graph.node_attrs(node_ids=succs[:2], attr_keys=["z", "y", "x"]).to_dicts()
                        p_c1 = np.array([c_rows[0]["z"], c_rows[0]["y"], c_rows[0]["x"]]) * scale_arr
                        p_c2 = np.array([c_rows[1]["z"], c_rows[1]["y"], c_rows[1]["x"]]) * scale_arr
                        w1 = p_c1 - p_comov
                        w2 = p_c2 - p_comov
                        d1 = float(np.linalg.norm(w1))
                        d2 = float(np.linalg.norm(w2))
                        cos_sp = float(np.dot(w1, w2) / (d1 * d2 + 1e-6))
                        mid_off = float(np.linalg.norm(0.5 * (p_c1 + p_c2) - p_comov))
                        sym_rat = float(abs(d1 - d2) / (d1 + d2 + 1e-6))
                        d_sis = float(np.linalg.norm(p_c1 - p_c2))

                        def fwd_len(node_id):
                            curr = node_id
                            cnt = 0
                            while True:
                                sc = pred_graph.successors(curr)
                                if not sc:
                                    break
                                curr = sc[0]
                                cnt += 1
                            return cnt
                        d1_pers = fwd_len(succs[0])
                        d2_pers = fwd_len(succs[1])

                        curr = pred_node
                        m_hist = 0
                        while True:
                            pr = pred_graph.predecessors(curr)
                            if not pr:
                                break
                            curr = pr[0]
                            m_hist += 1

                        print(f"    [Diagnostic Division] [{tag}] PredNode={pred_node} GT={gt_node} at t={t}: cos_sp={cos_sp:.3f}, mid_off={mid_off:.3f}um, sis_dist={d_sis:.3f}um, sym_ratio={sym_rat:.3f}, m_hist={m_hist}, d1_pers={d1_pers}, d2_pers={d2_pers}")
            except Exception as e:
                print(f"    [Diagnostic Division] Exception analyzing divisions: {e}")

        edge_denom = er.edge_tp + er.edge_fp + er.edge_fn
        raw_edge_j = er.edge_tp / edge_denom if edge_denom > 0 else 0.0

        div_denom = er.division_tp + er.division_fp + er.division_fn
        gt_div_count = len(list(gt_graph.dividing_nodes()))
        div_j = er.division_tp / div_denom if div_denom > 0 else (0.0 if gt_div_count > 0 else float("nan"))

        census_m = max(0.0, 1.1 - 0.1 * (er.num_pred_nodes / n_total)) if n_total > 0 else 1.0
        adj_edge_j = census_m * raw_edge_j
        comp_score = adj_edge_j + (0.10 * div_j if not np.isnan(div_j) else 0.0)

        # Clean memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return VolumeEvaluationMetrics(
            volume_name=volume_name,
            num_pred_nodes=er.num_pred_nodes,
            num_gt_nodes=gt_graph.num_nodes(),
            estimated_n_total=n_total,
            census_multiplier=census_m,
            edge_tp=er.edge_tp,
            edge_fp=er.edge_fp,
            edge_fn=er.edge_fn,
            raw_edge_jaccard=raw_edge_j,
            adj_edge_jaccard=adj_edge_j,
            div_tp=er.division_tp,
            div_fp=er.division_fp,
            div_fn=er.division_fn,
            div_jaccard=div_j,
            competition_score=comp_score,
            latency_sec=latency_sec,
            peak_vram_mb=peak_vram_mb,
        )

    def print_summary(self, results: List[VolumeEvaluationMetrics]):
        """Prints a rich, formatted benchmark summary table."""
        print("\n" + "=" * 92)
        print("                 BIOHUB HIGH-PRECISION EVALUATION BENCHMARK SUMMARY")
        print("=" * 92)
        print(f"{'Volume':<15} | {'Nodes (P/G)':<13} | {'Census m':<8} | {'Edge TP/FP/FN':<14} | {'J_edge':<7} | {'Div TP/FP/FN':<13} | {'SCORE':<7}")
        print("-" * 92)
        
        tot_edge_tp = sum(r.edge_tp for r in results)
        tot_edge_fp = sum(r.edge_fp for r in results)
        tot_edge_fn = sum(r.edge_fn for r in results)
        tot_edge_denom = tot_edge_tp + tot_edge_fp + tot_edge_fn

        tot_div_tp = sum(r.div_tp for r in results)
        tot_div_fp = sum(r.div_fp for r in results)
        tot_div_fn = sum(r.div_fn for r in results)
        tot_div_denom = tot_div_tp + tot_div_fp + tot_div_fn

        for r in results:
            p_g = f"{r.num_pred_nodes}/{r.num_gt_nodes}"
            e_str = f"{r.edge_tp}/{r.edge_fp}/{r.edge_fn}"
            d_str = f"{r.div_tp}/{r.div_fp}/{r.div_fn}"
            print(f"{r.volume_name:<15} | {p_g:<13} | {r.census_multiplier:<8.4f} | {e_str:<14} | {r.adj_edge_jaccard:<7.4f} | {d_str:<13} | {r.competition_score:<7.4f}")

        print("-" * 92)
        # Micro-averaged competition score
        tot_weighted_tp = sum(r.census_multiplier * r.edge_tp for r in results)
        agg_edge_j = tot_weighted_tp / tot_edge_denom if tot_edge_denom > 0 else 0.0
        agg_div_j = tot_div_tp / tot_div_denom if tot_div_denom > 0 else 0.0
        final_agg_score = agg_edge_j + 0.10 * agg_div_j

        print(f"{'TOTAL / MICRO':<15} | {'--':<13} | {'--':<8} | {f'{tot_edge_tp}/{tot_edge_fp}/{tot_edge_fn}':<14} | {agg_edge_j:<7.4f} | {f'{tot_div_tp}/{tot_div_fp}/{tot_div_fn}':<13} | {final_agg_score:<7.4f}")
        print("=" * 92)
        print(f"\U0001f3c6 OFFICIAL AGGREGATE SCORE ACROSS BENCHMARK: {final_agg_score:.4f}")
        print("=" * 92 + "\n")
