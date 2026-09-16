"""
Production-grade SOTA Post-Processing Module for Biohub Cell Tracking
Contains:
1. Candidate edge filtering & single-parent resolution
2. Density-adaptive single-frame gap closing (optical dropout recovery)
3. Directionally-consistent strict gap-2 recovery
4. Biological safe-division gating (parent-daughter <= 9.0um, sister <= 14.0um, sister-symmetry tau <= 0.60, divergence >= 2.25um)
5. Isolated node pruning & short-track filtering (length < 6, with division rescue)
6. Directional linefit track interior smoothing
"""

import math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)

# Hyperparameters
OUTPUT_EDGE_MAX_UM = 25.0  # Synchronized with M1 Smooth Energy candidate search radius (recovers transitions up to 25.0 um)
OUTPUT_ENFORCE_NEXT_FRAME = True
OUTPUT_SINGLE_PARENT_REPAIR = True
OUTPUT_PRUNE_ISOLATED = True

OUTPUT_GAP_CLOSE = True
GAP_CLOSE_MAX_GAP = 1
GAP_CLOSE_UM = 5.0
GAP_DENSITY_ADAPTIVE = True
GAP_DENSITY_REFERENCE_UM = 6.5
GAP_DENSITY_GAIN = 0.040
GAP_DENSITY_MAX_STEP_DELTA_UM = 0.125
GAP_DENSITY_NEIGHBORS = 3
GAP_CLOSE_REUSE_EXISTING = True
GAP_CLOSE_REUSE_UM = 3.0
GAP_CLOSE_INSERT_SYNTHETIC = False  # Dual-graph pattern: internal protection only, 0 synthetic nodes
GAP_CLOSE_MAX_ADDED_FRAC = 0.08
GAP_CLOSE_MAX_ADDED_ABS = 1900

OUTPUT_GAP2_RECOVERY = True
GAP2_MAX_TOTAL_UM = 12.0
GAP2_MAX_STEP_UM = 4.5
GAP2_MAX_LINKS_FRAC = 0.0026
GAP2_MAX_LINKS_ABS = 200
GAP2_REQUIRE_CONTEXT = True
GAP2_FRAME_FRAC_CAP = 0.0040
GAP2_INSERT_SYNTHETIC = False  # Dual-graph pattern: internal protection only

OUTPUT_SAFE_DIVISIONS = True
SAFE_DIV_MAX_UM = 9.0
SAFE_DIV_SISTER_MAX_UM = 14.0
SAFE_DIV_SISTER_SYMMETRY_TAU = 0.60
SAFE_DIV_DIVERGE_UM = 2.25
SAFE_DIV_EXISTING_CHILD_MAX_UM = 10.0
SAFE_DIV_FRAME_FRAC_CAP = 0.0076
SAFE_DIV_GLOBAL_FRAC_CAP = 0.00375
SAFE_DIV_REQUIRE_MUTUAL_NN = True
SAFE_DIV_REQUIRE_DIVERGENCE = True

OUTPUT_DIVISION_GEOMETRY_FILTER = True
DIV_PARENT_MAX_UM = 10.0
DIV_SISTER_MAX_UM = 14.0
DIV_DROP_TO_SINGLE_IF_BAD = True

OUTPUT_FILTER_SHORT_TRACKS = True
OUTPUT_MIN_TRACK_LEN = 6
OUTPUT_KEEP_DIVISION_COMPONENTS = True
ADAPTIVE_SHORT_TRACK_RESCUE = True
SHORT_TRACK_RESCUE_MIN_LEN = 4
SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB = 0.88
SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM = 3.0
SHORT_TRACK_RESCUE_MAX_NODES_FRAC = 0.012
SHORT_TRACK_RESCUE_MAX_NODES_ABS = 120
SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC = 0.10

OUTPUT_LINEFIT_SMOOTH = True
OUTPUT_LINEFIT_WEIGHT = 0.74
OUTPUT_LINEFIT_WINDOW = 2


def _position_um(node: dict[str, object]) -> np.ndarray:
    return np.array(
        [float(node["z"]) * VOXEL_SCALE_UM[0], float(node["y"]) * VOXEL_SCALE_UM[1], float(node["x"]) * VOXEL_SCALE_UM[2]],
        dtype=np.float64,
    )


def point_distance_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dz = (a[0] - b[0]) * VOXEL_SCALE_UM[0]
    dy = (a[1] - b[1]) * VOXEL_SCALE_UM[1]
    dx = (a[2] - b[2]) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def edge_distance_um(source: dict[str, object], target: dict[str, object]) -> float:
    dz = (float(source["z"]) - float(target["z"])) * VOXEL_SCALE_UM[0]
    dy = (float(source["y"]) - float(target["y"])) * VOXEL_SCALE_UM[1]
    dx = (float(source["x"]) - float(target["x"])) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def node_point(node: dict[str, object]) -> tuple[float, float, float]:
    return (float(node["z"]), float(node["y"]), float(node["x"]))


def edge_sort_key(edge: dict[str, object]) -> tuple[float, float]:
    prob = edge.get("edge_prob")
    prob_value = float(prob) if prob is not None else 0.0
    return prob_value, -float(edge.get("distance_um", 0.0))


def _next_node_id(nodes_by_id: dict[int, dict[str, object]]) -> int:
    return max(nodes_by_id) + 1 if nodes_by_id else 1


def _single_successor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_source: dict[int, list[int]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(int(edge["target_id"]))
    return {source: targets[0] for source, targets in by_source.items() if len(targets) == 1}


def _single_predecessor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_target: dict[int, list[int]] = {}
    for edge in edges:
        by_target.setdefault(int(edge["target_id"]), []).append(int(edge["source_id"]))
    return {target: sources[0] for target, sources in by_target.items() if len(sources) == 1}


def close_single_frame_gaps(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], list[tuple[int, int]]]:
    if not OUTPUT_GAP_CLOSE or GAP_CLOSE_MAX_GAP < 1 or not edges:
        return nodes_by_id, edges, []

    internal_gap_pairs: list[tuple[int, int]] = []

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    incident = outgoing | incoming

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    isolated_by_t: dict[int, list[int]] = {}
    all_ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        all_ids_by_t.setdefault(t, []).append(node_id)
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)
        if node_id not in incident:
            isolated_by_t.setdefault(t, []).append(node_id)

    max_synthetic = min(
        GAP_CLOSE_MAX_ADDED_ABS,
        max(1, int(round(len(nodes_by_id) * GAP_CLOSE_MAX_ADDED_FRAC))) if GAP_CLOSE_MAX_ADDED_FRAC > 0 else 0,
    )
    next_id = _next_node_id(nodes_by_id)
    used_starts: set[int] = set()
    used_isolated: set[int] = set()
    synthetic_added = 0
    new_edges: list[dict[str, object]] = []

    density_cache: dict[int, dict[int, float]] = {}

    def frame_local_spacing(t: int) -> dict[int, float]:
        cached = density_cache.get(t)
        if cached is not None:
            return cached
        frame_ids = all_ids_by_t.get(t, [])
        if len(frame_ids) <= 1:
            result = {nid: GAP_DENSITY_REFERENCE_UM for nid in frame_ids}
            density_cache[t] = result
            return result
        positions = np.stack([_position_um(nodes_by_id[nid]) for nid in frame_ids])
        tree = cKDTree(positions)
        query_k = min(len(frame_ids), max(2, GAP_DENSITY_NEIGHBORS + 1))
        distances, _ = tree.query(positions, k=query_k)
        if distances.ndim == 1:
            distances = distances[:, None]
        result = {}
        for idx, nid in enumerate(frame_ids):
            neighbour_distances = distances[idx, 1:]
            neighbour_distances = neighbour_distances[np.isfinite(neighbour_distances)]
            spacing = float(np.median(neighbour_distances)) if neighbour_distances.size else GAP_DENSITY_REFERENCE_UM
            result[nid] = spacing
        density_cache[t] = result
        return result

    effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)
    for gap in range(1, effective_gap_max + 1):
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = [sid for sid in starts_by_t.get(t + gap + 1, []) if sid not in used_starts]
            if not end_ids or not start_ids:
                continue

            end_points = [node_point(nodes_by_id[eid]) for eid in end_ids]
            start_points = [node_point(nodes_by_id[sid]) for sid in start_ids]
            threshold_um = GAP_CLOSE_UM * (gap + 1)
            d = np.zeros((len(end_ids), len(start_ids)), dtype=np.float64)
            adaptive_threshold = np.full_like(d, threshold_um)

            source_spacing = frame_local_spacing(t)
            target_spacing = frame_local_spacing(t + gap + 1)

            for i, ep in enumerate(end_points):
                for j, sp in enumerate(start_points):
                    d[i, j] = point_distance_um(ep, sp)
                    if GAP_DENSITY_ADAPTIVE:
                        local_spacing = 0.5 * (source_spacing.get(end_ids[i], GAP_DENSITY_REFERENCE_UM) + target_spacing.get(start_ids[j], GAP_DENSITY_REFERENCE_UM))
                        step_delta = float(np.clip(GAP_DENSITY_GAIN * (local_spacing - GAP_DENSITY_REFERENCE_UM), -GAP_DENSITY_MAX_STEP_DELTA_UM, GAP_DENSITY_MAX_STEP_DELTA_UM))
                        adaptive_threshold[i, j] = threshold_um + step_delta * (gap + 1)

            adaptive_allowed = d <= adaptive_threshold
            if not np.isfinite(d).any():
                continue

            max_threshold = float(np.max(adaptive_threshold))
            big = max_threshold * 1000.0 + 1.0
            cost = np.where(adaptive_allowed, d, big)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if not adaptive_allowed[r, c]:
                    continue
                source_id = end_ids[int(r)]
                target_id = start_ids[int(c)]
                if source_id in outgoing or target_id in used_starts:
                    continue

                source = nodes_by_id[source_id]
                target = nodes_by_id[target_id]
                mid_t = int(source["t"]) + gap
                mid_point = (
                    (float(source["z"]) + float(target["z"])) / 2.0,
                    (float(source["y"]) + float(target["y"])) / 2.0,
                    (float(source["x"]) + float(target["x"])) / 2.0,
                )

                middle_id = None
                middle_reused = False
                if GAP_CLOSE_REUSE_EXISTING:
                    candidates = [nid for nid in isolated_by_t.get(mid_t, []) if nid not in used_isolated]
                    if candidates:
                        distances = [point_distance_um(node_point(nodes_by_id[nid]), mid_point) for nid in candidates]
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= GAP_CLOSE_REUSE_UM:
                            middle_id = candidates[best_idx]
                            middle_reused = True

                if middle_id is None:
                    if not GAP_CLOSE_INSERT_SYNTHETIC:
                        # ASTRA DUAL-GRAPH PATTERN: Record internal gap link for short-track pruning protection!
                        internal_gap_pairs.append((source_id, target_id))
                        outgoing.add(source_id)
                        used_starts.add(target_id)
                        stats["gap_internal_protected"] = stats.get("gap_internal_protected", 0) + 1
                        continue

                    if synthetic_added >= max_synthetic:
                        stats["gap_skipped_node_cap"] = stats.get("gap_skipped_node_cap", 0) + 1
                        continue
                    middle_id = next_id
                    next_id += 1
                    nodes_by_id[middle_id] = {
                        "node_id": middle_id,
                        "t": mid_t,
                        "z": max(0.0, float(mid_point[0])),
                        "y": max(0.0, float(mid_point[1])),
                        "x": max(0.0, float(mid_point[2])),
                        "gap_synthetic": 1,
                    }
                    synthetic_added += 1
                    stats["gap_inserted_synthetic"] = stats.get("gap_inserted_synthetic", 0) + 1

                middle = nodes_by_id[middle_id]
                if middle_reused:
                    used_isolated.add(middle_id)

                e1 = {
                    "source_id": source_id,
                    "target_id": middle_id,
                    "edge_prob": 0.90,
                    "distance_um": edge_distance_um(source, middle),
                    "gap_closed": 1,
                }
                e2 = {
                    "source_id": middle_id,
                    "target_id": target_id,
                    "edge_prob": 0.90,
                    "distance_um": edge_distance_um(middle, target),
                    "gap_closed": 1,
                }
                new_edges.extend([e1, e2])
                outgoing.add(source_id)
                incoming.add(middle_id)
                outgoing.add(middle_id)
                incoming.add(target_id)
                used_starts.add(target_id)
                stats["gap_pairs_selected"] = stats.get("gap_pairs_selected", 0) + 1
                stats["gap_added_edges"] = stats.get("gap_added_edges", 0) + 2

    if new_edges:
        edges = [*edges, *new_edges]
    return nodes_by_id, edges, internal_gap_pairs


def recover_strict_gap2(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], list[tuple[int, int]]]:
    if not OUTPUT_GAP2_RECOVERY or not edges or not nodes_by_id:
        return nodes_by_id, edges, []

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    predecessor = _single_predecessor_map(edges)
    successor = _single_successor_map(edges)

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)

    cap = min(GAP2_MAX_LINKS_ABS, max(1, int(round(len(edges) * GAP2_MAX_LINKS_FRAC))))
    proposals: list[tuple[float, int, int, int, float]] = []

    def pos_um(node_id: int) -> np.ndarray:
        node = nodes_by_id[node_id]
        return np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64) * np.array(VOXEL_SCALE_UM)

    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = starts_by_t.get(t + 3, [])
        if not end_ids or not start_ids:
            continue
        for end_id in end_ids:
            end_pos = pos_um(end_id)
            for start_id in start_ids:
                start_pos = pos_um(start_id)
                dist = float(np.linalg.norm(start_pos - end_pos))
                if dist > GAP2_MAX_TOTAL_UM or dist / 3.0 > GAP2_MAX_STEP_UM:
                    continue
                step = (start_pos - end_pos) / 3.0
                context_penalty = 0.0
                if GAP2_REQUIRE_CONTEXT:
                    ok_context = False
                    prev_id = predecessor.get(end_id)
                    if prev_id is not None:
                        prev_step = end_pos - pos_um(prev_id)
                        prev_norm = float(np.linalg.norm(prev_step))
                        step_norm = float(np.linalg.norm(step))
                        if prev_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(prev_step, step) / (prev_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(prev_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    next_id = successor.get(start_id)
                    if next_id is not None:
                        next_step = pos_um(next_id) - start_pos
                        next_norm = float(np.linalg.norm(next_step))
                        step_norm = float(np.linalg.norm(step))
                        if next_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(next_step, step) / (next_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(next_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    if not ok_context:
                        continue
                proposals.append((dist + 2.0 * context_penalty, end_id, start_id, t, dist))

    proposals.sort(key=lambda item: item[0])
    if not proposals:
        return nodes_by_id, edges

    selected = []
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    per_frame_count: dict[int, int] = {}
    for proposal in proposals:
        if len(selected) >= cap:
            break
        _, end_id, start_id, t, _ = proposal
        if end_id in used_ends or start_id in used_starts:
            continue
        frame_cap = max(1, int(round(len(ends_by_t.get(t, [])) * GAP2_FRAME_FRAC_CAP)))
        if per_frame_count.get(t, 0) >= frame_cap:
            continue
        selected.append(proposal)
        used_ends.add(end_id)
        used_starts.add(start_id)
        per_frame_count[t] = per_frame_count.get(t, 0) + 1

    if not selected:
        return nodes_by_id, edges, []

    internal_gap_pairs: list[tuple[int, int]] = []
    if not GAP2_INSERT_SYNTHETIC:
        # ASTRA DUAL-GRAPH PATTERN: Record internal gap2 links for short-track pruning protection!
        for _, end_id, start_id, t, _ in selected:
            internal_gap_pairs.append((end_id, start_id))
            stats["gap2_internal_protected"] = stats.get("gap2_internal_protected", 0) + 1
        return nodes_by_id, edges, internal_gap_pairs

    next_node_id = _next_node_id(nodes_by_id)
    new_edges = []
    for _, end_id, start_id, t, _ in selected:
        source = nodes_by_id[end_id]
        target = nodes_by_id[start_id]
        previous_id = end_id
        inserted_ids = []
        for k in (1, 2):
            frac = k / 3.0
            mid_t = int(source["t"]) + k
            midpoint = (
                float(source["z"]) + (float(target["z"]) - float(source["z"])) * frac,
                float(source["y"]) + (float(target["y"]) - float(source["y"])) * frac,
                float(source["x"]) + (float(target["x"]) - float(source["x"])) * frac,
            )
            node_id = next_node_id
            next_node_id += 1
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": mid_t,
                "z": max(0.0, float(midpoint[0])),
                "y": max(0.0, float(midpoint[1])),
                "x": max(0.0, float(midpoint[2])),
            }
            inserted_ids.append(node_id)
            current = nodes_by_id[node_id]
            new_edges.append({
                "source_id": previous_id,
                "target_id": node_id,
                "edge_prob": 0.85,
                "distance_um": edge_distance_um(nodes_by_id[previous_id], current),
                "gap2_recovered": 1,
            })
            previous_id = node_id
        new_edges.append({
            "source_id": previous_id,
            "target_id": start_id,
            "edge_prob": 0.85,
            "distance_um": edge_distance_um(nodes_by_id[previous_id], target),
            "gap2_recovered": 1,
        })
        stats["gap2_pairs_selected"] = stats.get("gap2_pairs_selected", 0) + 1
        stats["gap2_added_nodes"] = stats.get("gap2_added_nodes", 0) + len(inserted_ids)
        stats["gap2_added_edges"] = stats.get("gap2_added_edges", 0) + 3

    return nodes_by_id, [*edges, *new_edges], internal_gap_pairs


def add_safe_divisions_postlink(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> list[dict[str, object]]:
    if not OUTPUT_SAFE_DIVISIONS or not edges or not nodes_by_id:
        return edges

    out_by_source: dict[int, list[dict[str, object]]] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.add(int(edge["target_id"]))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)

    existing_edges = {(int(edge["source_id"]), int(edge["target_id"])) for edge in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP)))
    added: list[dict[str, object]] = []
    used_targets: set[int] = set()
    used_sources: set[int] = set()

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [node_id for node_id in ids_by_t[t] if len(out_by_source.get(node_id, [])) == 1]
        candidate_ids = [node_id for node_id in child_frame_ids if node_id not in incoming and node_id not in used_targets]
        if not source_ids or not candidate_ids:
            continue

        candidate_tree = None
        if SAFE_DIV_REQUIRE_MUTUAL_NN:
            candidate_positions = np.stack([_position_um(nodes_by_id[cid]) for cid in candidate_ids])
            candidate_tree = cKDTree(candidate_positions)

        frame_cap = max(1, int(round(len(source_ids) * SAFE_DIV_FRAME_FRAC_CAP)))
        proposals: list[tuple[float, int, int, float, float]] = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge["target_id"])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child["t"]) != t + 1:
                continue
            child_dist = edge_distance_um(source, existing_child)
            if child_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM:
                continue

            mutual_nn_id = None
            if candidate_tree is not None:
                _, nn_idx = candidate_tree.query(_position_um(existing_child))
                mutual_nn_id = candidate_ids[int(nn_idx)]

            for candidate_id in candidate_ids:
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_distance_um(source, candidate)
                if parent_dist > SAFE_DIV_MAX_UM:
                    continue
                sister_dist = edge_distance_um(existing_child, candidate)
                if sister_dist > SAFE_DIV_SISTER_MAX_UM:
                    continue

                if SAFE_DIV_REQUIRE_MUTUAL_NN and candidate_id != mutual_nn_id:
                    stats["safe_division_mutual_nn_rejected"] = stats.get("safe_division_mutual_nn_rejected", 0) + 1
                    continue

                if SAFE_DIV_REQUIRE_DIVERGENCE:
                    c1_succ = out_by_source.get(existing_child_id, [])
                    q_succ = out_by_source.get(candidate_id, [])
                    if len(c1_succ) != 1 or len(q_succ) != 1:
                        stats["safe_division_divergence_rejected"] = stats.get("safe_division_divergence_rejected", 0) + 1
                        continue
                    c1_grandchild = nodes_by_id.get(int(c1_succ[0]["target_id"]))
                    q_grandchild = nodes_by_id.get(int(q_succ[0]["target_id"]))
                    if (
                        c1_grandchild is None or q_grandchild is None
                        or int(c1_grandchild["t"]) != t + 2
                        or int(q_grandchild["t"]) != t + 2
                    ):
                        stats["safe_division_divergence_rejected"] = stats.get("safe_division_divergence_rejected", 0) + 1
                        continue
                    grandchild_dist = edge_distance_um(c1_grandchild, q_grandchild)
                    if grandchild_dist - sister_dist < SAFE_DIV_DIVERGE_UM:
                        stats["safe_division_divergence_rejected"] = stats.get("safe_division_divergence_rejected", 0) + 1
                        continue

                # Sister symmetry precision gate (tau <= 0.60)
                if SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0:
                    _sym_denom = max((child_dist + parent_dist) / 2.0, 1e-6)
                    if abs(child_dist - parent_dist) / _sym_denom > SAFE_DIV_SISTER_SYMMETRY_TAU:
                        stats["safe_division_symmetry_rejected"] = stats.get("safe_division_symmetry_rejected", 0) + 1
                        continue

                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))

        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if len(added) >= global_cap:
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            if source_id in used_sources:
                continue
            added.append({
                "source_id": source_id,
                "target_id": candidate_id,
                "edge_prob": 0.90,
                "distance_um": parent_dist,
                "safe_division": 1,
            })
            used_targets.add(candidate_id)
            used_sources.add(source_id)
            added_this_frame += 1

    if added:
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added]
    return edges


def filter_short_track_components(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    internal_gap_pairs: list[tuple[int, int]] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_FILTER_SHORT_TRACKS or OUTPUT_MIN_TRACK_LEN <= 1 or not edges:
        return nodes_by_id, edges

    parent = {node_id: node_id for node_id in nodes_by_id}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[ra] = rb

    out_count: dict[int, int] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        union(source_id, target_id)
        out_count[source_id] = out_count.get(source_id, 0) + 1

    # ASTRA DUAL-GRAPH PATTERN: Union internal gap associations to calculate combined lineage support
    if internal_gap_pairs:
        for u, v in internal_gap_pairs:
            if u in parent and v in parent:
                union(u, v)
                stats["short_track_internal_gap_united"] = stats.get("short_track_internal_gap_united", 0) + 1

    components: dict[int, list[int]] = {}
    for node_id in nodes_by_id:
        components.setdefault(find(node_id), []).append(node_id)

    component_edges: dict[int, list[dict[str, object]]] = {root: [] for root in components}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        if source_id in parent and target_id in parent:
            component_edges.setdefault(find(source_id), []).append(edge)

    keep: set[int] = set()
    for root, members in components.items():
        has_division = any(out_count.get(node_id, 0) >= 2 for node_id in members)
        if len(members) >= OUTPUT_MIN_TRACK_LEN or (OUTPUT_KEEP_DIVISION_COMPONENTS and has_division):
            keep.update(members)

    if not keep:
        return nodes_by_id, edges

    removed_before_rescue = len(nodes_by_id) - len(keep)
    if removed_before_rescue > 0 and ADAPTIVE_SHORT_TRACK_RESCUE:
        removed_frac = removed_before_rescue / max(len(nodes_by_id), 1)
        if removed_frac >= SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC:
            budget = min(SHORT_TRACK_RESCUE_MAX_NODES_ABS, max(0, int(round(len(nodes_by_id) * SHORT_TRACK_RESCUE_MAX_NODES_FRAC))))
            proposals = []
            for root, members in components.items():
                if set(members) & keep:
                    continue
                if len(members) < SHORT_TRACK_RESCUE_MIN_LEN or len(members) >= OUTPUT_MIN_TRACK_LEN:
                    continue
                c_edges = component_edges.get(root, [])
                if not c_edges:
                    continue
                probs = [float(e.get("edge_prob", 0.0)) for e in c_edges if np.isfinite(float(e.get("edge_prob", 0.0)))]
                dists = [float(e.get("distance_um", np.nan)) for e in c_edges if np.isfinite(float(e.get("distance_um", np.nan)))]
                mean_prob = float(np.mean(probs)) if probs else 0.0
                mean_dist = float(np.mean(dists)) if dists else float("inf")
                if mean_prob < SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB or mean_dist > SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM:
                    continue
                score = mean_prob - 0.02 * mean_dist + 0.004 * len(members)
                proposals.append((score, len(members), root, members))
            proposals.sort(reverse=True)
            rescued_nodes = 0
            for _, size, _, members in proposals:
                if budget <= 0 or rescued_nodes + size > budget:
                    continue
                keep.update(members)
                rescued_nodes += size
            stats["short_track_rescued_nodes"] = rescued_nodes

    kept_nodes = {nid: node for nid, node in nodes_by_id.items() if nid in keep}
    kept_edges = [edge for edge in edges if int(edge["source_id"]) in kept_nodes and int(edge["target_id"]) in kept_nodes]
    stats["short_track_nodes_removed"] = len(nodes_by_id) - len(kept_nodes)
    stats["short_track_edges_removed"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


def linefit_smooth_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> dict[int, dict[str, object]]:
    if not OUTPUT_LINEFIT_SMOOTH or OUTPUT_LINEFIT_WEIGHT <= 0 or OUTPUT_LINEFIT_WINDOW <= 0 or not edges:
        return nodes_by_id

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for edge in edges:
        src = int(edge["source_id"])
        tgt = int(edge["target_id"])
        predecessor.setdefault(tgt, []).append(src)
        successor.setdefault(src, []).append(tgt)

    updated_pos: dict[int, np.ndarray] = {}
    w = OUTPUT_LINEFIT_WINDOW
    weight = OUTPUT_LINEFIT_WEIGHT

    for nid, node in nodes_by_id.items():
        preds = predecessor.get(nid, [])
        succs = successor.get(nid, [])
        if len(preds) != 1 or len(succs) != 1:
            continue
        p_id = preds[0]
        s_id = succs[0]
        if len(successor.get(p_id, [])) != 1 or len(predecessor.get(s_id, [])) != 1:
            continue

        p_node = nodes_by_id.get(p_id)
        s_node = nodes_by_id.get(s_id)
        if p_node is None or s_node is None:
            continue
        if int(node["t"]) != int(p_node["t"]) + 1 or int(s_node["t"]) != int(node["t"]) + 1:
            continue

        raw_pos = np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64)
        pred_pos = np.array([float(p_node["z"]), float(p_node["y"]), float(p_node["x"])], dtype=np.float64)
        succ_pos = np.array([float(s_node["z"]), float(s_node["y"]), float(s_node["x"])], dtype=np.float64)
        linear_est = 0.5 * (pred_pos + succ_pos)
        smoothed = (1.0 - weight) * raw_pos + weight * linear_est
        updated_pos[nid] = smoothed
        stats["linefit_smoothed_nodes"] = stats.get("linefit_smoothed_nodes", 0) + 1

    for nid, pos in updated_pos.items():
        nodes_by_id[nid]["z"] = max(0.0, float(pos[0]))
        nodes_by_id[nid]["y"] = max(0.0, float(pos[1]))
        nodes_by_id[nid]["x"] = max(0.0, float(pos[2]))

    return nodes_by_id


def filter_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
    dataset: str | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, int]]:
    stats = {}
    edges = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge["source_id"]))
        target = nodes_by_id.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if OUTPUT_ENFORCE_NEXT_FRAME and int(target["t"]) != int(source["t"]) + 1:
            continue
        dist = edge_distance_um(source, target)
        edge["distance_um"] = dist
        if OUTPUT_EDGE_MAX_UM > 0 and dist > OUTPUT_EDGE_MAX_UM:
            continue
        edges.append(edge)

    # Single parent repair
    if OUTPUT_SINGLE_PARENT_REPAIR and edges:
        best_by_target = {}
        for edge in edges:
            tid = int(edge["target_id"])
            prev = best_by_target.get(tid)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_target[tid] = edge
        kept_ids = {id(e) for e in best_by_target.values()}
        edges = [e for e in edges if id(e) in kept_ids]

    # Gap closing (1-frame & 2-frame) with Dual-Graph Internal Gap Protection
    nodes_by_id, edges, gap1_internal = close_single_frame_gaps(nodes_by_id, edges, stats)
    nodes_by_id, edges, gap2_internal = recover_strict_gap2(nodes_by_id, edges, stats)
    internal_gap_pairs = [*gap1_internal, *gap2_internal]
    stats["total_internal_gap_pairs"] = len(internal_gap_pairs)

    # Biological safe divisions
    edges = add_safe_divisions_postlink(nodes_by_id, edges, stats)

    # Division geometry filter
    if OUTPUT_DIVISION_GEOMETRY_FILTER and edges:
        by_source = {}
        for edge in edges:
            by_source.setdefault(int(edge["source_id"]), []).append(edge)
        filtered = []
        for src_id, src_edges in by_source.items():
            if len(src_edges) <= 1:
                filtered.extend(src_edges)
                continue
            ranked = sorted(src_edges, key=edge_sort_key, reverse=True)
            top1, top2 = ranked[0], ranked[1]
            d1, d2 = float(top1["distance_um"]), float(top2["distance_um"])
            n1 = nodes_by_id.get(int(top1["target_id"]))
            n2 = nodes_by_id.get(int(top2["target_id"]))
            sister = edge_distance_um(n1, n2) if (n1 and n2) else 999.0
            if max(d1, d2) <= DIV_PARENT_MAX_UM and sister <= DIV_SISTER_MAX_UM:
                filtered.extend([top1, top2])
            elif DIV_DROP_TO_SINGLE_IF_BAD:
                filtered.append(top1)
            else:
                filtered.extend(ranked)
        edges = filtered

    # Prune isolated nodes (keep nodes with edges or internal gap endpoints)
    if OUTPUT_PRUNE_ISOLATED and edges:
        incident = {int(e["source_id"]) for e in edges} | {int(e["target_id"]) for e in edges}
        gap_nodes = {u for u, v in internal_gap_pairs} | {v for u, v in internal_gap_pairs}
        nodes_by_id = {nid: n for nid, n in nodes_by_id.items() if (nid in incident or nid in gap_nodes)}
        edges = [e for e in edges if int(e["source_id"]) in nodes_by_id and int(e["target_id"]) in nodes_by_id]

    # Filter short track components (length < 6) with Dual-Graph protection
    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats, internal_gap_pairs=internal_gap_pairs)

    # Linefit track interior smoothing
    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)

    return nodes_by_id, edges, stats
