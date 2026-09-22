"""Quick diagnostic: do the new hard gates reject the 3 GT divisions from HANDOFF.md?"""
import numpy as np
import math

VOXEL = (1.625, 0.40625, 0.40625)
PRIOR_SYMMETRY_GATE = -0.97  # from duplicate_parent_solver.py (uncommitted change)

# (parent, d1, d2) in (z,y,x) voxels, from HANDOFF.md section 4
gt_divs = [
    ((38, 84, 116), (40, 94, 110), (37, 70, 125), 24),
    ((49, 199, 233), (49, 206, 247), (48, 196, 229), 52),
    ((36, 45, 40), (37, 45, 43), (32, 42, 32), 62),
]

print("=== Gate check: PRIOR_SYMMETRY_GATE = %.2f (reject if cos_theta > gate) ===" % PRIOR_SYMMETRY_GATE)
for p, d1, d2, t in gt_divs:
    p = np.array(p, float); d1 = np.array(d1, float); d2 = np.array(d2, float)
    w1 = (d1 - p) * np.array(VOXEL)
    w2 = (d2 - p) * np.array(VOXEL)
    cos = float(np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2)))
    rejected = cos > (PRIOR_SYMMETRY_GATE + 1e-6)
    print(f"t={t}: cos={cos:+.4f} -> {'REJECTED by symmetry gate' if rejected else 'passes'}")

print()
print("=== Momentum-prior suppression check (sigma_accel=3.0, floor 0.35) ===")
# Daughter displacement deviates from parent's constant-velocity prediction.
# For a stationary dividing parent, daughters sit ~d_parent/2 off the predicted spot.
for d_advect in [3.0, 5.0, 7.0]:
    m_old = 0.5 + 0.5 * math.exp(-(d_advect**2) / (2 * 4.5**2))  # committed version
    m_new = 0.35 + 0.65 * math.exp(-(d_advect**2) / (2 * 3.0**2))  # uncommitted version
    print(f"advect error {d_advect:.0f} um: prob multiplier old={m_old:.3f} new={m_new:.3f}")

print()
print("=== End-to-end solver check on GT division geometry ===")
import sys
sys.path.insert(0, ".")
from duplicate_parent_solver import DuplicateParentTrackingSolver, compute_pairwise_physical_distances

solver = DuplicateParentTrackingSolver(use_mejc=True, c_app=0.10, c_div=0.65)
for p, d1, d2, t in gt_divs:
    src = np.array([p], float)
    tgt = np.array([d1, d2], float)
    dists = compute_pairwise_physical_distances(src, tgt, VOXEL)
    probs = np.exp(-(dists**2) / (2 * 5.5**2))
    edges = solver.solve_frame_pair(src, tgt, probs)
    n_div = sum(e.is_division for e in edges)
    print(f"t={t}: raw_probs={np.round(probs,3).tolist()} -> edges={len(edges)}, division_edges={n_div}"
          + ("   <-- DIVISION LOST" if n_div < 2 else ""))
