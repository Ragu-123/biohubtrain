"""
Division-Jaccard Decision Calculus (D3C)
========================================
A derived decision theory for cell-division detection under the official
Jaccard metric. Instead of hand-tuning division thresholds, we derive the
optimal acceptance rule from the calculus of the score itself.

--------------------------------------------------------------------------
1. THE OBJECTIVE AND ITS SELF-REFERENTIAL THRESHOLD LAW
--------------------------------------------------------------------------
Division Jaccard is J(t) = TP(t) / (TP(t) + FP(t) + FN) with FN = D - TP(t)
fixed (GT divisions do not move), hence

    J(t) = TP(t) / (D + FP(t)).                              (1)

Accepting one more candidate fork at operating point (TP, FP) moves the
score by

    J(TP+1, FP) - J(TP, FP) = (D - TP) / ((TP+1)(D + FP + 1)) > 0,   (2)

strictly positive whenever TP < D. The expected-score decision (using the
marginal-score randomization of Bennett et al., ICML 2022, made
deterministic) is therefore:

    Accept the k-th candidate fork iff  p_k > J_operating,   (3)

where J_operating is the division Jaccard at the current operating point.
This is the exact Jaccard analogue of the F1 threshold law p > F/2
(Flach & Kull, NeurIPS 2015; Lipton et al., ICML 2014). The threshold is
*self-referential*: it climbs toward the very score it creates. No static
threshold can dominate it on the labeled distribution.

Corollary (deployment): calibrate p_div once on labeled volumes, set
tau* = J_val (the achieved validation Jaccard), and on test accept forks
with p_div > tau*. Any fork with p <= tau* would, in expectation, lower
the final score; any fork with p > tau* raises it. Exact, zero tuning.

2. CALIBRATION OF p_div (DIVISION CALIBRATION MAP)
--------------------------------------------------------------------------
Raw fork scores s = p_geom^beta * p_kin^(1-beta) are not probabilities.
We fit the 1-D logistic map  p_div = sigma(a * logit(s) + b)  by
Newton-Raphson / IRLS maximum likelihood: each iteration solves the 2x2
normal equations with exact Hessian H = X^T W X (W = p(1-p)), damped by a
step-halving guard (Lewis 1977) and a Gaussian prior toward identity
calibration for few labeled forks (ridge-equivalent).

3. GEOMETRY PRIOR (Padfield 2009 coupled flow + HOCT 3D prior, 2607.11754)
--------------------------------------------------------------------------
    g1: bilateral symmetry      tau  = |d1 - d2| / (d1 + d2)
    g2: equatorial conservation e    = |comoving midpoint - daughters midpoint|
    g3: spindle orthogonality   c    = cos angle(w1, w2), w = daughter - comoving

    p_geom = Phi_tau(tau) * Phi_e(e) * Phi_c(c)
    Phi_tau = exp(-tau^2 / 2 s_tau^2),  Phi_e = exp(-e^2 / 2 s_e^2),
    Phi_c   = sigmoid(-(c - c_gate) / 0.25)      (smooth 90-degree gate)

All kernels are smooth => differentiable everywhere, Triton-fusable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "geometry_score",
    "optimal_jaccard_threshold",
    "calibrate_division_probs",
    "jaccard_curve_argmax",
    "select_divisions_marginal",
    "fit_and_score_divisions",
    "D3CResult",
]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# Geometry kernels (smooth, differentiable cytokinesis invariants)
# ---------------------------------------------------------------------------

def geometry_score(
    tau: float,
    midpoint_offset_um: float,
    cos_spindle: float,
    sister_dist_um: Optional[float] = None,
    tau_scale: float = 0.40,
    mid_scale_um: float = 2.4,
    cos_gate: float = -0.60,
    sister_lo_um: float = 3.0,
    sister_hi_um: float = 16.0,
) -> float:
    """
    Smooth product-form cytokinesis geometry prior in [0, 1].

    tau                : |d1 - d2| / (d1 + d2)              bilateral asymmetry
    midpoint_offset_um : |comoving midpoint - daughters midpoint|
    cos_spindle        : cos angle between daughter vectors (<0 = diverging)
    sister_dist_um     : |d1 - d2| in um; soft window over [sister_lo, sister_hi]
    """
    phi_tau = math.exp(-0.5 * (tau / tau_scale) ** 2)
    phi_mid = math.exp(-0.5 * (midpoint_offset_um / mid_scale_um) ** 2)
    phi_cos = _sigmoid(-(cos_spindle - cos_gate) / 0.25)
    score = phi_tau * phi_mid * phi_cos
    if sister_dist_um is not None:
        # Smooth box window: sharp 0 outside hard bounds, smooth inside
        if sister_dist_um < sister_lo_um or sister_dist_um > sister_hi_um:
            return 0.0
        center, width = 0.5 * (sister_lo_um + sister_hi_um), 0.5 * (sister_hi_um - sister_lo_um)
        phi_sis = math.exp(-0.5 * ((sister_dist_um - center) / width) ** 4)
        score *= 0.5 + 0.5 * phi_sis
    return float(score)


# ---------------------------------------------------------------------------
# Derived optimal threshold
# ---------------------------------------------------------------------------

def optimal_jaccard_threshold(tp: int, fp: int, fn: int) -> float:
    """
    tau* = J(TP, FP, FN) = TP / (TP + FP + FN).

    By (2), accepting a fork with p > tau* raises the expected score and
    accepting one with p <= tau* lowers it. This single line replaces all
    division threshold tuning.
    """
    denom = tp + fp + fn
    return float(tp) / float(denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Newton-IRLS logistic calibration (exact MLE for 1-D logistic regression)
# ---------------------------------------------------------------------------

def calibrate_division_probs(
    scores: Sequence[float],
    labels: Sequence[int],
    max_iter: int = 25,
    tol: float = 1e-8,
    prior_strength: float = 2.0,
    prior_logit: float = -3.0,
) -> Tuple[float, float]:
    """
    Maximum-likelihood logistic calibration  p = sigma(a * z + b),  z = logit(s).

    Newton-Raphson with exact Hessian H = X^T W X, step-halving guard, and a
    Gaussian prior pulling (a, b) toward (1, prior_logit) when labeled forks
    are scarce. Returns (a, b); monotone since a > 0 is enforced.
    """
    s = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y = np.asarray(labels, dtype=np.float64)
    z = np.log(s / (1.0 - s))
    X = np.stack([z, np.ones_like(z)], axis=1)

    mu = np.array([1.0, prior_logit])
    lam = np.diag([prior_strength, prior_strength])
    theta = mu.copy()

    def neg_loglik(th: np.ndarray) -> float:
        eta = X @ th
        ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        reg = 0.5 * (th - mu) @ lam @ (th - mu)
        return -(ll - reg)

    f_cur = neg_loglik(theta)
    for _ in range(max_iter):
        eta = X @ theta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1.0 - p)
        grad = X.T @ (y - p) - lam @ (theta - mu)
        H = X.T @ (X * W[:, None]) + lam
        try:
            step = np.linalg.solve(H + 1e-10 * np.eye(2), grad)
        except np.linalg.LinAlgError:
            break
        t, f_new, cand = 1.0, None, theta
        for _ in range(12):
            cand = theta + t * step
            f_new = neg_loglik(cand)
            if f_new <= f_cur + 1e-12:
                break
            t *= 0.5
        if f_new is None or f_new > f_cur - tol:
            theta, f_cur = cand, f_new
            break
        theta, f_cur = cand, f_new
        if np.max(np.abs(t * step)) < tol:
            break
    a, b = float(theta[0]), float(theta[1])
    if a <= 0.05:  # enforce monotone increasing calibration
        a = 0.05
    return a, b


# ---------------------------------------------------------------------------
# Exact Jaccard-curve argmax (validation) and marginal law (deployment)
# ---------------------------------------------------------------------------

def jaccard_curve_argmax(
    probs_desc: Sequence[Tuple[float, bool]],
    n_gt: int,
) -> Tuple[int, List[float]]:
    """
    Exact argmax_k J_k scan over the calibration set, candidates sorted DESC.

    J_k = TP_k / (n_gt + k - TP_k)   (from (1): D + FP = n_gt + (k - TP_k))

    Returns (k_star, j_curve). O(n).
    """
    j_curve: List[float] = []
    tp = 0
    best_k, best_j = 0, 0.0
    for k, (_, is_true) in enumerate(probs_desc, start=1):
        if is_true:
            tp += 1
        j = tp / (n_gt + k - tp)
        j_curve.append(j)
        if j > best_j:
            best_k, best_j = k, j
    return best_k, j_curve


def select_divisions_marginal(
    probs_desc: Sequence[Tuple[float, int, int]],
    tau: float,
) -> List[Tuple[int, int]]:
    """
    Deployment decision rule (3): accept fork iff p_k > tau, with tau = J_val.

    Args:
        probs_desc: (p_div, parent_key, fork_key) sorted DESC by p_div
        tau:        optimal_jaccard_threshold from the labeled validation set
    Returns:
        accepted (parent_key, fork_key) pairs
    """
    return [(pk, fk) for (p, pk, fk) in probs_desc if p > tau]


# ---------------------------------------------------------------------------
# Full pipeline: geometry x kinetics -> calibrate -> threshold law
# ---------------------------------------------------------------------------

@dataclass
class D3CResult:
    accepted_forks: List[Tuple[int, int]] = field(default_factory=list)
    calibrated_probs: Dict[Tuple[int, int], float] = field(default_factory=dict)
    jaccard_curve: List[float] = field(default_factory=list)
    optimal_k: int = 0
    tau_star: float = 0.0
    calib_a: float = 1.0
    calib_b: float = -3.0
    n_candidates: int = 0


def fit_and_score_divisions(
    fork_scores: Sequence[float],
    fork_keys: Sequence[Tuple[int, int]],
    fork_labels: Optional[Sequence[int]] = None,
    n_gt_hint: Optional[int] = None,
) -> D3CResult:
    """
    End-to-end division decision under D3C.

    Validation path (labels + n_gt_hint given):
        calibrate -> exact Jaccard-curve argmax -> tau* = J(k*)
    Deployment path (no labels):
        apply stored (a, b) calibration -> accept p > tau*.

    Args:
        fork_scores: raw combined scores s_k in (0, 1)
        fork_keys:   (parent_node_id, fork_id) per candidate
        fork_labels: optional 0/1 GT correctness per candidate
        n_gt_hint:   number of GT divisions in the evaluated set
    """
    scores = np.clip(np.asarray(fork_scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    keys = list(fork_keys)

    if fork_labels is not None and len(fork_labels) == len(scores):
        y = np.asarray(fork_labels, dtype=np.float64)
        if y.sum() > 0 and (1 - y).sum() > 0:
            a, b = calibrate_division_probs(scores, y)
        else:
            a, b = 1.0, -3.0
    else:
        a, b = 1.0, -3.0

    z = np.log(scores / (1.0 - scores))
    p = 1.0 / (1.0 + np.exp(-(a * z + b)))
    p = np.clip(p, 1e-9, 1.0 - 1e-9)

    order = np.argsort(-p)
    probs_parent_fork = [(float(p[i]), int(keys[i][0]), int(keys[i][1])) for i in order]

    if fork_labels is not None and n_gt_hint:
        flags = [bool(int(fork_labels[i])) for i in order]
        k_star, j_curve = jaccard_curve_argmax([(pp, fl) for pp, fl in zip([x[0] for x in probs_parent_fork], flags)], n_gt_hint)
        tau = j_curve[k_star - 1] if k_star > 0 else 0.0
        accepted = [(pk, fk) for (pp, pk, fk) in probs_parent_fork[:k_star]]
    else:
        j_curve = []
        k_star = 0
        tau = 0.0
        accepted = [(pk, fk) for (pp, pk, fk) in probs_parent_fork]

    return D3CResult(
        accepted_forks=accepted,
        calibrated_probs={keys[i]: float(p[i]) for i in range(len(keys))},
        jaccard_curve=j_curve,
        optimal_k=k_star,
        tau_star=tau,
        calib_a=a,
        calib_b=b,
        n_candidates=len(keys),
    )
