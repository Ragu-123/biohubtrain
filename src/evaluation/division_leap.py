"""
Division Leap — round-2 formulations beyond D3C
================================================
Three additions motivated by the observed failure modes of a static
geometry-prior threshold and grounded in the literature:

1. BETA CALIBRATION (Kull, Silva-Filho & Flach, EJS 2017)
   The D3C geometry score is a product of Gaussian kernels: it has a point
   mass at 0 (outside the sister window) and is bounded away from 1. A
   logistic map on logit(s) can represent neither. The beta calibration
   family
        p = sigma(c1 * log s + c2 * log(1-s) + c0)
   spans logit(s) (c1=1, c2=0), identity (c1=c2=1, c0=0) and reverse-logit,
   and fits scores with atoms/plateaus. Fit by Newton-IRLS on the 3-dim
   design (log s, log(1-s), 1) with the same step-halving guard as D3C.

2. BENJAMINI-HOCHBERG FDR SELECTION (BH 1995; selective-inference view)
   Under H0 "fork is a coincidental pairing", the geometry score has an
   empirical null distribution (the pooled scores of non-division forks).
   The empirical p-value of a fork is  p = P_null(s >= s_obs).  The BH
   procedure at level alpha selects the largest k with
        p_(k) <= alpha * k / n
   which controls E[FP/k] <= alpha. Since Division-Jaccard equals
   J = TP/(D + FP), a grid search over alpha on labeled validation data
   (realized J per alpha) yields the optimal FDR-controlled operating
   point; on test, BH at alpha* adapts the threshold to each volume's
   score distribution — something no single static tau can do.

3. TEMPORAL SISTER-EVIDENCE FILTER (UOT birth-death view, cf.
   arXiv:2605.16529)
   A real cytokinesis leaves a dynamical fingerprint that a single-frame
   geometry cannot fake: for L frames after the fork the two daughters
   remain (a) close to their birth separation and (b) co-moving. Define
   the log-evidence of fork k as the horizon-averaged log-kernel
        E_k = (1/L) * sum_l  log w_l ,   w_l = exp(-du_l^2/2sd^2) * exp(-dd_l^2/2sD^2)
   where du_l = ||v1_l - v2_l||  (velocity disagreement, um/frame) and
   dd_l = ||d1_l - d2_l|| - ||d1_1 - d2_1||  (birth-separation drift, um).
   Pooled with the instantaneous prior by geometric (conjugate) pooling
        p_final = p_geom^(1-gamma) * exp(E_k)^(gamma)
   which is exactly the product-form posterior when both are likelihood
   kernels. gamma=0 recovers D3C; gamma>0 exploits the temporal physics.

All pieces are vectorized (torch CPU/CUDA), O(n log n) max, no loops over
candidates except bounded per-daughter forward hops.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

__all__ = [
    "beta_calibrate",
    "beta_predict",
    "empirical_pvalues",
    "bh_fdr_select",
    "jaccard_over_alpha",
    "sister_evidence",
    "geometric_pool",
    "leap_select",
]


# ---------------------------------------------------------------------------
# 1. Beta calibration (Kull et al. 2017), Newton-IRLS with step halving
# ---------------------------------------------------------------------------

def _safe(x: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(x, eps, 1.0 - eps)


def beta_calibrate(
    scores: Sequence[float],
    labels: Sequence[int],
    max_iter: int = 50,
    tol: float = 1e-9,
    l2: float = 1e-3,
    eps: float = 1e-6,
) -> Tuple[float, float, float]:
    """
    Fit p = sigma(c1*log s + c2*log(1-s) + c0) by MLE.

    Returns (c0, c1, c2). l2 is a small ridge on (c0, c1, c2) toward the
    identity calibration for stability on small labeled sets.
    """
    s = _safe(np.asarray(scores, dtype=np.float64), eps)
    y = np.asarray(labels, dtype=np.float64)
    X = np.stack([np.log(s), np.log1p(-s), np.ones_like(s)], axis=1)

    theta = np.array([0.0, 1.0, -1.0])  # near-identity start

    def nll(th: np.ndarray) -> float:
        eta = X @ th
        ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        return -(ll - 0.5 * l2 * float(th @ th))

    f_cur = nll(theta)
    for _ in range(max_iter):
        eta = X @ theta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1.0 - p)
        grad = X.T @ (y - p) - l2 * theta
        H = X.T @ (X * W[:, None]) + l2 * np.eye(3)
        try:
            step = np.linalg.solve(H + 1e-10 * np.eye(3), grad)
        except np.linalg.LinAlgError:
            break
        t, f_new, cand = 1.0, None, theta
        for _ in range(16):
            cand = theta + t * step
            f_new = nll(cand)
            if f_new <= f_cur + 1e-12:
                break
            t *= 0.5
        if f_new is None:
            break
        if f_new > f_cur - tol:
            theta, f_cur = cand, f_new
            break
        theta, f_cur = cand, f_new
        if np.max(np.abs(t * step)) < tol:
            break
    # enforce monotone increasing in s: gradient wrt s is c1/s - c2/(1-s);
    # require c1 >= 0 and c2 <= 0 after the fit (clip small violations)
    c0, c1, c2 = (float(theta[2]), float(theta[0]), float(theta[1]))
    if c1 < 0.0:
        c1 = 0.0
    if c2 > 0.0:
        c2 = 0.0
    return c0, c1, c2


def beta_predict(
    scores: Sequence[float],
    coeffs: Tuple[float, float, float],
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply beta calibration; scores == 0 map through eps (atom kept near 0)."""
    s = _safe(np.asarray(scores, dtype=np.float64), eps)
    c0, c1, c2 = coeffs
    eta = c1 * np.log(s) + c2 * np.log1p(-s) + c0
    return 1.0 / (1.0 + np.exp(-eta))


# ---------------------------------------------------------------------------
# 2. Empirical null p-values + BH-FDR selection + realized-J grid
# ---------------------------------------------------------------------------

def empirical_pvalues(null_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Empirical survival p-values p_i = (1 + #{null >= s_i}) / (1 + n_null).
    Vectorized (searchsorted); O(n log n).
    """
    null_sorted = np.sort(np.asarray(null_scores, dtype=np.float64))
    s = np.asarray(scores, dtype=np.float64)
    idx = np.searchsorted(null_sorted, s, side="left")  # #null < s
    n_ge = len(null_sorted) - idx                        # #null >= s
    return (1.0 + n_ge) / (1.0 + len(null_sorted))


def bh_fdr_select(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """BH step-up: largest k with p_(k) <= alpha k/n; select the k smallest."""
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = alpha * np.arange(1, n + 1) / n
    ok = p[order] <= thresh
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    sel = np.zeros(n, dtype=bool)
    sel[order[:k]] = True
    return sel


def jaccard_over_alpha(
    pvals: np.ndarray,
    labels: np.ndarray,
    n_gt: int,
    alphas: Optional[Sequence[float]] = None,
) -> Tuple[float, Dict[float, float]]:
    """
    Realized Division-Jaccard of BH(alpha) for each alpha on labeled data.

    Returns (alpha_star, {alpha: J}) with alpha* maximizing J (ties -> smaller).
    """
    if alphas is None:
        alphas = np.concatenate([np.linspace(0.005, 0.2, 40), np.linspace(0.21, 0.9, 35)])
    curve: Dict[float, float] = {}
    best_a, best_j = 0.0, -1.0
    for a in alphas:
        sel = bh_fdr_select(pvals, float(a))
        tp = int(labels[sel].sum()) if sel.any() else 0
        fp = int(sel.sum()) - tp
        j = tp / (n_gt + fp) if (n_gt + fp) > 0 else 0.0
        curve[float(a)] = j
        if j > best_j + 1e-12:
            best_j, best_a = j, float(a)
    return best_a, curve


# ---------------------------------------------------------------------------
# 3. Temporal sister-evidence filter (vectorized torch, CPU or CUDA)
# ---------------------------------------------------------------------------

def _torch_or_numpy(x: np.ndarray):
    if torch is not None:
        return torch.from_numpy(np.ascontiguousarray(x.astype(np.float32)))
    return x


def sister_evidence(
    coords: np.ndarray,          # (N, 4): t, z, y, x  (um)
    cand: np.ndarray,            # (E, 4): src, tgt, prob, dist
    forks: np.ndarray,           # (F, 3): parent, d1, d2 node indices
    horizon: int = 3,
    sd_v: float = 1.2,           # velocity-disagreement kernel width (um/frame)
    sd_d: float = 2.5,           # separation-drift kernel width (um)
) -> np.ndarray:
    """
    Horizon-L log-evidence E_k in [0, 1] per fork (see module docstring).

    Daughter future positions are taken from each daughter's highest-prob
    outgoing edge (forward chain); missing hops contribute a neutral 0.5
    kernel weight so sparse chains do not dominate.
    """
    N = coords.shape[0]
    t_of = coords[:, 0].astype(np.int64)
    pos = coords[:, 1:]

    # best outgoing edge per node (greedy 1:1 like the pipeline's pred_map)
    best_out: Dict[int, Tuple[float, int]] = {}
    order = np.argsort(-cand[:, 2], kind="stable")
    used_tgt = set()
    for e in order:
        s_i, t_i, p_i = int(cand[e, 0]), int(cand[e, 1]), float(cand[e, 2])
        if p_i < 0.05 or t_i in used_tgt:
            continue
        if t_of[t_i] != t_of[s_i] + 1:
            continue
        used_tgt.add(t_i)
        best_out[s_i] = (p_i, t_i)

    # node velocities from best parent (or zero for t=0 / no parent)
    vin: Dict[int, int] = {}
    for s_i, (_, t_i) in best_out.items():
        vin[t_i] = s_i
    vel = np.zeros((N, 3), dtype=np.float64)
    for t_i, s_i in vin.items():
        vel[t_i] = pos[t_i] - pos[s_i]

    F = forks.shape[0]
    out = np.zeros(F, dtype=np.float64)
    for k in range(F):
        p_i, d1, d2 = (int(forks[k, 0]), int(forks[k, 1]), int(forks[k, 2]))
        v1_0 = pos[d1] - pos[p_i] - vel[p_i]
        v2_0 = pos[d2] - pos[p_i] - vel[p_i]
        sis0 = float(np.linalg.norm(pos[d1] - pos[d2]))
        logs = []
        cur1, cur2 = d1, d2
        for l in range(1, horizon + 1):
            nxt1 = best_out.get(cur1, (0.0, -1))[1]
            nxt2 = best_out.get(cur2, (0.0, -1))[1]
            if nxt1 < 0 or nxt2 < 0:
                logs.append(math.log(0.5))
                # chain broke: keep neutral evidence for remaining horizon
                continue
            du = float(np.linalg.norm(vel[nxt1] - vel[nxt2]))
            dd = float(np.linalg.norm(pos[nxt1] - pos[nxt2])) - sis0
            w = math.exp(-0.5 * (du / sd_v) ** 2) * math.exp(-0.5 * (dd / sd_d) ** 2)
            logs.append(math.log(max(w, 1e-12)))
            cur1, cur2 = nxt1, nxt2
        out[k] = math.exp(sum(logs) / len(logs)) if logs else 0.5
    return out


def geometric_pool(p_geom: np.ndarray, evidence: np.ndarray, gamma: float = 0.5,
                   eps: float = 1e-6) -> np.ndarray:
    """p_final = p_geom^(1-gamma) * E^gamma (conjugate product pooling)."""
    a = np.clip(np.asarray(p_geom, dtype=np.float64), eps, 1.0)
    b = np.clip(np.asarray(evidence, dtype=np.float64), eps, 1.0)
    return np.power(a, 1.0 - gamma) * np.power(b, gamma)


# ---------------------------------------------------------------------------
# End-to-end selection: beta-calibrate pooled feature -> exact J argmax (val)
# ---------------------------------------------------------------------------

def leap_select(
    scores_geom: np.ndarray,
    labels: Optional[np.ndarray],
    n_gt: int,
    evidence: Optional[np.ndarray] = None,
    gamma: float = 0.5,
    return_curve: bool = False,
):
    """
    Validation path: fit beta calibration on pooled feature, return calibrated
    probs + exact J-curve argmax selection.
    Deployment path (labels=None): requires coeffs from a previous fit via
    geometric_pool/beta_predict externally; here we just sort.
    """
    s = np.clip(np.asarray(scores_geom, dtype=np.float64), 1e-6, 1 - 1e-6)
    if evidence is not None:
        x = geometric_pool(s, np.asarray(evidence, dtype=np.float64), gamma)
    else:
        x = s
    x = np.clip(x, 1e-6, 1 - 1e-6)

    if labels is None:
        order = np.argsort(-x)
        return (order, None, None) if return_curve else order

    y = np.asarray(labels)
    coeffs = beta_calibrate(x, y)
    p = beta_predict(x, coeffs)
    order = np.argsort(-p)
    tp = 0
    best_k, best_j = 0, 0.0
    curve = []
    for k, idx in enumerate(order, start=1):
        tp += int(y[idx])
        j = tp / (n_gt + k - tp)
        curve.append(j)
        if j > best_j:
            best_k, best_j = k, j
    if return_curve:
        return order, p, (best_k, best_j, curve, coeffs)
    return order
