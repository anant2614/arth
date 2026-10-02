"""Certified auto-label thresholds (FR-5).

A threshold t is *certified* when the one-sided Clopper–Pearson lower bound on the
precision of answers with confidence >= t is at least the target (98%) at the given
confidence level (95%). Thresholds are tested strictest-first on a fixed grid and
the walk stops at the first failure (fixed-sequence testing, as in Jevstiller), so
no multiple-testing correction is needed. Grid points with too few calibration
items to *ever* pass are skipped rather than ending the walk: whether a point is
testable depends only on the confidences, not on correctness, so skipping keeps
the guarantee conditional on the confidences.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import beta

DEFAULT_GRID = np.round(np.concatenate([
    [0.9999, 0.9995, 0.999, 0.998, 0.997, 0.996, 0.995],
    np.arange(0.99, 0.5 - 1e-9, -0.005),
]), 4)


def cp_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) Clopper–Pearson lower bound for k successes out of n."""
    if n == 0 or k == 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))


def min_n_to_pass(target: float, alpha: float = 0.05) -> int:
    """Smallest n for which n/n correct clears the bound: target**n <= alpha."""
    return math.ceil(math.log(alpha) / math.log(target))


@dataclass
class Certificate:
    threshold: float | None  # None: nothing can be auto-labelled
    target_precision: float
    alpha: float
    n_calibration: int
    n_above: int
    precision_above: float | None
    lower_bound: float | None
    coverage: float  # share of the calibration set auto-labelled

    def to_dict(self) -> dict:
        return asdict(self)

    def auto(self, confidence: float) -> bool:
        return self.threshold is not None and confidence >= self.threshold


def certify(confidence: np.ndarray, correct: np.ndarray, target: float = 0.98, alpha: float = 0.05,
            grid: np.ndarray = DEFAULT_GRID) -> Certificate:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n_min = min_n_to_pass(target, alpha)
    best: tuple[float, int, int, float] | None = None
    for t in sorted(grid, reverse=True):
        mask = confidence >= t
        n = int(mask.sum())
        if n < n_min:
            continue  # untestable; depends only on confidences
        k = int(correct[mask].sum())
        lb = cp_lower(k, n, alpha)
        if lb < target:
            break
        best = (float(t), n, k, lb)
    total = len(confidence)
    if best is None:
        return Certificate(None, target, alpha, total, 0, None, None, 0.0)
    t, n, k, lb = best
    return Certificate(t, target, alpha, total, n, k / n, lb, n / total if total else 0.0)


def coverage_at(confidence: np.ndarray, correct: np.ndarray, threshold: float | None) -> dict:
    """Coverage and realised precision of a certificate on a different (eval) split."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if threshold is None or len(confidence) == 0:
        return {"coverage": 0.0, "precision": None, "n_auto": 0}
    m = confidence >= threshold
    return {
        "coverage": float(m.mean()),
        "precision": float(correct[m].mean()) if m.any() else None,
        "n_auto": int(m.sum()),
    }


def cascade_pick(tier_probs: list[np.ndarray | None], threshold: float | None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Route rows through tiers in order; the first tier whose top probability clears the
    threshold answers. Rows no tier clears are answered (not auto) by the last tier that
    applies to them. Rows where a tier does not apply carry NaN in that tier's matrix.

    Returns (tier index, predicted label, confidence, auto) per row.
    """
    n = next(p.shape[0] for p in tier_probs if p is not None)
    tier = np.full(n, -1)
    pred = np.zeros(n, dtype=int)
    conf = np.zeros(n)
    auto = np.zeros(n, dtype=bool)
    for ti, p in enumerate(tier_probs):
        if p is None:
            continue
        valid = ~np.isnan(p).any(axis=1)
        top = np.nan_to_num(p, nan=-1.0).max(axis=1)
        arg = np.nan_to_num(p, nan=-1.0).argmax(axis=1)
        clears = valid & (top >= threshold) if threshold is not None else np.zeros(n, dtype=bool)
        take = ~auto & clears
        fallback = ~auto & ~clears & valid
        for m in (take, fallback):
            tier[m], pred[m], conf[m] = ti, arg[m], top[m]
        auto |= take
    return tier, pred, conf, auto


def certify_cascade(tier_probs: list[np.ndarray | None], y: np.ndarray, target: float = 0.98,
                    alpha: float = 0.05, grid: np.ndarray = DEFAULT_GRID) -> Certificate:
    """Certify one threshold shared by all tiers of the cascade (strictest-first).

    For a threshold t the auto-labelled set is every row some tier clears at t, answered
    by the first such tier. Each t defines a fixed rule, so fixed-sequence testing over
    the grid applies exactly as for a single model.
    """
    y = np.asarray(y)
    n_min = min_n_to_pass(target, alpha)
    total = len(y)
    best = None
    for t in sorted(grid, reverse=True):
        _, pred, _, auto = cascade_pick(tier_probs, float(t))
        n = int(auto.sum())
        if n < n_min:
            continue
        k = int((pred[auto] == y[auto]).sum())
        lb = cp_lower(k, n, alpha)
        if lb < target:
            break
        best = (float(t), n, k, lb)
    if best is None:
        return Certificate(None, target, alpha, total, 0, None, None, 0.0)
    t, n, k, lb = best
    return Certificate(t, target, alpha, total, n, k / n, lb, n / total if total else 0.0)
