"""Temperature scaling, expected calibration error and reliability diagrams."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    p = softmax(logits, temperature)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds: tuple[float, float] = (0.05, 20.0)) -> float:
    """Single scalar T minimising NLL on held-out labels (Guo et al., 2017)."""
    if len(y) == 0:
        return 1.0
    res = minimize_scalar(lambda lt: nll(logits, y, float(np.exp(lt))),
                          bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(res.x))


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of top-label confidence, equal-width bins."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(confidence[m].mean() - correct[m].mean())
    return float(total)


def reliability(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict]:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "count": int(m.sum()),
            "mean_confidence": float(confidence[m].mean()) if m.any() else None,
            "accuracy": float(correct[m].mean()) if m.any() else None,
        })
    return rows
