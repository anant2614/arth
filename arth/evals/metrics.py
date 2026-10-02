"""Classification metrics for the quality gates."""

from __future__ import annotations

import numpy as np


def per_class_f1(y_true, y_pred, labels) -> dict[str, dict]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for lab in labels:
        tp = int(((y_pred == lab) & (y_true == lab)).sum())
        fp = int(((y_pred == lab) & (y_true != lab)).sum())
        fn = int(((y_pred != lab) & (y_true == lab)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[str(lab)] = {"precision": p, "recall": r, "f1": f, "support": int((y_true == lab).sum())}
    return out


def macro_f1(y_true, y_pred, labels=None) -> float:
    """Macro-F1 over labels present in the gold labels (absent labels are not scored)."""
    y_true = np.asarray(y_true)
    labels = sorted(set(y_true.tolist())) if labels is None else [l for l in labels if (y_true == l).any()]
    if not labels:
        return 0.0
    pc = per_class_f1(y_true, y_pred, labels)
    return float(np.mean([v["f1"] for v in pc.values()]))


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    return float((y_true == np.asarray(y_pred)).mean()) if len(y_true) else 0.0


def summarize(y_true, y_pred, labels=None) -> dict:
    return {"n": len(y_true), "accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred, labels)}
