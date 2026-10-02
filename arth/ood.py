"""FR-9: flag inputs unlike the training data by kNN distance in embedding space."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _dense(m) -> np.ndarray:
    return m.toarray() if sp.issparse(m) else np.asarray(m)


class KnnOOD:
    """Mean cosine similarity to the k nearest reference lines.

    The threshold is a low quantile of the same score on held-out in-distribution
    lines, so by construction about ``quantile`` of normal traffic is flagged.
    """

    def __init__(self, k: int = 5, quantile: float = 0.01):
        self.k = k
        self.quantile = quantile
        self.ref = None
        self.threshold = -1.0

    def fit(self, X_ref, X_holdout) -> "KnnOOD":
        self.ref = sp.csr_matrix(X_ref) if sp.issparse(X_ref) else np.asarray(X_ref, dtype=np.float32)
        s = self.score(X_holdout)
        self.threshold = float(np.quantile(s, self.quantile)) if len(s) else -1.0
        return self

    def score(self, X, chunk: int = 512) -> np.ndarray:
        out = []
        for i in range(0, X.shape[0], chunk):
            sims = _dense(X[i:i + chunk] @ self.ref.T)
            k = min(self.k, sims.shape[1])
            top = np.partition(sims, -k, axis=1)[:, -k:]
            out.append(top.mean(axis=1))
        return np.concatenate(out) if out else np.zeros(0)

    def is_ood(self, X) -> np.ndarray:
        return self.score(X) < self.threshold
