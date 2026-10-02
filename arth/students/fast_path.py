"""Tier 1a: per-taxonomy fast path. Frozen embeddings + softmax regression on soft labels.

Trained directly against soft targets (cross-entropy to the teacher's distribution)
with minibatch Adam, which handles sparse 16k-dimensional hashing features in a
couple of seconds. The same small model is what ships on-device for fixed taxonomies.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..calibration import softmax


def _dense(m) -> np.ndarray:
    return m.toarray() if sp.issparse(m) else np.asarray(m)


class FastPath:
    def __init__(self, n_classes: int, l2: float = 1e-6, seed: int = 0):
        self.n_classes = n_classes
        self.l2 = l2
        self.seed = seed
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.temperature = 1.0

    def fit(self, X, Y: np.ndarray, epochs: int = 8, batch: int = 256, lr: float = 0.02) -> "FastPath":
        """Y: N x k soft targets (rows sum to 1)."""
        X = sp.csr_matrix(X, dtype=np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.W = np.zeros((d, self.n_classes), dtype=np.float32)
        # start from the class prior so rare options are not over-predicted early
        self.b = np.log(np.clip(Y.mean(axis=0), 1e-4, None)).astype(np.float32)
        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        t = 0
        for _ in range(epochs):
            order = rng.permutation(n)
            for i in range(0, n, batch):
                idx = order[i:i + batch]
                Xb = X[idx]
                P = softmax(_dense(Xb @ self.W) + self.b)
                G = (P - Y[idx]) / len(idx)
                gW = _dense(Xb.T @ G) + self.l2 * self.W
                gb = G.sum(axis=0)
                t += 1
                for p, g, m, v in ((self.W, gW, mW, vW), (self.b, gb, mb, vb)):
                    m *= 0.9
                    m += 0.1 * g
                    v *= 0.999
                    v += 0.001 * g * g
                    p -= (lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)).astype(np.float32)
        return self

    def logits(self, X) -> np.ndarray:
        return _dense(X @ self.W) + self.b

    def predict_proba(self, X) -> np.ndarray:
        return softmax(self.logits(X), self.temperature)
