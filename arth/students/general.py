"""Tier 1b: the general bring-your-own-taxonomy student.

A bilinear dual encoder over the shared embedder: the score of option o for line x is

    z(x, o) = s * <x, o> + <x A, o B> + <o, w>

where x and o are embeddings of the rendered line and of the option's name plus
description. Options only enter through their text, so any caller taxonomy works at
inference time. Yes/no questions score the question text against a learned null
logit ``c``: P(yes) = sigmoid(z(x, q) - c).

It is trained on many taxonomy *views* of the same pool (built-in taxonomies plus
random regroupings with paraphrased descriptions) so it learns what descriptions mean
rather than memorising one label list. Targets are soft: each pool line carries a
distribution over the 32 concepts (from the teacher, or smoothed generator labels
offline) that is pushed through each view's mapping.

The PRD's production plan uses GLiClass-modern-base for this tier; this numpy model
keeps the same contract (options in the input, calibrated probabilities) and runs in
milliseconds on CPU. It is the bar GLiClass has to beat before it ships.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..calibration import softmax
from ..concepts import CONCEPT_LIST, CONCEPTS, PARAPHRASES
from ..questions import Question


def _dense(m) -> np.ndarray:
    return m.toarray() if sp.issparse(m) else np.asarray(m)


@dataclass
class View:
    """A training taxonomy: a question plus a concept -> option (or bool) mapping."""

    question: Question
    mapping: dict[str, str] | None = None  # choice: concept -> option
    positives: frozenset[str] = field(default_factory=frozenset)  # bool

    def target(self, concept_probs: np.ndarray) -> np.ndarray:
        """Soft target for rows given their concept distributions (N x 32)."""
        if self.question.type == "bool":
            idx = [CONCEPT_LIST.index(c) for c in self.positives]
            return concept_probs[:, idx].sum(axis=1)
        opts = self.question.option_names
        M = np.zeros((len(CONCEPT_LIST), len(opts)), dtype=np.float32)
        for c, o in self.mapping.items():
            M[CONCEPT_LIST.index(c), opts.index(o)] = 1.0
        return concept_probs @ M


def random_views(n: int, seed: int = 0) -> list[View]:
    """Random regroupings of concepts with paraphrased option names and descriptions."""
    r = random.Random(seed)
    views = []
    for i in range(n):
        concepts = CONCEPT_LIST[:]
        r.shuffle(concepts)
        mode = r.random()
        if mode < 0.25:  # subset of concepts plus a catch-all
            keep = concepts[: r.randint(3, 10)]
            groups = [[c] for c in keep] + [[c for c in concepts if c not in keep]]
            names = [r.choice([c, c.replace("_", " "), PARAPHRASES[c][0].lower().replace(" ", "_")]) for c in keep]
            names.append(r.choice(["other", "everything_else", "misc", "uncategorized"]))
            descs = [r.choice(PARAPHRASES[c] + [CONCEPTS[c]]) for c in keep]
            descs.append(r.choice(["Anything else", "None of the above", "All other transactions"]))
        else:
            k = r.randint(4, 24)
            cuts = sorted(r.sample(range(1, len(concepts)), k - 1))
            groups = [concepts[a:b] for a, b in zip([0] + cuts, cuts + [len(concepts)])]
            names, descs = [], []
            for g in groups:
                head = g[0]
                names.append(r.choice([head, f"{head}_etc", PARAPHRASES[head][0].lower().replace(" ", "_")]))
                parts = [r.choice(PARAPHRASES[c] + [CONCEPTS[c]]) for c in g[:4]]
                descs.append(r.choice(["", "", "Includes: "]) + "; ".join(parts))
        # De-duplicate option names.
        seen: dict[str, int] = {}
        uniq = []
        for nm in names:
            seen[nm] = seen.get(nm, 0) + 1
            uniq.append(nm if seen[nm] == 1 else f"{nm}_{seen[nm]}")
        q = Question.choice(f"view{i}", dict(zip(uniq, descs)))
        mapping = {c: uniq[gi] for gi, g in enumerate(groups) for c in g}
        views.append(View(q, mapping))
    # Random yes/no questions over one or two concepts.
    for i in range(n // 2):
        cs = r.sample(CONCEPT_LIST, r.choice([1, 1, 2]))
        desc = " or ".join(r.choice(PARAPHRASES[c] + [CONCEPTS[c]]) for c in cs)
        views.append(View(Question.boolean(f"bview{i}", r.choice(["", "Is this: ", "Is it "]) + desc),
                          positives=frozenset(cs)))
    return views


class GeneralStudent:
    def __init__(self, dim: int, rank: int = 96, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.A = (rng.standard_normal((dim, rank)) * 0.01).astype(np.float32)
        self.B = (rng.standard_normal((dim, rank)) * 0.01).astype(np.float32)
        self.w = np.zeros(dim, dtype=np.float32)
        self.s = np.float32(5.0)
        self.c = np.float32(2.0)
        self.temperature = 1.0
        self.seed = seed

    # ---- scoring --------------------------------------------------------
    def _logits(self, X, O) -> np.ndarray:
        XA = _dense(X @ self.A)
        OB = _dense(O @ self.B)
        lex = _dense(X @ O.T)
        prior = _dense(O @ self.w).reshape(1, -1)
        return self.s * lex + XA @ OB.T + prior

    def logits(self, X, question: Question, embed) -> np.ndarray:
        """Raw logits (N x k for choice; N x 2 [no, yes] for bool)."""
        if question.type == "bool":
            q = embed([question.bool_text()])
            z = self._logits(X, q)[:, 0]
            return np.stack([np.full_like(z, self.c), z], axis=1)
        return self._logits(X, embed(question.option_texts()))

    def predict_proba(self, X, question: Question, embed) -> np.ndarray:
        return softmax(self.logits(X, question, embed), self.temperature)

    # ---- training -------------------------------------------------------
    def fit(self, X, concept_probs: np.ndarray, views: list[View], embed, epochs: int = 6,
            batch: int = 256, lr: float = 2e-3, weight_decay: float = 1e-5, verbose: bool = False) -> "GeneralStudent":
        rng = np.random.default_rng(self.seed)
        X = sp.csr_matrix(X) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        prepared = []
        for v in views:
            if v.question.type == "bool":
                O = embed([v.question.bool_text()])
            else:
                O = embed(v.question.option_texts())
            prepared.append((v, O, v.target(concept_probs).astype(np.float32)))
        params = {"A": self.A, "B": self.B, "w": self.w}
        m = {k: np.zeros_like(p) for k, p in params.items()}
        vv = {k: np.zeros_like(p) for k, p in params.items()}
        sc = np.zeros(2, dtype=np.float64)  # s, c are scalars: plain SGD with momentum
        sc_m = np.zeros(2)
        t = 0
        steps_per_epoch = max(1, n // batch)
        for ep in range(epochs):
            losses = []
            for _ in range(steps_per_epoch * 2):
                v, O, Y = prepared[rng.integers(len(prepared))]
                idx = rng.choice(n, size=min(batch, n), replace=False)
                Xb = X[idx]
                XA = _dense(Xb @ self.A)
                OB = _dense(O @ self.B)
                lex = _dense(Xb @ O.T)
                z = self.s * lex + XA @ OB.T + _dense(O @ self.w).reshape(1, -1)
                if v.question.type == "bool":
                    y = Y[idx]
                    zz = z[:, 0] - self.c
                    p = 1.0 / (1.0 + np.exp(-zz))
                    losses.append(float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))))
                    dz = ((p - y) / len(idx)).reshape(-1, 1).astype(np.float32)
                    dc = -float(dz.sum())
                else:
                    y = Y[idx]
                    P = softmax(z)
                    losses.append(float(-np.mean(np.sum(y * np.log(P + 1e-9), axis=1))))
                    dz = ((P - y) / len(idx)).astype(np.float32)
                    dc = 0.0
                gA = _dense(Xb.T @ (dz @ OB)) + weight_decay * self.A
                gB = _dense(O.T @ (dz.T @ XA)) + weight_decay * self.B
                gw = _dense(O.T @ dz.sum(axis=0)).ravel()
                gs = float(np.sum(dz * lex))
                t += 1
                for k, g in (("A", gA), ("B", gB), ("w", gw)):
                    m[k] = 0.9 * m[k] + 0.1 * g
                    vv[k] = 0.999 * vv[k] + 0.001 * g * g
                    mh = m[k] / (1 - 0.9 ** t)
                    vh = vv[k] / (1 - 0.999 ** t)
                    params[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
                sc_m = 0.9 * sc_m + np.array([gs, dc])
                sc -= 0.05 * sc_m
                self.s = np.float32(max(0.0, 5.0 + sc[0]))
                self.c = np.float32(2.0 + sc[1])
            if verbose:
                print(f"  general epoch {ep + 1}/{epochs} loss={np.mean(losses):.4f} s={self.s:.2f} c={self.c:.2f}")
        return self
