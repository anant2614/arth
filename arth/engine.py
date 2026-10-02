"""The runtime cascade (FR-4, FR-6, FR-7, FR-9, FR-10).

Per line and question, tiers run cheapest first:

    correction memory -> rules (gazetteer) -> fast path (per taxonomy) -> general student -> teacher

A line is auto-labelled by the first tier whose calibrated confidence clears the
question's certified threshold. Unfamiliar inputs (kNN OOD), the fixed 2% audit slice
and everything during an audit alert go to the teacher. Without a teacher those lines
are answered by the students with ``auto = false`` and land in the review queue.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
import zlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .calibration import fit_temperature, softmax
from .certify import Certificate, cascade_pick, certify_cascade
from .concepts import BUILTIN_BOOLS, BUILTIN_TAXONOMIES, CONCEPT_LIST
from .embed import get_embedder
from .features import _cp_key
from .gazetteer import Gazetteer
from .ood import KnnOOD
from .prepare import Prepared, prepare
from .questions import Question
from .scrub import hash_id
from .students.fast_path import FastPath
from .students.general import GeneralStudent
from .synth import SPECS
from .teacher import Teacher, TeacherInput

log = logging.getLogger("arth")

TIER_NAMES = ["rules", "fast_path", "general", "ensemble"]
CORRECTION_WEIGHT = 5
CONCEPT_DIRECTION = {c: s.direction for c, s in SPECS.items()}


@dataclass
class QuestionModel:
    question: Question
    general_T: float = 1.0
    fast: FastPath | None = None
    rule_map: dict[str, int] | None = None  # gazetteer concept -> label index
    rule_conf: float = 0.9
    certificate: Certificate | None = None
    source: str = "generic"  # builtin | saved | generic

    @property
    def n_labels(self) -> int:
        return 2 if self.question.type == "bool" else len(self.question.options)


@dataclass
class TaxonomyModel:
    id: str
    questions: dict[str, QuestionModel]
    model_version: str


@dataclass
class StudentBundle:
    version: str
    embedder_spec: str
    general: GeneralStudent
    builtin: dict[str, QuestionModel]
    generic_T: dict[str, float]  # question type -> temperature for unseen taxonomies
    generic_cert: dict[str, Certificate]  # question type -> certificate for unseen taxonomies
    ood: KnnOOD
    pool_rendered: list[str]
    pool_concepts: list[str]
    manifest: dict = field(default_factory=dict)

    @property
    def embedder(self):
        return get_embedder(self.embedder_spec)


# ---------------------------------------------------------------------------
# Tier scoring


def rules_probs(qm: QuestionModel, preps: list[Prepared]) -> np.ndarray | None:
    if not qm.rule_map:
        return None
    out = np.full((len(preps), qm.n_labels), np.nan)
    for i, p in enumerate(preps):
        c = p.extracted.merchant_concept
        if c is None or c not in qm.rule_map:
            continue
        if p.direction and CONCEPT_DIRECTION.get(c) and p.direction != CONCEPT_DIRECTION[c]:
            continue  # a credit from a shop is a refund, not a purchase
        row = np.full(qm.n_labels, (1 - qm.rule_conf) / max(1, qm.n_labels - 1))
        row[qm.rule_map[c]] = qm.rule_conf
        out[i] = row
    return out


@lru_cache(maxsize=2048)
def _encode_options(spec: str, texts: tuple[str, ...]):
    return get_embedder(spec).encode(list(texts))


def option_encoder(bundle: StudentBundle):
    """Embedder for option/question texts, cached: taxonomies repeat on every request."""
    return lambda texts: _encode_options(bundle.embedder_spec, tuple(texts))


def tier_probs(bundle: StudentBundle, qm: QuestionModel, preps: list[Prepared], X) -> list[np.ndarray | None]:
    emb = option_encoder(bundle)
    general = softmax(bundle.general.logits(X, qm.question, emb), qm.general_T)
    fast = qm.fast.predict_proba(X) if qm.fast is not None else None
    # Last tier: both calibrated students averaged. It answers whatever nothing cheaper
    # certified, and is usually the most accurate and best calibrated of the three.
    ensemble = (fast + general) / 2 if fast is not None else None
    return [rules_probs(qm, preps), fast, general, ensemble]


def labels_for(q: Question, values: list) -> np.ndarray:
    if q.type == "bool":
        return np.array([1 if v in (True, 1, "yes", "true", "True") else 0 for v in values])
    names = q.option_names
    return np.array([names.index(v) for v in values])


# ---------------------------------------------------------------------------
# Fitting question models (built-in at bundle time, saved taxonomies at request time)


def learn_rule_map(bundle: StudentBundle, qm: QuestionModel, X_pool, min_agree: float = 0.9,
                   min_support: int = 8) -> dict[str, int]:
    """Map gazetteer concepts to options using the general student's majority answer on
    pool lines from that concept; keep only consistent, well-supported mappings."""
    probs = softmax(bundle.general.logits(X_pool, qm.question, bundle.embedder.encode), qm.general_T)
    pred = probs.argmax(axis=1)
    out = {}
    concepts = np.array(bundle.pool_concepts)
    for c in CONCEPT_LIST:
        m = concepts == c
        if m.sum() < min_support:
            continue
        vals, counts = np.unique(pred[m], return_counts=True)
        j = counts.argmax()
        if counts[j] / m.sum() >= min_agree:
            out[c] = int(vals[j])
    return out


def calibrate_and_certify(bundle: StudentBundle, qm: QuestionModel, preps: list[Prepared], X, y: np.ndarray,
                          target: float = 0.98, alpha: float = 0.05) -> QuestionModel:
    """Temperature-scale each tier on the labelled sample, set the rule confidence to its
    smoothed precision there, then certify one shared threshold for the cascade."""
    emb = bundle.embedder.encode
    if len(y):
        qm.general_T = fit_temperature(bundle.general.logits(X, qm.question, emb), y)
        if qm.fast is not None:
            qm.fast.temperature = fit_temperature(qm.fast.logits(X), y)
        if qm.rule_map:
            rp = rules_probs(QuestionModel(qm.question, rule_map=qm.rule_map, rule_conf=1.0), preps)
            hit = ~np.isnan(rp).any(axis=1)
            if hit.any():
                k = int((rp[hit].argmax(axis=1) == y[hit]).sum())
                qm.rule_conf = (k + 1) / (hit.sum() + 2)
    qm.certificate = certify_cascade(tier_probs(bundle, qm, preps, X), y, target, alpha)
    return qm


def fit_saved_taxonomy(bundle: StudentBundle, tax_id: str, questions: list[Question],
                       sample: list[dict], labels: dict[str, list], gaz: Gazetteer | None = None,
                       corrections: dict[str, list[tuple[str, object]]] | None = None,
                       target: float = 0.98, alpha: float = 0.05) -> TaxonomyModel:
    """FR-5. Fast path distilled from the general student's soft labels on the pool (plus
    any reviewer corrections, given as rendered student inputs); thresholds certified on
    the caller's labelled sample."""
    gaz = gaz or Gazetteer.default()
    emb = bundle.embedder.encode
    X_pool = emb(bundle.pool_rendered)
    preps = [prepare(r["text"], r.get("amount"), r.get("direction"), r.get("history"), gaz) for r in sample]
    X = emb([p.rendered for p in preps]) if preps else None
    out = {}
    for q in questions:
        qm = QuestionModel(q, general_T=bundle.generic_T.get(q.type, 1.0), source="saved")
        soft = softmax(bundle.general.logits(X_pool, q, emb), qm.general_T)
        Xf, Yf = X_pool, soft
        corr = (corrections or {}).get(q.name) or []
        if corr:  # reviewer labels, up-weighted by repetition
            Xc = emb([t for t, _ in corr] * CORRECTION_WEIGHT)
            Yc = np.eye(qm.n_labels)[labels_for(q, [v for _, v in corr] * CORRECTION_WEIGHT)]
            Xf = sp.vstack([X_pool, Xc]) if sp.issparse(X_pool) else np.vstack([X_pool, Xc])
            Yf = np.vstack([soft, Yc])
        qm.fast = FastPath(qm.n_labels).fit(Xf, Yf)
        qm.rule_map = learn_rule_map(bundle, qm, X_pool)
        y = labels_for(q, labels.get(q.name, [])) if q.name in labels else np.zeros(0, dtype=int)
        if X is not None and len(y) == len(preps) and len(y):
            calibrate_and_certify(bundle, qm, preps, X, y, target, alpha)
        else:
            qm.certificate = Certificate(None, target, alpha, 0, 0, None, None, 0.0)
        out[q.name] = qm
    return TaxonomyModel(tax_id, out, bundle.version)


def builtin_taxonomy(bundle: StudentBundle, name: str) -> TaxonomyModel | None:
    if name in bundle.builtin:
        return TaxonomyModel(name, {name: bundle.builtin[name]}, bundle.version)
    if name == "builtin":
        return TaxonomyModel(name, dict(bundle.builtin), bundle.version)
    return None


# ---------------------------------------------------------------------------
# Audit monitoring (FR-10)


class AuditMonitor:
    def __init__(self, rate: float = 0.02, target: float = 0.9, window: int = 500, min_samples: int = 50):
        self.rate = rate
        self.target = target
        self.window: deque[int] = deque(maxlen=window)
        self.min_samples = min_samples
        self.alert = False
        self._lock = threading.Lock()

    def selected(self, key: str) -> bool:
        return (zlib.crc32(key.encode()) % 10000) < self.rate * 10000

    def record(self, agreed: bool) -> None:
        with self._lock:
            self.window.append(int(agreed))
            n = len(self.window)
            if n >= self.min_samples:
                rate = sum(self.window) / n
                if rate < self.target and not self.alert:
                    log.warning("audit agreement %.3f below target %.2f: routing to teacher", rate, self.target)
                    self.alert = True
                elif rate >= self.target and self.alert:
                    log.warning("audit agreement recovered to %.3f", rate)
                    self.alert = False

    def status(self) -> dict:
        n = len(self.window)
        return {"samples": n, "agreement": (sum(self.window) / n) if n else None,
                "target": self.target, "alert": self.alert, "rate": self.rate}


# ---------------------------------------------------------------------------
# Inference


@dataclass
class Answer:
    question: Question
    probs: np.ndarray
    auto: bool
    route: str

    def to_dict(self) -> dict:
        q = self.question
        if q.type == "bool":
            p = float(self.probs[1])
            return {"value": p >= 0.5, "prob": round(p, 4), "auto": self.auto, "route": self.route}
        names = q.option_names
        i = int(self.probs.argmax())
        return {"value": names[i], "probs": {n: round(float(v), 4) for n, v in zip(names, self.probs)},
                "auto": self.auto, "route": self.route}

    def top3(self) -> list[dict]:
        q = self.question
        names = ["no", "yes"] if q.type == "bool" else q.option_names
        order = np.argsort(-self.probs)[:3]
        return [{"value": names[i], "prob": round(float(self.probs[i]), 4)} for i in order]


ROUTE_RANK = {"correction": 0, "rules": 1, "fast_path": 2, "general": 2, "ensemble": 2, "teacher": 3}
RESULT_ROUTE = {"correction": "rules", "rules": "rules", "fast_path": "student", "general": "student",
                "ensemble": "student", "teacher": "teacher"}


class Engine:
    def __init__(self, bundle: StudentBundle, teacher: Teacher | None = None, audit: AuditMonitor | None = None,
                 gaz: Gazetteer | None = None):
        self.bundle = bundle
        self.teacher = teacher
        self.audit = audit or AuditMonitor()
        self.gaz = gaz or Gazetteer.default()

    def generic_model(self, q: Question) -> QuestionModel:
        # A question identical to a built-in reuses its certified model.
        for qm in self.bundle.builtin.values():
            if qm.question.signature() == q.signature():
                return QuestionModel(q, qm.general_T, qm.fast, qm.rule_map, qm.rule_conf, qm.certificate, "builtin")
        return QuestionModel(q, self.bundle.generic_T.get(q.type, 1.0), None, None, 0.9,
                             self.bundle.generic_cert.get(q.type), "generic")

    def categorize(self, transactions: list[dict], questions: list[Question],
                   taxonomy: TaxonomyModel | None = None, correction_lookup=None) -> list[dict]:
        """``correction_lookup(keys: list[str]) -> dict[key, value]`` returns reviewer labels."""
        preps = [prepare(t["text"], t.get("amount"), t.get("direction"), t.get("history"), self.gaz)
                 for t in transactions]
        X = self.bundle.embedder.encode([p.rendered for p in preps])
        ood = self.bundle.ood.is_ood(X)
        line_keys = [hash_id(t.get("id") or "", p.scrubbed) for t, p in zip(transactions, preps)]
        audited = np.array([self.audit.selected(k) for k in line_keys])
        degraded = self.audit.alert
        tax_key = taxonomy.id if taxonomy else None

        answers: list[dict[str, Answer]] = [dict() for _ in preps]
        for q in questions:
            qm = (taxonomy.questions.get(q.name) if taxonomy else None) or self.generic_model(q)
            q = qm.question
            tiers = tier_probs(self.bundle, qm, preps, X)
            thr = qm.certificate.threshold if qm.certificate else None
            tier, pred, conf, auto = cascade_pick(tiers, thr)

            corr = {}
            ckeys = [correction_key(tax_key, q, p) for p in preps]
            if correction_lookup is not None:
                corr = correction_lookup([k for k in ckeys if k]) or {}

            need_teacher = np.zeros(len(preps), dtype=bool)
            for i, p in enumerate(preps):
                ck = ckeys[i]
                if ck and ck in corr:
                    probs = np.full(qm.n_labels, 0.01 / max(1, qm.n_labels - 1))
                    probs[int(labels_for(q, [corr[ck]])[0])] = 0.99
                    answers[i][q.name] = Answer(q, probs, True, "correction")
                    continue
                probs = tiers[tier[i]][i]
                a = Answer(q, probs, bool(auto[i]) and not ood[i] and not degraded, TIER_NAMES[tier[i]])
                answers[i][q.name] = a
                if self.teacher is not None and (ood[i] or audited[i] or degraded or not a.auto):
                    need_teacher[i] = True

            if need_teacher.any():
                idx = np.nonzero(need_teacher)[0]
                items = [TeacherInput(preps[i].scrubbed, preps[i].direction, preps[i].amount,
                                      preps[i].extracted.channel, preps[i].extracted.merchant,
                                      transactions[i].get("history")) for i in idx]
                try:
                    tres = self.teacher.answer(items, [q])
                except Exception as e:  # teacher outage: keep student answers, never fail the call
                    log.warning("teacher unavailable: %s", type(e).__name__)
                    tres = None
                if tres is not None:
                    for i, res in zip(idx, tres):
                        tp = np.asarray(res[q.name], dtype=float)
                        student = answers[i][q.name]
                        if audited[i]:
                            self.audit.record(int(tp.argmax()) == int(student.probs.argmax()))
                        answers[i][q.name] = Answer(q, tp, bool(qm.certificate and qm.certificate.auto(float(tp.max()))),
                                                    "teacher")

        results = []
        for t, p, a, o in zip(transactions, preps, answers, ood):
            ex = p.extracted
            route = max((x.route for x in a.values()), key=lambda r: ROUTE_RANK[r], default="rules")
            results.append({
                "id": t.get("id"),
                "extracted": {
                    "amount": ex.amount, "balance": ex.balance, "date": ex.date,
                    "account_suffix": ex.account_suffix, "channel": ex.channel,
                    "direction": p.direction, "counterparty_type": ex.counterparty_type, "merchant": ex.merchant,
                },
                "answers": {k: v.to_dict() for k, v in a.items()},
                "route": RESULT_ROUTE[route],
                "ood": bool(o),
                "_answers": a,
                "_prep": p,
            })
        return results


def correction_key(tax_key: str | None, q: Question, p: Prepared) -> str | None:
    """Corrections generalise to the same counterparty in the same direction."""
    cp = _cp_key(p.extracted)
    if cp is None:
        return None
    scope = tax_key or hash_id(repr(q.signature()))
    return hash_id(scope, q.name, p.direction or "", cp.lower())
