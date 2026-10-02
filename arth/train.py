"""Training pipeline: data pool -> scrub -> (teacher labels) -> students -> calibration -> bundle.

    python -m arth.train --pool-size 20000 --embedder hash --seed 0 --models-dir models

Offline (no HF_TOKEN) the pool's concept labels come from the generator, softened by
label smoothing. With ``--teacher hf`` the open-licensed teacher labels every pool
line under the 32-concept taxonomy once; all training views derive from those soft
concept distributions. Gold labels are used only for calibration and certification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time

import numpy as np

from .calibration import fit_temperature
from .certify import Certificate, certify_cascade
from .concepts import BUILTIN_BOOLS, BUILTIN_TAXONOMIES, CONCEPT_LIST, CONCEPTS
from .embed import get_embedder
from .engine import QuestionModel, StudentBundle, calibrate_and_certify, labels_for, learn_rule_map
from .gold import DEFAULT_GOLD, load_jsonl, split_gold
from .ood import KnnOOD
from .prepare import prepare_rows
from .questions import Question
from .students.fast_path import FastPath
from .students.general import GeneralStudent, View, random_views
from .synth import generate
from .teacher import Teacher, TeacherInput


def builtin_views() -> tuple[list[View], list[View]]:
    """(training views, held-out views)."""
    train, held = [], []
    for b in BUILTIN_TAXONOMIES.values():
        (held if b.heldout else train).append(View(Question.from_builtin(b), b.mapping))
    for b in BUILTIN_BOOLS.values():
        (held if b.heldout else train).append(View(Question.from_builtin(b), positives=b.positives))
    return train, held


def concept_question() -> Question:
    return Question.choice("concept", CONCEPTS)


def smoothed_concepts(concepts: list[str], eps: float = 0.05) -> np.ndarray:
    k = len(CONCEPT_LIST)
    out = np.full((len(concepts), k), eps / k, dtype=np.float32)
    for i, c in enumerate(concepts):
        out[i, CONCEPT_LIST.index(c)] += 1 - eps
    return out


def teacher_concepts(teacher: Teacher, preps, rows) -> np.ndarray:
    q = concept_question()
    items = [TeacherInput(p.scrubbed, p.direction, p.amount, p.extracted.channel, p.extracted.merchant,
                          r.get("history")) for p, r in zip(preps, rows)]
    res = teacher.answer(items, [q])
    return np.stack([r["concept"] for r in res]).astype(np.float32)


def snapshot_hash(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r['text']}\x1f{r['concept']}\n".encode())
    return h.hexdigest()[:16]


def view_targets(view: View, concept_probs: np.ndarray) -> np.ndarray:
    t = view.target(concept_probs)
    return np.stack([1 - t, t], axis=1) if view.question.type == "bool" else t


def build_bundle(pool_rows: list[dict], cal_rows: list[dict], version: str = "v1", embedder_spec: str = "hash",
                 seed: int = 0, epochs: int = 6, rank: int = 96, n_random_views: int = 60,
                 teacher: Teacher | None = None, target: float = 0.98, alpha: float = 0.05,
                 verbose: bool = False) -> StudentBundle:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    emb = get_embedder(embedder_spec)
    preps = prepare_rows(pool_rows)
    X = emb.encode([p.rendered for p in preps])
    if teacher is not None:
        cprobs = teacher_concepts(teacher, preps, pool_rows)
        teacher_version = teacher.version
    else:
        cprobs = smoothed_concepts([r["concept"] for r in pool_rows])
        teacher_version = "none:generator-labels+smoothing"
    perm = rng.permutation(len(pool_rows))
    n_hold = max(200, len(pool_rows) // 10)
    hold, tr = perm[:n_hold], perm[n_hold:]
    X_tr, X_hold = X[tr], X[hold]
    c_tr = cprobs[tr]
    concepts_tr = [pool_rows[i]["concept"] for i in tr]
    if verbose:
        print(f"pool {len(pool_rows)} lines embedded with {emb.name} in {time.time() - t0:.1f}s")

    train_views, held_views = builtin_views()
    views = train_views + random_views(n_random_views, seed=seed + 1)
    general = GeneralStudent(emb.dim, rank=rank, seed=seed).fit(X_tr, c_tr, views, emb.encode, epochs=epochs,
                                                                 verbose=verbose)

    ref_idx = rng.choice(len(tr), size=min(6000, len(tr)), replace=False)
    ood = KnnOOD().fit(X_tr[ref_idx], X_hold)

    pool_idx = rng.choice(len(tr), size=min(6000, len(tr)), replace=False)
    bundle = StudentBundle(
        version=version, embedder_spec=embedder_spec, general=general, builtin={}, generic_T={},
        generic_cert={}, ood=ood,
        pool_rendered=[preps[tr[i]].rendered for i in pool_idx],
        pool_concepts=[concepts_tr[i] for i in pool_idx],
    )

    cal_preps = prepare_rows(cal_rows)
    X_cal = emb.encode([p.rendered for p in cal_preps]) if cal_rows else None
    cal_concepts = [r["concept"] for r in cal_rows]

    for v in train_views + held_views:
        q = v.question
        qm = QuestionModel(q, source="builtin")
        if v in train_views:
            qm.fast = FastPath(qm.n_labels, seed=seed).fit(X_tr, view_targets(v, c_tr))
            qm.rule_map = ({c: q.option_names.index(o) for c, o in v.mapping.items()} if q.type == "choice"
                           else {c: int(c in v.positives) for c in CONCEPT_LIST})
        else:
            qm.rule_map = learn_rule_map(bundle, qm, emb.encode(bundle.pool_rendered))
        if X_cal is not None:
            y = labels_for(q, [v.mapping[c] if q.type == "choice" else c in v.positives for c in cal_concepts])
            calibrate_and_certify(bundle, qm, cal_preps, X_cal, y, target, alpha)
        bundle.builtin[q.name] = qm
        if verbose:
            c = qm.certificate
            print(f"  {q.name:22s} T_gen={qm.general_T:.2f} cert_t={c.threshold if c else None} "
                  f"coverage={c.coverage if c else 0:.2f}")

    # Generic temperature and certificate for taxonomies we have never seen, measured on
    # the held-out built-ins with the general student alone.
    for qtype in ("choice", "bool"):
        hv = [v for v in held_views if v.question.type == qtype]
        if X_cal is None or not hv:
            bundle.generic_T[qtype] = 1.0
            bundle.generic_cert[qtype] = Certificate(None, target, alpha, 0, 0, None, None, 0.0)
            continue
        logits, ys = [], []
        for v in hv:
            q = v.question
            L = general.logits(X_cal, q, emb.encode)
            y = labels_for(q, [v.mapping[c] if q.type == "choice" else c in v.positives for c in cal_concepts])
            logits.append(L)
            ys.append(y)
        k = max(L.shape[1] for L in logits)
        Lp = np.vstack([np.pad(L, ((0, 0), (0, k - L.shape[1])), constant_values=-1e9) for L in logits])
        yp = np.concatenate(ys)
        T = fit_temperature(Lp, yp)
        bundle.generic_T[qtype] = T
        from .calibration import softmax

        bundle.generic_cert[qtype] = certify_cascade([softmax(Lp, T)], yp, target, alpha)

    bundle.manifest = {
        "model_version": f"student:{version}",
        "embedder": embedder_spec,
        "seed": seed,
        "epochs": epochs,
        "rank": rank,
        "random_views": n_random_views,
        "pool_size": len(pool_rows),
        "data_snapshot": snapshot_hash(pool_rows),
        "calibration_snapshot": snapshot_hash(cal_rows) if cal_rows else None,
        "teacher_version": teacher_version,
        "train_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(),
    }
    return bundle


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-size", type=int, default=20000)
    ap.add_argument("--pool-file", help="JSONL pool with text, amount, direction, history, concept")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--embedder", default="hash", help="hash | bge-small | hash+bge-small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--teacher", choices=["none", "hf"], default="none")
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--eval", action="store_true", help="run the evaluation gates and attach the report")
    args = ap.parse_args(argv)

    from .registry import Registry

    pool = load_jsonl(args.pool_file) if args.pool_file else generate(args.pool_size, seed=args.seed)
    cal, _ = split_gold(load_jsonl(args.gold))
    teacher = None
    if args.teacher == "hf":
        from .teacher import HFTeacher

        teacher = HFTeacher()
    reg = Registry(args.models_dir)
    version = reg.next_version()
    bundle = build_bundle(pool, cal, version, args.embedder, args.seed, args.epochs, teacher=teacher, verbose=True)
    reg.save(bundle, promote=not args.no_promote)
    print(f"saved student:{version} -> {args.models_dir}/{version}")
    if args.eval:
        from .evals.run import evaluate_bundle

        report = evaluate_bundle(bundle, gold_path=args.gold)
        reg.attach_report(version, report)
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
