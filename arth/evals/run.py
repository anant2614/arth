"""Evaluation harness and quality gates (PRD "Evaluation and quality gates").

    python -m arth.evals.run --seeds 0 1 2 --embedder hash --out runs/eval

Trains one bundle per seed on the synthetic pool, scores it against the gold eval
split, and checks every gate. Gates that need components not available in this
environment (teacher LLM without HF_TOKEN, on-device ONNX export) are reported as
``skipped`` with the reason, never silently passed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..calibration import ece, reliability
from ..certify import cascade_pick, coverage_at
from ..concepts import BUILTIN_BOOLS, BUILTIN_TAXONOMIES
from ..engine import TIER_NAMES, Engine, StudentBundle, labels_for, tier_probs
from ..gold import DEFAULT_GOLD, load_jsonl, split_gold
from ..prepare import prepare_rows
from ..questions import Question
from ..synth import generate
from ..teacher import Teacher, TeacherInput
from .baselines import embedding_zero_shot, regex_concept
from .metrics import accuracy, macro_f1, per_class_f1

PRIMARY = "personal_finance"
HELDOUT_CHOICE = "gst_ledger"
GATES = {
    "accuracy": "Macro-F1 beats every baseline on the gold eval split",
    "per_slice": "Every slice with 30+ items within 2 points of the teacher",
    "certified_coverage": "70%+ auto-labelled at 98% precision, 95% confidence",
    "calibration": "Expected calibration error <= 0.05",
    "generalization": "Held-out taxonomy accuracy within 5 points of in-taxonomy accuracy",
    "synthetic_gap": "Synthetic test accuracy minus gold accuracy under 10 points",
    "latency": "p95 < 50 ms per line on CPU; batch of 1,000 < 10 s",
    "device": "ONNX INT8 <= 40 MB and < 30 ms on a budget Android phone",
    "stability": "No gate flips between seeds",
}


def _label_of(name: str, concept: str):
    if name in BUILTIN_TAXONOMIES:
        return BUILTIN_TAXONOMIES[name].label(concept)
    return BUILTIN_BOOLS[name].label(concept)


def score_question(bundle: StudentBundle, name: str, cal, ev, preps_ev, X_ev) -> dict:
    qm = bundle.builtin[name]
    q = qm.question
    y = labels_for(q, [_label_of(name, r["concept"]) for r in ev])
    tiers = tier_probs(bundle, qm, preps_ev, X_ev)
    thr = qm.certificate.threshold if qm.certificate else None
    tier, pred, conf, auto = cascade_pick(tiers, thr)
    correct = pred == y
    labels = list(range(qm.n_labels))
    names = ["no", "yes"] if q.type == "bool" else q.option_names
    cov = coverage_at(np.where(auto, 1.0, 0.0), correct, 0.5) if thr is not None else coverage_at(conf, correct, None)
    general = tiers[2].argmax(axis=1)
    fast = tiers[1].argmax(axis=1) if tiers[1] is not None else None
    out = {
        "question": name,
        "type": q.type,
        "n_eval": len(y),
        "cascade": {"accuracy": accuracy(y, pred), "macro_f1": macro_f1(y, pred, labels)},
        "general_only": {"accuracy": accuracy(y, general), "macro_f1": macro_f1(y, general, labels)},
        "fast_path_only": ({"accuracy": accuracy(y, fast), "macro_f1": macro_f1(y, fast, labels)}
                           if fast is not None else None),
        "ece": ece(conf, correct),
        "certificate": qm.certificate.to_dict() if qm.certificate else None,
        "eval_coverage": cov["coverage"],
        "eval_auto_precision": cov["precision"],
        "eval_n_auto": cov["n_auto"],
        "tier_share": {t: float((tier == i).mean()) for i, t in enumerate(TIER_NAMES)},
        "per_class": {names[int(k)]: v for k, v in per_class_f1(y, pred, labels).items() if v["support"]},
        "reliability": reliability(conf, correct),
        "_pred": pred,
        "_y": y,
        "_conf": conf,
    }
    return out


def slices(ev: list[dict], preps, y, pred, min_items: int = 30) -> dict:
    def group(key_fn):
        g: dict[str, list[int]] = {}
        for i, (r, p) in enumerate(zip(ev, preps)):
            g.setdefault(key_fn(r, p), []).append(i)
        return {k: {"n": len(ix), "macro_f1": macro_f1(y[ix], pred[ix]), "accuracy": accuracy(y[ix], pred[ix])}
                for k, ix in sorted(g.items())}

    p2p = {"p2p_out", "p2p_in", "rent", "self_transfer"}
    out = {
        "bank": group(lambda r, p: r.get("bank", "?")),
        "channel": group(lambda r, p: p.extracted.channel),
        "merchant_known": group(lambda r, p: "known" if p.extracted.merchant else "unseen"),
        "p2p_vs_merchant": group(lambda r, p: "p2p" if r["concept"] in p2p else "merchant_or_other"),
    }
    for dim in out.values():
        for v in dim.values():
            v["gated"] = v["n"] >= min_items
    return out


def latency(bundle: StudentBundle, rows: list[dict], n_single: int = 200) -> dict:
    eng = Engine(bundle)
    q = bundle.builtin[PRIMARY].question
    tax = None
    eng.categorize([rows[0]], [q])  # warm-up
    times = []
    for i in range(n_single):
        r = rows[i % len(rows)]
        t = time.perf_counter()
        eng.categorize([r], [q], tax)
        times.append((time.perf_counter() - t) * 1000)
    batch = [rows[i % len(rows)] for i in range(1000)]
    t = time.perf_counter()
    eng.categorize(batch, [q], tax)
    batch_s = time.perf_counter() - t
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
            "batch_1000_s": batch_s, "embedder": bundle.embedder_spec}


def teacher_eval(teacher: Teacher, ev, preps, names: list[str]) -> dict[str, np.ndarray]:
    qs = [bundle_q for bundle_q in (Question.from_builtin(BUILTIN_TAXONOMIES[n]) for n in names)]
    items = [TeacherInput(p.scrubbed, p.direction, p.amount, p.extracted.channel, p.extracted.merchant, r.get("history"))
             for p, r in zip(preps, ev)]
    res = teacher.answer(items, qs)
    return {q.name: np.array([r[q.name].argmax() for r in res]) for q in qs}


def evaluate_bundle(bundle: StudentBundle, gold_path: str | Path = DEFAULT_GOLD, teacher: Teacher | None = None,
                    synth_seed: int = 12345, synth_n: int = 3000, measure_latency: bool = True) -> dict:
    gold = load_jsonl(gold_path)
    cal, ev = split_gold(gold)
    emb = bundle.embedder
    preps_ev = prepare_rows(ev)
    X_ev = emb.encode([p.rendered for p in preps_ev])

    per_q = {name: score_question(bundle, name, cal, ev, preps_ev, X_ev) for name in bundle.builtin}
    prim = per_q[PRIMARY]
    y, pred = prim["_y"], prim["_pred"]
    q_prim = bundle.builtin[PRIMARY].question

    # Baselines on the primary taxonomy.
    tax = BUILTIN_TAXONOMIES[PRIMARY]
    regex_pred = labels_for(q_prim, [tax.label(regex_concept(r["text"], p)) for r, p in zip(ev, preps_ev)])
    zs = embedding_zero_shot(X_ev, emb.encode(q_prim.option_texts())).argmax(axis=1)
    baselines = {
        "regex_rules": {"macro_f1": macro_f1(y, regex_pred), "accuracy": accuracy(y, regex_pred)},
        "embedding_zero_shot": {"macro_f1": macro_f1(y, zs), "accuracy": accuracy(y, zs)},
        "llm_zero_shot": None,
        "banking_sms_json_parser_v8": None,
    }
    teacher_preds = None
    if teacher is not None:
        teacher_preds = teacher_eval(teacher, ev, preps_ev, [PRIMARY])[PRIMARY]
        baselines["llm_zero_shot"] = {"macro_f1": macro_f1(y, teacher_preds), "accuracy": accuracy(y, teacher_preds)}

    sl = slices(ev, preps_ev, y, pred)
    teacher_slices = slices(ev, preps_ev, y, teacher_preds) if teacher_preds is not None else None

    # Synthetic gap: same student, fresh synthetic lines it has never seen.
    syn = generate(synth_n, seed=synth_seed)
    preps_syn = prepare_rows(syn)
    X_syn = emb.encode([p.rendered for p in preps_syn])
    qm = bundle.builtin[PRIMARY]
    y_syn = labels_for(q_prim, [tax.label(r["concept"]) for r in syn])
    _, pred_syn, _, _ = cascade_pick(tier_probs(bundle, qm, preps_syn, X_syn), qm.certificate.threshold)
    syn_acc = accuracy(y_syn, pred_syn)

    held = per_q[HELDOUT_CHOICE]
    gates: dict[str, dict] = {}
    beaten = [k for k, v in baselines.items() if v is not None]
    student_f1 = prim["cascade"]["macro_f1"]
    gates["accuracy"] = {
        "status": "pass" if all(student_f1 > baselines[k]["macro_f1"] for k in beaten) else "fail",
        "student_macro_f1": student_f1,
        "baselines": {k: (v["macro_f1"] if v else None) for k, v in baselines.items()},
        "not_run": [k for k, v in baselines.items() if v is None],
    }
    if teacher_slices is None:
        gates["per_slice"] = {"status": "skipped", "reason": "no teacher configured (set HF_TOKEN)",
                              "worst_gated_slice": _worst(sl)}
    else:
        fails = [(d, k) for d, dim in sl.items() for k, v in dim.items()
                 if v["gated"] and v["macro_f1"] < teacher_slices[d][k]["macro_f1"] - 0.02]
        gates["per_slice"] = {"status": "fail" if fails else "pass", "failing": fails}
    gates["certified_coverage"] = {
        "status": "pass" if prim["eval_coverage"] >= 0.70 and prim["certificate"]["threshold"] is not None else "fail",
        "coverage_eval": prim["eval_coverage"], "precision_eval": prim["eval_auto_precision"],
        "certificate": prim["certificate"],
    }
    gates["calibration"] = {"status": "pass" if prim["ece"] <= 0.05 else "fail", "ece": prim["ece"]}
    gap_g = prim["general_only"]["accuracy"] - held["general_only"]["accuracy"]
    gates["generalization"] = {
        "status": "pass" if gap_g <= 0.05 else "fail",
        "in_taxonomy_accuracy": prim["general_only"]["accuracy"],
        "heldout_taxonomy": HELDOUT_CHOICE,
        "heldout_accuracy": held["general_only"]["accuracy"],
        "gap": gap_g,
    }
    gap_s = syn_acc - prim["cascade"]["accuracy"]
    gates["synthetic_gap"] = {"status": "pass" if gap_s < 0.10 else "fail", "synthetic_accuracy": syn_acc,
                              "gold_accuracy": prim["cascade"]["accuracy"], "gap": gap_s}
    if measure_latency:
        lat = latency(bundle, ev)
        gates["latency"] = {"status": "pass" if lat["p95_ms"] < 50 and lat["batch_1000_s"] < 10 else "fail", **lat}
    else:
        gates["latency"] = {"status": "skipped", "reason": "disabled"}
    gates["device"] = {"status": "skipped", "reason": "ONNX INT8 export and Android SDK are V1.5 (FR-14)"}

    for v in per_q.values():
        for k in ("_pred", "_y", "_conf"):
            v.pop(k)
    return {
        "model_version": f"student:{bundle.version}",
        "manifest": bundle.manifest,
        "gold": {"path": str(gold_path), "n": len(gold), "n_calibration": len(cal), "n_eval": len(ev),
                 "split": "by bank"},
        "primary_taxonomy": PRIMARY,
        "questions": per_q,
        "baselines": baselines,
        "slices": sl,
        "teacher_slices": teacher_slices,
        "gates": gates,
        "summary": {
            "macro_f1": student_f1,
            "accuracy": prim["cascade"]["accuracy"],
            "certified_coverage_eval": prim["eval_coverage"],
            "ece": prim["ece"],
            "gates": {k: v["status"] for k, v in gates.items()},
            "all_gates_passed": all(v["status"] == "pass" for v in gates.values() if v["status"] != "skipped"),
        },
    }


def _worst(sl: dict) -> dict | None:
    worst = None
    for d, dim in sl.items():
        for k, v in dim.items():
            if v["gated"] and (worst is None or v["macro_f1"] < worst["macro_f1"]):
                worst = {"dimension": d, "slice": k, **v}
    return worst


def stability(reports: list[dict]) -> dict:
    names = list(GATES)
    if len(reports) < 2:
        return {"status": "skipped", "reason": "needs 2+ seeds", "seeds": len(reports)}
    flips = [g for g in names if len({r["gates"].get(g, {}).get("status") for r in reports}) > 1]
    return {"status": "fail" if flips else "pass", "flipped": flips, "seeds": len(reports)}


def to_markdown(full: dict) -> str:
    lines = ["# Arth evaluation report", ""]
    r0 = full["seeds"][0]
    g = r0["gold"]
    lines += [f"Gold set: `{g['path']}`, {g['n']} lines ({g['n_calibration']} calibration / {g['n_eval']} eval, "
              f"split {g['split']}). Embedder: `{r0['manifest']['embedder']}`. "
              f"Teacher: `{r0['manifest']['teacher_version']}`.", ""]
    lines += ["## Gates", "", "| Gate | Pass rule | " + " | ".join(f"seed {r['manifest']['seed']}" for r in full["seeds"]) + " |",
              "| --- | --- | " + " | ".join("---" for _ in full["seeds"]) + " |"]
    for k, rule in GATES.items():
        if k == "stability":
            cells = [full["stability"]["status"]] * len(full["seeds"])
        else:
            cells = [r["gates"][k]["status"] for r in full["seeds"]]
        lines.append(f"| {k} | {rule} | " + " | ".join(cells) + " |")
    lines += ["", "## Headline numbers (primary taxonomy: personal_finance)", "",
              "| Seed | Macro-F1 | Accuracy | ECE | Certified coverage (eval) | Auto precision (eval) | Threshold |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in full["seeds"]:
        q = r["questions"][PRIMARY]
        c = q["certificate"] or {}
        prec = q["eval_auto_precision"]
        lines.append(f"| {r['manifest']['seed']} | {q['cascade']['macro_f1']:.3f} | {q['cascade']['accuracy']:.3f} | "
                     f"{q['ece']:.3f} | {q['eval_coverage']:.1%} | {'-' if prec is None else f'{prec:.3f}'} | "
                     f"{c.get('threshold')} |")
    lines += ["", "## Baselines vs student (seed 0, macro-F1 on gold eval)", "", "| Model | Macro-F1 | Accuracy |",
              "| --- | --- | --- |"]
    for k, v in r0["baselines"].items():
        f1 = "not run" if v is None else f"{v['macro_f1']:.3f}"
        acc = "" if v is None else f"{v['accuracy']:.3f}"
        lines.append(f"| {k} | {f1} | {acc} |")
    q = r0["questions"][PRIMARY]
    lines.append(f"| student: fast path only | {q['fast_path_only']['macro_f1']:.3f} | {q['fast_path_only']['accuracy']:.3f} |")
    lines.append(f"| student: general only | {q['general_only']['macro_f1']:.3f} | {q['general_only']['accuracy']:.3f} |")
    lines.append(f"| **student: cascade** | **{q['cascade']['macro_f1']:.3f}** | **{q['cascade']['accuracy']:.3f}** |")
    lines += ["", "## All built-in questions (seed 0)", "",
              "| Question | Type | Cascade acc | Macro-F1 | ECE | Cert. threshold | Coverage (eval) | Auto precision |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, q in r0["questions"].items():
        c = q["certificate"] or {}
        prec = q["eval_auto_precision"]
        lines.append(f"| {name} | {q['type']} | {q['cascade']['accuracy']:.3f} | {q['cascade']['macro_f1']:.3f} | "
                     f"{q['ece']:.3f} | {c.get('threshold')} | {q['eval_coverage']:.1%} | "
                     f"{'-' if prec is None else f'{prec:.3f}'} |")
    gates = r0["gates"]
    lines += ["", "## Gate details (seed 0)", ""]
    for k, v in gates.items():
        lines.append(f"- **{k}**: {v['status']} — " + json.dumps({a: b for a, b in v.items() if a != 'status'},
                                                                   default=str)[:600])
    lines += ["", "## Slices (seed 0, primary taxonomy)", ""]
    for d, dim in r0["slices"].items():
        lines.append(f"**{d}**: " + ", ".join(f"{k} {v['macro_f1']:.2f} (n={v['n']})" for k, v in dim.items()))
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from ..train import build_bundle

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--embedder", default="hash")
    ap.add_argument("--pool-size", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--teacher", choices=["none", "hf"], default="none")
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args(argv)

    teacher = None
    if args.teacher == "hf":
        from ..teacher import HFTeacher

        teacher = HFTeacher()
    cal, _ = split_gold(load_jsonl(args.gold))
    reports = []
    for seed in args.seeds:
        print(f"== seed {seed}")
        pool = generate(args.pool_size, seed=seed)
        bundle = build_bundle(pool, cal, f"eval-s{seed}", args.embedder, seed, args.epochs, verbose=True)
        rep = evaluate_bundle(bundle, args.gold, teacher, measure_latency=not args.no_latency)
        print(json.dumps(rep["summary"], indent=2))
        reports.append(rep)
    full = {"seeds": reports, "stability": stability(reports)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(full, indent=2, default=str))
    (out / "report.md").write_text(to_markdown(full))
    print(f"wrote {out}/report.json and {out}/report.md")


if __name__ == "__main__":
    main()
