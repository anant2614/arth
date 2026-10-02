"""Evals as tests: behaviour cases plus quality floors on the gold eval split.

The floors are set below what the small CI bundle (8k synthetic lines, hashing
embedder) reaches, so they catch regressions without flaking. The full launch gates
run with ``python -m arth.evals.run`` and are reported, not asserted, because several
of them are meant to fail until real gold data and the teacher exist.
"""

import json

import pytest

from arth.evals.behaviour import CASES, run_cases
from arth.evals.run import GATES, evaluate_bundle, stability, to_markdown


@pytest.fixture(scope="module")
def report(bundle):
    return evaluate_bundle(bundle, measure_latency=True, synth_n=1500)


@pytest.fixture(scope="module")
def behaviour(bundle):
    return run_cases(bundle)


@pytest.mark.parametrize("case", [c.name for c in CASES])
def test_behaviour_case(behaviour, case):
    r = next(r for r in behaviour if r["case"] == case)
    assert r["ok"], r


def test_report_has_every_gate(report):
    assert set(report["gates"]) == set(GATES) - {"stability"}
    assert all(g["status"] in {"pass", "fail", "skipped"} for g in report["gates"].values())
    skipped = {k for k, g in report["gates"].items() if g["status"] == "skipped"}
    assert skipped <= {"per_slice", "device"}  # only gates that need a teacher or a phone
    json.dumps(report, default=str)  # serialisable


def test_student_beats_baselines(report):
    assert report["gates"]["accuracy"]["status"] == "pass"
    base = report["baselines"]
    assert report["summary"]["macro_f1"] > base["regex_rules"]["macro_f1"] + 0.03


def test_quality_floors(report):
    q = report["questions"]["personal_finance"]
    assert q["cascade"]["macro_f1"] >= 0.72
    assert q["ece"] <= 0.10
    for name in ("lender_flags", "coarse"):
        assert report["questions"][name]["cascade"]["accuracy"] >= 0.78
    for name in ("is_salary", "is_emi", "is_refund", "is_cash", "is_tax", "is_investment", "is_bounce"):
        assert report["questions"][name]["cascade"]["accuracy"] >= 0.95, name


def test_certified_answers_are_precise_on_unseen_banks(report):
    """Where a yes/no question certified a threshold on the calibration banks, the
    auto-labelled answers on the (different) eval banks should stay near 98%."""
    checked = 0
    for name, q in report["questions"].items():
        if q["type"] == "bool" and q["certificate"]["threshold"] is not None and q["eval_n_auto"] >= 50:
            assert q["eval_auto_precision"] >= 0.95, (name, q["eval_auto_precision"])
            checked += 1
    assert checked >= 4


def test_heldout_taxonomy_transfers(report):
    g = report["gates"]["generalization"]
    assert g["heldout_accuracy"] >= 0.40  # zero-shot on GST ledger heads it never saw (gate wants ~in-taxonomy)
    assert report["questions"]["is_rent"]["general_only"]["accuracy"] >= 0.9


def test_latency_budget(report):
    lat = report["gates"]["latency"]
    assert lat["p95_ms"] < 50 and lat["batch_1000_s"] < 10


def test_slices_present(report):
    assert {"bank", "channel", "merchant_known", "p2p_vs_merchant"} <= set(report["slices"])


def test_markdown_and_stability(report):
    full = {"seeds": [report, report], "stability": stability([report, report])}
    assert full["stability"]["status"] == "pass"
    md = to_markdown(full)
    assert "| accuracy |" in md and "regex_rules" in md
    assert stability([report])["status"] == "skipped"
