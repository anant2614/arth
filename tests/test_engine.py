import numpy as np
import pytest

from arth.certify import Certificate
from arth.engine import (AuditMonitor, Engine, QuestionModel, builtin_taxonomy, correction_key,
                         fit_saved_taxonomy)
from arth.prepare import prepare
from arth.questions import Question
from arth.teacher import OracleTeacher

PF = "personal_finance"


def tx(text, amount=None, direction=None, history=None, id=None):
    return {"id": id, "text": text, "amount": amount, "direction": direction, "history": history}


def test_prd_examples_end_to_end(bundle):
    eng = Engine(bundle)
    tax = builtin_taxonomy(bundle, PF)
    q = tax.questions[PF].question
    res = eng.categorize([
        tx("Rs.450 debited from A/c XX1234 to VPA zomatoonline@paytm on 01-10-26"),
        tx("UPI/DR/402155/SWIGGY/YESB/swiggy@ybl/Pay", 320, "debit"),
        tx("NACH DR BSE STAR MF 0012345", 5000, "debit"),
        tx("IMPS/P2A/402166/RAMESH K/rent oct", 18000, "debit",
           {"same_counterparty_count": 6, "amount_cv": 0.02, "median_gap_days": 30}),
    ], [q], tax)
    values = [r["answers"][PF]["value"] for r in res]
    assert values == ["food_dining", "food_dining", "investments", "rent"]
    assert res[0]["extracted"]["amount"] == 450.0 and res[0]["extracted"]["merchant"] == "Zomato"
    for r in res:
        probs = r["answers"][PF]["probs"]
        assert set(probs) == set(q.option_names)
        assert abs(sum(probs.values()) - 1) < 1e-3
        assert r["route"] in {"rules", "student", "teacher"}


def test_rules_tier_answers_known_merchants(bundle):
    eng = Engine(bundle)
    tax = builtin_taxonomy(bundle, PF)
    qm = tax.questions[PF]
    assert qm.rule_map, "built-in taxonomies ship a gazetteer mapping"
    # force a permissive certificate so the cheapest tier wins
    qm2 = QuestionModel(qm.question, qm.general_T, qm.fast, qm.rule_map, 0.999,
                        Certificate(0.95, 0.98, 0.05, 500, 400, 1.0, 0.99, 0.8), "builtin")
    tax.questions[PF] = qm2
    r = eng.categorize([tx("UPI/DR/402155/SWIGGY/YESB/swiggy@ybl/Pay", 320, "debit")], [qm.question], tax)[0]
    a = r["answers"][PF]
    assert a["route"] == "rules" and a["auto"] and r["route"] == "rules"
    # a credit from the same merchant is not a purchase: rules must stand aside
    r = eng.categorize([tx("UPI/CR/402155/SWIGGY/YESB/swiggy@ybl/Refund", 320, "credit")], [qm.question], tax)[0]
    assert r["answers"][PF]["route"] != "rules"


def test_bool_answer_shape(bundle):
    eng = Engine(bundle)
    q = Question.boolean("is_salary", "Salary credited by an employer")
    r = eng.categorize([tx("NEFT CR-HDFC0000001-ACME TECHNOLOGIES PVT LTD-SALARY SEP", 85000, "credit")], [q])[0]
    a = r["answers"]["is_salary"]
    assert a["value"] is True and 0.5 <= a["prob"] <= 1 and isinstance(a["auto"], bool)


def test_generic_questions_use_generic_certificate(bundle):
    eng = Engine(bundle)
    q = Question.choice("k", {"a": "Food", "b": "Travel"})
    qm = eng.generic_model(q)
    assert qm.source == "generic" and qm.certificate is bundle.generic_cert["choice"]
    same = eng.generic_model(bundle.builtin[PF].question)
    assert same.source == "builtin"


def test_ood_lines_are_never_auto(bundle):
    eng = Engine(bundle)
    tax = builtin_taxonomy(bundle, PF)
    q = tax.questions[PF].question
    res = eng.categorize([tx("zxqv wplk mmmrr ttt 0x0x0x qqqq jjjj"), tx("UPI/DR/402155/SWIGGY/YESB/swiggy@ybl/Pay")],
                         [q], tax)
    assert res[0]["ood"] is True and res[0]["answers"][PF]["auto"] is False
    assert res[1]["ood"] is False


def _oracle(bundle, rows, accuracy=1.0):
    return OracleTeacher({prepare(r["text"], r.get("amount"), r.get("direction")).scrubbed: r["concept"]
                          for r in rows}, accuracy=accuracy)


def test_teacher_fallback_for_unsure_and_ood(bundle, gold_split):
    _, ev = gold_split
    rows = ev[:60]
    teacher = _oracle(bundle, rows)
    eng = Engine(bundle, teacher=teacher)
    tax = builtin_taxonomy(bundle, PF)
    q = tax.questions[PF].question
    res = eng.categorize([tx(r["text"], r["amount"], r["direction"], r.get("history")) for r in rows], [q], tax)
    routes = [r["answers"][PF]["route"] for r in res]
    assert "teacher" in routes and teacher.calls > 0
    assert all(r["route"] == "teacher" for r in res if r["ood"])
    from arth.concepts import PERSONAL_FINANCE

    acc = np.mean([r["answers"][PF]["value"] == PERSONAL_FINANCE.label(g["concept"]) for r, g in zip(res, rows)])
    assert acc > 0.9  # the oracle teacher fixes what the students were unsure about


def test_teacher_outage_never_fails_the_call(bundle):
    class Down:
        version = "teacher:down"

        def answer(self, items, questions):
            raise ConnectionError("provider down")

    eng = Engine(bundle, teacher=Down())
    r = eng.categorize([tx("zxqv wplk mmmrr ttt 0x0x0x")], [bundle.builtin[PF].question])[0]
    assert r["answers"][PF]["auto"] is False and r["answers"][PF]["route"] != "teacher"


def test_audit_slice_is_about_two_percent():
    mon = AuditMonitor(rate=0.02)
    picked = sum(mon.selected(f"line-{i}") for i in range(20000))
    assert 300 < picked < 500


def test_audit_alert_and_recovery():
    mon = AuditMonitor(target=0.9, window=100, min_samples=50)
    for _ in range(60):
        mon.record(False)
    assert mon.alert
    for _ in range(100):
        mon.record(True)
    assert not mon.alert


def test_degraded_mode_sends_everything_to_teacher(bundle, gold_split):
    _, ev = gold_split
    rows = ev[:20]
    mon = AuditMonitor()
    mon.alert = True
    teacher = _oracle(bundle, rows)
    eng = Engine(bundle, teacher=teacher, audit=mon)
    res = eng.categorize([tx(r["text"], r["amount"], r["direction"]) for r in rows], [bundle.builtin[PF].question])
    assert all(r["answers"][PF]["route"] == "teacher" for r in res)


def test_audited_lines_record_agreement(bundle, gold_split):
    _, ev = gold_split
    mon = AuditMonitor(rate=1.0)  # audit everything
    teacher = _oracle(bundle, ev, accuracy=0.0)  # always disagrees
    eng = Engine(bundle, teacher=teacher, audit=mon)
    eng.categorize([tx(r["text"], r["amount"], r["direction"]) for r in ev[:60]], [bundle.builtin[PF].question])
    assert mon.status()["samples"] == 60 and mon.alert


def test_corrections_override_and_generalise_to_same_counterparty(bundle):
    eng = Engine(bundle)
    tax = builtin_taxonomy(bundle, PF)
    q = tax.questions[PF].question
    first = tx("IMPS/P2A/402166/RAMESH K/oct", 18000, "debit")
    p = prepare(first["text"], 18000, "debit")
    key = correction_key(tax.id, q, p)
    store = {key: "rent"}
    nxt = tx("IMPS/P2A/509999/RAMESH K/nov", 18000, "debit")  # same counterparty, next month
    r = eng.categorize([nxt], [q], tax, correction_lookup=lambda keys: {k: store[k] for k in keys if k in store})[0]
    a = r["answers"][PF]
    assert a["value"] == "rent" and a["auto"] and a["route"] == "correction"


def test_saved_taxonomy_fit_and_certify(bundle, gold_split):
    cal, _ = gold_split
    from arth.concepts import LENDER_FLAGS

    q = Question.choice("flag", LENDER_FLAGS.options)
    b = Question.boolean("is_salary", "Salary credited by an employer")
    labels = {"flag": [LENDER_FLAGS.label(r["concept"]) for r in cal], "is_salary": [r["concept"] == "salary" for r in cal]}
    tm = fit_saved_taxonomy(bundle, "tax_test", [q, b], cal, labels)
    assert set(tm.questions) == {"flag", "is_salary"}
    for qm in tm.questions.values():
        assert qm.fast is not None and qm.certificate is not None and qm.source == "saved"
        assert qm.certificate.n_calibration == len(cal)
    assert tm.questions["is_salary"].certificate.threshold is not None  # easy yes/no question certifies
    eng = Engine(bundle)
    r = eng.categorize([tx("NEFT CR-HDFC0000001-ACME TECHNOLOGIES PVT LTD-SALARY SEP", 85000, "credit")],
                       [q, b], tm)[0]
    assert r["answers"]["flag"]["value"] == "income" and r["answers"]["is_salary"]["value"] is True


def test_saved_taxonomy_without_labels_is_never_auto(bundle):
    q = Question.choice("x", {"food": "Food and dining", "other": "Everything else"})
    tm = fit_saved_taxonomy(bundle, "tax_nolabels", [q], [], {})
    assert tm.questions["x"].certificate.threshold is None
    r = Engine(bundle).categorize([tx("UPI/DR/402155/SWIGGY/YESB/swiggy@ybl/Pay", 320, "debit")], [q], tm)[0]
    assert r["answers"]["x"]["auto"] is False


@pytest.mark.parametrize("n", [1, 1000])
def test_batch_sizes(bundle, n):
    rows = [tx("UPI/DR/402155/SWIGGY/YESB/swiggy@ybl/Pay", 320, "debit", id=str(i)) for i in range(n)]
    res = Engine(bundle).categorize(rows, [bundle.builtin[PF].question])
    assert [r["id"] for r in res] == [str(i) for i in range(n)]


def test_corrections_retrain_the_fast_path(bundle):
    """FR-8: reviewer labels flow into the taxonomy's fast path on refit."""
    q = Question.choice("kind", {"household": "Household and personal spending", "business": "Business expense"})
    line = prepare("UPI/DR/412345678901/ZEPTO/YESB/zeptonow@ybl/office snacks", 900, "debit")
    before = fit_saved_taxonomy(bundle, "tax_c", [q], [], {})
    p0 = before.questions["kind"].fast.predict_proba(bundle.embedder.encode([line.rendered]))[0]
    corr = {"kind": [(line.rendered, "business" if p0.argmax() == 0 else "household")]}
    after = fit_saved_taxonomy(bundle, "tax_c", [q], [], {}, corrections=corr)
    p1 = after.questions["kind"].fast.predict_proba(bundle.embedder.encode([line.rendered]))[0]
    assert p1.argmax() != p0.argmax()
