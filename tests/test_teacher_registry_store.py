import json
import math
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pytest

from arth.questions import Question
from arth.store import Store
from arth.teacher import HF_ROUTER_URL, HFTeacher, TeacherInput, fit_teacher_temperature

Q = Question.choice("category", {"rent": "House rent", "p2p_transfer": "Transfer to a person", "emi": "Loan EMI"})
B = Question.boolean("is_salary", "Salary credited by an employer")


def _logprob_response(content: str, alts: dict[int, list[tuple[str, float]]]):
    """Fake OpenAI-style response; ``alts`` maps token index -> top_logprobs."""
    toks = []
    i = 0
    # tokenise into the pieces a real tokenizer would roughly produce
    for piece in ['{"', "category", '":"', "A", '","', "is_salary", '":"', "no", '"}']:
        toks.append({"token": piece, "logprob": 0.0,
                     "top_logprobs": [{"token": t, "logprob": lp} for t, lp in alts.get(i, [(piece, 0.0)])]})
        i += 1
    return {"choices": [{"message": {"content": content}, "logprobs": {"content": toks}}]}


def test_teacher_logprobs_mode_and_scrubbing():
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.url == httpx.URL(HF_ROUTER_URL)
        assert request.headers["authorization"] == "Bearer hf_test"
        return httpx.Response(200, json=_logprob_response(
            '{"category":"A","is_salary":"no"}',
            {3: [("A", math.log(0.8)), ("B", math.log(0.15)), ("C", math.log(0.05))],
             7: [("no", math.log(0.9)), ("yes", math.log(0.1))]}))

    t = HFTeacher(token="hf_test", client=httpx.Client(transport=httpx.MockTransport(handler)))
    # Raw PII passed by mistake must be scrubbed before it leaves the process.
    out = t.answer([TeacherInput("IMPS/P2A/402166/RAMESH K/rent oct call 9123456789", "debit", 18000)], [Q, B])[0]
    assert np.allclose(out["category"], [0.8, 0.15, 0.05])
    assert np.allclose(out["is_salary"], [0.9, 0.1])
    sent = json.dumps(seen[0])
    assert "RAMESH" not in sent and "9123456789" not in sent
    assert seen[0]["logprobs"] is True and seen[0]["response_format"]["type"] == "json_schema"
    schema = seen[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["category"]["enum"] == ["A", "B", "C"]
    assert schema["properties"]["is_salary"]["enum"] == ["yes", "no"]


def test_teacher_falls_back_to_votes_without_logprobs():
    answers = iter(['{"category":"A","is_salary":"no"}'] * 3 + ['{"category":"B","is_salary":"no"}'] * 2)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        body = json.loads(request.content)
        if body.get("logprobs"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "logprobs": None}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": next(answers)}}]})

    t = HFTeacher(token="hf_test", samples=5, client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = t.answer([TeacherInput("UPI/DR/1/<NAME>/rent")], [Q, B])[0]
    assert calls["n"] == 6  # one logprob attempt + five votes
    assert out["category"].argmax() == 0 and out["category"][0] == pytest.approx((3.25) / (5.75))
    assert out["is_salary"].argmax() == 0


def test_teacher_requires_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        HFTeacher()


def test_teacher_temperature_head():
    rng = np.random.default_rng(0)
    probs, labels = [], []
    for _ in range(3000):  # overconfident teacher: says 0.99, right 80% of the time
        y = rng.integers(3)
        pred = y if rng.uniform() < 0.8 else (y + 1) % 3
        p = np.full(3, 0.005)
        p[pred] = 0.99
        probs.append(p)
        labels.append(y)
    assert fit_teacher_temperature(probs, labels) > 1.5  # softens it


def test_registry_versions_and_rollback(bundle, tmp_path):
    from arth.registry import Registry

    reg = Registry(tmp_path)
    assert reg.next_version() == "v1"
    bundle.version = "v1"
    reg.save(bundle)
    b2 = reg.load("v1")
    b2.version = "v2"
    reg.save(b2)
    assert reg.current() == "v2" and set(reg.list()["versions"]) == {"v1", "v2"}
    assert reg.rollback() == "v1" and reg.current() == "v1"
    with pytest.raises(RuntimeError):
        reg.rollback()
    assert reg.promote("v2") == "v2"
    manifest = json.loads((tmp_path / "v2" / "manifest.json").read_text())
    for key in ("data_snapshot", "teacher_version", "seed", "embedder"):  # reproducibility NFR
        assert key in manifest
    reg.attach_report("v2", {"summary": {"all_gates_passed": False}})
    assert reg.list()["versions"]["v2"]["gates_passed"] is False
    bundle.version = "t1"


def test_store_review_corrections_and_retention():
    s = Store(":memory:")
    s.save_taxonomy("tax_1", "t", [{"name": "c", "type": "choice", "options": {"a": "", "b": ""}}], [], None, "v1",
                    opt_in_training=False)
    ids = s.enqueue([{"line_hash": "h", "taxonomy_id": "tax_1", "question": "c",
                      "question_spec": {"type": "choice", "options": ["a", "b"]}, "scrubbed_text": "UPI/<NAME>",
                      "top3": [{"value": "a", "prob": 0.6}], "value": "a", "route": "general",
                      "correction_key": "k1", "model_version": "student:v1"}])
    assert len(s.review_queue("tax_1")) == 1
    s.add_correction("c", "b", "tax_1", "k1", "UPI/<NAME>", review_id=ids[0], rendered_text="debit upi | upi/<name>")
    assert s.review_queue("tax_1") == [] and s.get_review(ids[0])["corrected_value"] == "b"
    s.add_correction("c", "b", "tax_1", "k1", "UPI/<NAME>")
    s.add_correction("c", "a", "tax_1", "k1", "UPI/<NAME>")
    assert s.lookup_corrections(["k1", "missing"]) == {"k1": "b"}  # majority
    assert s.corrections_for("tax_1")["c"] == [("debit upi | upi/<name>", "b")]
    later = datetime.now(timezone.utc) + timedelta(days=31)
    purged = s.purge_expired(30, now=later)
    assert purged == {"review_items_purged": 1, "corrections_purged": 3}
    assert s.export_corrections() == []
    assert s.lookup_corrections(["k1"]) == {"k1": "b"}  # labels survive, text does not
