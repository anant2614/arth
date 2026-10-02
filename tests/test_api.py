import logging

import pytest

PRD_REQUEST = {
    "transactions": [
        {"id": "t1", "text": "IMPS/P2A/402166/RAMESH K/rent oct", "amount": 18000, "direction": "debit",
         "date": "2026-10-01", "history": {"same_counterparty_count": 6, "amount_cv": 0.02, "median_gap_days": 30}}
    ],
    "questions": [
        {"name": "category", "type": "choice",
         "options": {"rent": "Monthly house rent paid to a person or landlord",
                     "p2p_transfer": "One-off transfer to a friend or family member",
                     "emi": "Fixed loan instalment"}},
        {"name": "is_salary", "type": "bool"},
    ],
}


def test_prd_request_and_response_shape(client):
    r = client.post("/v1/categorize", json=PRD_REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()
    res = body["results"][0]
    assert res["id"] == "t1"
    assert res["extracted"]["channel"] == "IMPS" and res["extracted"]["counterparty_type"] == "person"
    assert res["extracted"]["merchant"] is None
    cat = res["answers"]["category"]
    assert cat["value"] == "rent"
    assert set(cat["probs"]) == {"rent", "p2p_transfer", "emi"} and isinstance(cat["auto"], bool)
    sal = res["answers"]["is_salary"]
    assert sal["value"] is False and 0 <= sal["prob"] < 0.5
    assert res["route"] in {"rules", "student", "teacher"}
    assert res["model_version"].startswith("student:")
    # nothing in the response echoes the raw name
    assert "RAMESH" not in r.text


def test_builtin_taxonomy(client):
    r = client.post("/v1/categorize", json={
        "taxonomy_id": "builtin:personal_finance",
        "transactions": [{"text": "Rs.450 debited from A/c XX1234 to VPA zomatoonline@paytm on 01-10-26"}]})
    assert r.status_code == 200
    a = r.json()["results"][0]["answers"]["personal_finance"]
    assert a["value"] == "food_dining"


@pytest.mark.parametrize("payload,msg", [
    ({"transactions": [{"text": "x"}]}, "questions"),
    ({"transactions": [], "questions": [{"name": "a", "type": "bool"}]}, "at least 1"),
    ({"transactions": [{"text": "x"}], "questions": [{"name": "c", "type": "choice", "options": {"a": ""}}]},
     "at least 2 options"),
    ({"transactions": [{"text": "x"}],
      "questions": [{"name": "c", "type": "choice", "options": {f"o{i}": "" for i in range(256)}}]}, "max is 255"),
    ({"transactions": [{"text": "x"}], "questions": [{"name": "b", "type": "bool", "options": {"a": "", "b": ""}}]},
     "takes no options"),
    ({"transactions": [{"text": "x"}], "questions": [{"name": "a", "type": "bool"}, {"name": "a", "type": "bool"}]},
     "duplicate"),
    ({"transactions": [{"text": "x", "direction": "sideways"}], "questions": [{"name": "a", "type": "bool"}]},
     "direction"),
    ({"transactions": [{"text": "x", "amount": -5}], "questions": [{"name": "a", "type": "bool"}]}, "amount"),
    ({"transactions": [{"text": "x"}], "questions": [{"name": "1bad", "type": "bool"}]}, "pattern"),
])
def test_validation_errors_are_clear(client, payload, msg):
    r = client.post("/v1/categorize", json=payload)
    assert r.status_code == 422
    assert msg in r.text


def test_batch_limit(client):
    q = [{"name": "is_salary", "type": "bool"}]
    ok = client.post("/v1/categorize", json={"transactions": [{"text": "UPI/DR/1/SWIGGY/YESB/swiggy@ybl/Pay"}] * 1000,
                                             "questions": q})
    assert ok.status_code == 200 and len(ok.json()["results"]) == 1000
    too_many = client.post("/v1/categorize", json={"transactions": [{"text": "x"}] * 1001, "questions": q})
    assert too_many.status_code == 422


def test_255_options_accepted(client):
    opts = {f"opt_{i}": f"Option number {i}" for i in range(255)}
    r = client.post("/v1/categorize", json={"transactions": [{"text": "UPI/DR/1/SWIGGY/YESB/swiggy@ybl/Pay"}],
                                            "questions": [{"name": "big", "type": "choice", "options": opts}]})
    assert r.status_code == 200 and len(r.json()["results"][0]["answers"]["big"]["probs"]) == 255


def test_unknown_taxonomy_404(client):
    r = client.post("/v1/categorize", json={"taxonomy_id": "tax_nope", "transactions": [{"text": "x"}]})
    assert r.status_code == 404 and "unknown taxonomy" in r.text


def _labeled(gold_rows, n=None):
    from arth.concepts import LENDER_FLAGS

    rows = gold_rows[:n] if n else gold_rows
    return [{"text": g["text"], "amount": g["amount"], "direction": g["direction"], "history": g.get("history"),
             "labels": {"flag": LENDER_FLAGS.label(g["concept"]), "is_salary": g["concept"] == "salary"}}
            for g in rows]


def test_save_taxonomy_certify_review_and_correct(client, gold_split):
    from arth.concepts import LENDER_FLAGS

    cal, _ = gold_split
    r = client.post("/v1/taxonomies", json={
        "name": "lender flags",
        "questions": [{"name": "flag", "type": "choice", "options": LENDER_FLAGS.options},
                      {"name": "is_salary", "type": "bool", "description": "Salary credited by an employer"}],
        "labeled_sample": _labeled(cal)})
    assert r.status_code == 201, r.text
    tax = r.json()
    tid = tax["id"]
    assert tax["n_labeled"] == len(cal)
    cert = tax["questions"]["is_salary"]["certificate"]
    assert cert["n_calibration"] == len(cal) and cert["threshold"] is not None
    assert client.get(f"/v1/taxonomies/{tid}").json()["id"] == tid
    assert any(t["id"] == tid for t in client.get("/v1/taxonomies").json()["saved"])

    # an ambiguous line ends up in the review queue with its top 3
    line = {"id": "x1", "text": "UPI/DR/617912340101/PRIYA NAIR/HDFC/priya.nair@okhdfcbank/oct", "amount": 15000,
            "direction": "debit"}
    res = client.post("/v1/categorize", json={"taxonomy_id": tid, "transactions": [line]}).json()["results"][0]
    assert "PRIYA" not in str(res)
    if res["answers"]["flag"]["auto"]:
        pytest.skip("model was certain; nothing to review")
    rid = res["review"]["flag"]
    queue = client.get("/v1/review", params={"taxonomy_id": tid}).json()["items"]
    item = next(i for i in queue if i["id"] == rid)
    assert len(item["top3"]) == 3 and "priya" not in item["scrubbed_text"].lower()

    # reviewer says it was a loan repayment to a person: correction resolves the item ...
    c = client.post("/v1/corrections", json={"review_id": rid, "value": "obligation"})
    assert c.status_code == 201
    assert all(i["id"] != rid for i in client.get("/v1/review", params={"taxonomy_id": tid}).json()["items"])
    # ... and the same counterparty next month is labelled the same way, automatically
    nxt = {**line, "id": "x2", "text": "UPI/DR/620012340202/PRIYA NAIR/HDFC/priya.nair@okhdfcbank/nov"}
    res2 = client.post("/v1/categorize", json={"taxonomy_id": tid, "transactions": [nxt]}).json()["results"][0]
    assert res2["answers"]["flag"] == {**res2["answers"]["flag"], "value": "obligation", "auto": True,
                                       "route": "correction"}

    # invalid correction values are rejected
    bad = client.post("/v1/corrections", json={"review_id": rid, "value": "not_an_option"})
    assert bad.status_code == 422
    exp = client.get("/v1/corrections/export").json()["corrections"]
    assert exp and all("priya" not in e["scrubbed_text"].lower() for e in exp)

    # corrections flow into the taxonomy's fast path on recertify
    rc = client.post(f"/v1/taxonomies/{tid}/recertify")
    assert rc.status_code == 200 and rc.json()["questions"]["flag"]["certificate"] is not None


def test_correction_by_transaction(client):
    line = {"text": "IMPS/P2A/402166/SURESH IYER/oct", "amount": 12000, "direction": "debit"}
    r = client.post("/v1/corrections", json={"transaction": line, "taxonomy_id": "builtin:personal_finance",
                                             "question": "personal_finance", "value": "rent"})
    assert r.status_code == 201
    res = client.post("/v1/categorize", json={"taxonomy_id": "builtin:personal_finance",
                                              "transactions": [{**line, "text": "IMPS/P2A/509999/SURESH IYER/nov"}]})
    a = res.json()["results"][0]["answers"]["personal_finance"]
    assert a["value"] == "rent" and a["route"] == "correction"


def test_taxonomy_label_validation(client):
    r = client.post("/v1/taxonomies", json={
        "name": "t", "questions": [{"name": "c", "type": "choice", "options": {"a": "A", "b": "B"}}],
        "labeled_sample": [{"text": "x", "labels": {"c": "zzz"}}]})
    assert r.status_code == 422 and "not an option" in r.text


def test_models_rollback_and_health(client, service, bundle):
    import copy

    h = client.get("/v1/health").json()
    assert h["status"] == "ok" and h["model_version"] == "student:t1" and h["audit"]["alert"] is False
    b2 = copy.copy(bundle)
    b2.version = "t2"
    service.registry.save(b2)
    service.set_bundle(b2)
    assert client.get("/v1/health").json()["model_version"] == "student:t2"
    r = client.post("/v1/models/rollback")
    assert r.status_code == 200 and r.json()["current"] == "student:t1"
    assert client.get("/v1/health").json()["model_version"] == "student:t1"
    assert client.post("/v1/models/rollback").status_code == 409
    assert client.post("/v1/models/promote", json={"version": "t2"}).json()["current"] == "student:t2"
    assert client.post("/v1/models/promote", json={"version": "nope"}).status_code == 404
    assert set(client.get("/v1/models").json()["versions"]) == {"t1", "t2"}


def test_api_keys(service, monkeypatch):
    from fastapi.testclient import TestClient

    from arth.api import create_app

    monkeypatch.setenv("ARTH_API_KEYS", "k1,k2")
    c = TestClient(create_app(service))
    body = {"transactions": [{"text": "x"}], "questions": [{"name": "a", "type": "bool"}]}
    assert c.post("/v1/categorize", json=body).status_code == 401
    assert c.post("/v1/categorize", json=body, headers={"X-API-Key": "k2"}).status_code == 200
    assert c.get("/v1/health").status_code == 200  # health stays open


def test_no_raw_text_in_logs(client, caplog):
    caplog.set_level(logging.DEBUG)
    client.post("/v1/categorize", json=PRD_REQUEST)
    client.post("/v1/categorize", json={"transactions": [{"text": "zzqx call 9123456789 RAMESH"}],
                                        "questions": [{"name": "a", "type": "bool"}]})
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "/v1/categorize" in logged
    for secret in ("RAMESH", "9123456789", "rent oct"):
        assert secret not in logged


def test_no_model_loaded_is_503():
    from fastapi.testclient import TestClient

    from arth.api import Service, create_app
    from arth.store import Store

    c = TestClient(create_app(Service(Store(":memory:"))))
    r = c.post("/v1/categorize", json={"transactions": [{"text": "x"}], "questions": [{"name": "a", "type": "bool"}]})
    assert r.status_code == 503
    assert c.get("/v1/health").json()["status"] == "no_model"
