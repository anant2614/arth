import numpy as np

from arth.concepts import BUILTIN_BOOLS, BUILTIN_TAXONOMIES, CONCEPT_LIST, CONCEPTS
from arth.embed import get_embedder
from arth.extract import extract
from arth.features import History, amount_bucket, compute_history, history_tokens, render
from arth.questions import Question
from arth.students.fast_path import FastPath
from arth.students.general import GeneralStudent, View, random_views
from arth.synth import SPECS, generate


def test_history_tokens():
    assert history_tokens(None) == ["hist_none"]
    toks = history_tokens(History(6, 0.02, 30))
    assert {"hist_repeat", "amt_fixed", "gap_monthly", "recurring_monthly_fixed"} <= set(toks)
    assert "recurring_monthly_fixed" not in history_tokens(History(1, 0.6, None))


def test_amount_bucket():
    assert amount_bucket(None) == "amt_unknown"
    assert amount_bucket(50) == "amt_lt100"
    assert amount_bucket(18000) == "amt_10k_50k"
    assert amount_bucket(5_00_000) == "amt_gt2l"


def test_render_contains_hints_not_raw_pii():
    ex = extract("IMPS/P2A/402166/RAMESH K/rent oct")
    out = render("IMPS/P2A/<NUM>/<NAME>/rent oct", ex, 18000, "debit", History(6, 0.02, 30))
    assert out.startswith("debit imps person amt_10k_50k amt_round")
    assert "recurring_monthly_fixed" in out and "ramesh" not in out


def test_compute_history_fr13():
    lines = [
        {"text": "IMPS/P2A/1/RAMESH K/rent", "amount": 18000, "date": "2026-07-01", "direction": "debit"},
        {"text": "IMPS/P2A/2/RAMESH K/rent", "amount": 18000, "date": "2026-08-01", "direction": "debit"},
        {"text": "IMPS/P2A/3/RAMESH K/rent", "amount": 18000, "date": "2026-09-01", "direction": "debit"},
        {"text": "UPI/DR/4/SWIGGY/YESB/swiggy@ybl/Pay", "amount": 350, "date": "2026-09-03", "direction": "debit"},
    ]
    hist = compute_history(lines, [extract(l["text"]) for l in lines])
    assert hist[0].same_counterparty_count == 3 and hist[0].amount_cv == 0.0 and hist[0].median_gap_days == 31
    assert hist[3].same_counterparty_count == 1 and hist[3].median_gap_days is None


def test_builtin_taxonomies_cover_every_concept():
    for t in BUILTIN_TAXONOMIES.values():
        assert set(t.mapping) == set(CONCEPTS)
        assert set(t.mapping.values()) <= set(t.options)
    assert all(b.positives <= set(CONCEPTS) for b in BUILTIN_BOOLS.values())


def test_generator_is_deterministic_and_covers_concepts():
    a, b = generate(800, seed=5), generate(800, seed=5)
    assert a == b
    assert {r["concept"] for r in a} == set(SPECS) == set(CONCEPTS)
    assert all(r["direction"] in ("debit", "credit") for r in a)


def test_random_views_are_valid():
    for v in random_views(20, seed=0):
        t = v.target(np.eye(len(CONCEPT_LIST)))
        if v.question.type == "choice":
            assert len(set(v.question.option_names)) == len(v.question.options)
            assert np.allclose(t.sum(axis=1), 1.0)
        else:
            assert set(np.unique(t)) <= {0.0, 1.0}


def _toy():
    emb = get_embedder("hash")
    texts, labels = [], []
    words = {"food": ["swiggy order", "zomato meal", "biryani delivery"],
             "travel": ["irctc ticket", "indigo flight", "redbus seat"]}
    rng = np.random.default_rng(0)
    for _ in range(300):
        lab = rng.choice(["food", "travel"])
        texts.append(rng.choice(words[lab]) + f" ref {rng.integers(1000)}")
        labels.append(lab)
    return emb, texts, labels


def test_fast_path_learns_soft_labels():
    emb, texts, labels = _toy()
    X = emb.encode(texts)
    Y = np.array([[0.9, 0.1] if l == "food" else [0.1, 0.9] for l in labels])
    fp = FastPath(2).fit(X, Y)
    pred = fp.predict_proba(emb.encode(["swiggy order ref 1", "indigo flight ref 2"])).argmax(axis=1)
    assert pred.tolist() == [0, 1]


def test_general_student_answers_unseen_option_wording():
    pool = generate(3000, seed=1)
    from arth.prepare import prepare, prepare_rows

    emb = get_embedder("hash")
    X = emb.encode([p.rendered for p in prepare_rows(pool)])
    cp = np.zeros((len(pool), len(CONCEPT_LIST)), dtype=np.float32)
    for i, r in enumerate(pool):
        cp[i, CONCEPT_LIST.index(r["concept"])] = 1
    pf = BUILTIN_TAXONOMIES["personal_finance"]
    views = [View(Question.from_builtin(pf), pf.mapping)] + random_views(20, seed=2)
    gs = GeneralStudent(emb.dim, rank=32).fit(X, cp, views, emb.encode, epochs=3)
    # A taxonomy the student never saw, with new names and descriptions.
    q = Question.choice("kind", {"eating": "Meals from restaurants or delivery apps",
                                 "salary_income": "Monthly pay from an employer",
                                 "cash_withdrawal": "Taking cash out of an ATM"})
    lines = [prepare(t, a, d).rendered for t, a, d in [
        ("UPI/DR/402155/SWIGGY/YESB/swiggy@ybl/Pay", 350, "debit"),
        ("NEFT CR-HDFC0000001-ACME TECHNOLOGIES PVT LTD-SALARY SEP", 85000, "credit"),
        ("ATW-512345XXXXXX1234-S1ANMU12-MUMBAI", 5000, "debit")]]
    pred = gs.predict_proba(emb.encode(lines), q, emb.encode).argmax(axis=1)
    assert pred.tolist() == [0, 1, 2]
    b = Question.boolean("is_salary", "Salary credited by an employer")
    p_yes = gs.predict_proba(emb.encode(lines), b, emb.encode)[:, 1]
    assert p_yes[1] > max(p_yes[0], p_yes[2])  # ranking: never trained on this question
