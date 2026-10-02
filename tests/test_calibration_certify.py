import numpy as np
import pytest

from arth.calibration import ece, fit_temperature, reliability, softmax
from arth.certify import (Certificate, cascade_pick, certify, certify_cascade, coverage_at, cp_lower,
                          min_n_to_pass)


def _sample_logits(rng, n, k, true_T):
    """Logits whose softmax at temperature true_T is perfectly calibrated."""
    logits = rng.normal(0, 3, size=(n, k))
    p = softmax(logits, true_T)
    y = np.array([rng.choice(k, p=row) for row in p])
    return logits, y


def test_temperature_recovers_known_value():
    rng = np.random.default_rng(0)
    logits, y = _sample_logits(rng, 20000, 5, true_T=2.5)
    assert abs(fit_temperature(logits, y) - 2.5) < 0.15


def test_ece_small_when_calibrated_large_when_not():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.3, 1.0, 50000)
    correct = rng.uniform(size=conf.size) < conf
    assert ece(conf, correct) < 0.01
    assert ece(np.clip(conf + 0.2, 0, 1), correct) > 0.1


def test_reliability_bins():
    rows = reliability(np.array([0.05, 0.95, 0.96]), np.array([0, 1, 1]), n_bins=10)
    assert len(rows) == 10 and rows[0]["count"] == 1 and rows[-1]["count"] == 2 and rows[-1]["accuracy"] == 1.0


def test_clopper_pearson_known_values():
    # 98% target at 95% confidence needs 149 straight successes.
    assert min_n_to_pass(0.98, 0.05) == 149
    assert cp_lower(149, 149) >= 0.98 > cp_lower(148, 148)
    assert cp_lower(0, 10) == 0.0
    # one-sided 95% lower bound for 95/100 (reference: scipy/R binom.test ~= 0.9 to 2 dp)
    assert cp_lower(95, 100) == pytest.approx(0.8971, abs=1e-3)


def test_certify_prefers_loosest_passing_threshold():
    conf = np.concatenate([np.full(300, 0.999), np.full(300, 0.9), np.full(300, 0.6)])
    correct = np.concatenate([np.ones(300), np.ones(300), np.zeros(300)]).astype(bool)
    c = certify(conf, correct)
    assert c.threshold is not None and 0.6 < c.threshold <= 0.9
    assert c.coverage == pytest.approx(600 / 900)
    assert c.lower_bound >= 0.98


def test_certify_refuses_when_precision_too_low():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1, 1000)
    correct = rng.uniform(size=1000) < 0.9
    assert certify(conf, correct).threshold is None


def test_certify_needs_enough_points():
    c = certify(np.full(100, 0.999), np.ones(100, dtype=bool))
    assert c.threshold is None and c.coverage == 0.0


def test_certificate_guarantee_holds_in_simulation():
    """Over many simulated calibration sets the certified precision on fresh data is >= 98%
    in at least ~95% of trials (the guarantee is one-sided at alpha = 0.05)."""
    rng = np.random.default_rng(42)
    trials, ok, certified = 200, 0, 0
    for _ in range(trials):
        conf = rng.beta(30, 1, 2000)
        # mildly mis-calibrated model: true accuracy is conf**1.5
        correct = rng.uniform(size=2000) < conf ** 1.5
        c = certify(conf, correct)
        if c.threshold is None:
            ok += 1
            continue
        certified += 1
        fresh = rng.beta(30, 1, 200000)
        fresh_correct = rng.uniform(size=fresh.size) < fresh ** 1.5
        prec = fresh_correct[fresh >= c.threshold].mean()
        ok += prec >= 0.98
    assert certified > trials * 0.5
    assert ok / trials >= 0.93


def test_cascade_pick_routes_cheapest_tier_first():
    nan = np.nan
    rules = np.array([[0.99, 0.01], [nan, nan], [nan, nan]])
    fast = np.array([[0.2, 0.8], [0.97, 0.03], [0.6, 0.4]])
    general = np.array([[0.5, 0.5], [0.5, 0.5], [0.3, 0.7]])
    tier, pred, conf, auto = cascade_pick([rules, fast, general], 0.95)
    assert tier.tolist() == [0, 1, 2]
    assert pred.tolist() == [0, 0, 1]
    assert auto.tolist() == [True, True, False]
    # without a certificate nothing is auto and the last tier answers
    tier, _, _, auto = cascade_pick([rules, fast, general], None)
    assert not auto.any() and tier.tolist() == [2, 2, 2]


def test_certify_cascade_matches_single_model_when_one_tier():
    rng = np.random.default_rng(3)
    p1 = rng.uniform(0.8, 1.0, 1000)
    probs = np.stack([p1, 1 - p1], axis=1)
    y = (rng.uniform(size=1000) > p1).astype(int)
    a = certify_cascade([probs], y)
    b = certify(p1, y == 0)
    assert a.threshold == b.threshold and a.n_above == b.n_above


def test_coverage_at():
    out = coverage_at(np.array([0.99, 0.5, 0.97]), np.array([True, False, False]), 0.95)
    assert out == {"coverage": pytest.approx(2 / 3), "precision": 0.5, "n_auto": 2}
    assert coverage_at(np.array([0.9]), np.array([True]), None)["coverage"] == 0.0


def test_certificate_auto():
    c = Certificate(0.9, 0.98, 0.05, 10, 5, 1.0, 0.99, 0.5)
    assert c.auto(0.95) and not c.auto(0.85)
    assert not Certificate(None, 0.98, 0.05, 0, 0, None, None, 0.0).auto(1.0)
