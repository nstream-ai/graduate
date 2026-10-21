import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import binom

from graduate.certify.stats import accepted_samples_needed, critical_errors, plan_evidence


def test_rule_of_three_regime():
    # Zero true risk: certify as soon as zero errors suffices, 0.97^99 = 0.0490.
    assert accepted_samples_needed(0.0, 0.03) == 99


def test_exact_needs_more_than_the_normal_approximation():
    # Design §13.7 normal approximation said ~240 for alpha=3%, true 1%, 80% power.
    # The exact test first touches 80% at n=303 but only stays there from n=392.
    assert accepted_samples_needed(0.01, 0.03) == 392


def test_power_holds_for_every_larger_n():
    n = accepted_samples_needed(0.01, 0.03, max_n=5000)
    assert n is not None
    ns = np.arange(n, 5001)
    k = critical_errors(ns, 0.03, 0.05)
    assert (binom.cdf(k, ns, 0.01) >= 0.8).all()


def test_false_claims_are_never_planned():
    assert accepted_samples_needed(0.03, 0.03) is None
    assert accepted_samples_needed(0.05, 0.03) is None


@settings(max_examples=40, deadline=None)
@given(st.floats(0.0, 0.025), st.floats(0.6, 0.95))
def test_returned_n_achieves_power(risk, power):
    n = accepted_samples_needed(risk, 0.03, power=power, max_n=50_000)
    if n is not None:
        k = int(critical_errors(np.array([n]), 0.03, 0.05)[0])
        assert k >= 0 and binom.cdf(k, n, risk) >= power


def test_monte_carlo_agrees_with_plan():
    rng = np.random.default_rng(0)
    n = accepted_samples_needed(0.01, 0.03, power=0.8)
    assert n is not None
    k = int(critical_errors(np.array([n]), 0.03, 0.05)[0])
    hits = (rng.binomial(n, 0.01, 20_000) <= k).mean()
    assert hits == pytest.approx(binom.cdf(k, n, 0.01), abs=0.01)
    assert hits >= 0.79


def test_plan_converts_through_coverage():
    plan = plan_evidence(0.0, coverage=0.5, sessions_collected=50, alpha=0.03)
    assert plan is not None
    assert plan.accepted_needed == 99
    assert plan.sessions_needed == 198
    assert plan.more_sessions == 148


def test_plan_none_for_unsafe_slice():
    assert plan_evidence(0.04, coverage=0.9, sessions_collected=0, alpha=0.03) is None


# --- betting planner --------------------------------------------------------

from graduate.certify.stats import plan_betting_risk  # noqa: E402


def test_betting_planner_none_when_not_certifiable():
    a = np.full(500, 5.0)
    e = np.random.default_rng(0).binomial(5, 0.04, 500).astype(float)
    assert plan_betting_risk(e, a, 0.03, 5) is None


def test_betting_planner_matches_direct_simulation():
    rng = np.random.default_rng(1)
    a = np.full(2000, 5.0)
    e = rng.binomial(5, 0.01, 2000).astype(float)
    plan = plan_betting_risk(e, a, 0.03, 5, sims=200, n_max=5000, rng=np.random.default_rng(2))
    assert plan is not None and plan.sessions_needed is not None
    # Independent check from the true model (binomial 1% per call).
    from graduate.certify.stats import betting_risk_test

    stops = []
    for _ in range(200):
        aa = np.full(5000, 5)
        ee = rng.binomial(5, 0.01, 5000)
        cap_path = betting_risk_test(ee, aa, 0.03, 5, c=(0.5, 0.75, 0.9, 0.99)).capital
        hit = np.nonzero(cap_path >= 20)[0]
        stops.append(hit[0] + 1 if hit.size else np.inf)
    truth = np.quantile(stops, 0.8)
    assert abs(plan.sessions_needed - truth) <= 0.2 * truth


def test_planning_range_is_ordered_and_brackets_point():
    from graduate.certify.stats import plan_betting_range

    rng = np.random.default_rng(4)
    a = np.full(300, 5.0)
    e = rng.binomial(5, 0.015, 300).astype(float)
    r = plan_betting_range(
        e - 0.03 * a, 5 * 0.97, sims=60, n_max=20_000, rng=np.random.default_rng(5)
    )
    assert r.low <= r.median <= r.high


def test_plan_range_tolerates_values_at_the_upper_limit() -> None:
    """Regression: the +z*SE shift used to push X = upper past the limit and raise."""
    import numpy as np

    from graduate.certify.stats import plan_betting_range

    alpha, cap = 0.05, 2
    a = np.full(300, 2.0)
    e = (np.random.default_rng(0).random(300) < 0.04) * 2.0
    x = e - alpha * a  # includes values equal to cap * (1 - alpha)
    r = plan_betting_range(x, cap * (1 - alpha), sessions_collected=300, sims=50, n_max=5000,
                           rng=np.random.default_rng(1))  # fmt: skip
    assert r.low <= r.median <= r.high
