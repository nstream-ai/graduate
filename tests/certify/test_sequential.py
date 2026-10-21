import numpy as np
import pytest

from graduate.certify.stats import betting_risk_test, session_counts

ALPHA, DELTA, CAP = 0.03, 0.05, 5


def test_session_counts_caps_and_orders_by_arrival():
    s = np.array([7, 7, 7, 7, 7, 7, 3, 3, 9])
    acc = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0], bool)
    bad = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0], bool)
    e, a = session_counts(s, acc, bad, cap=4)
    assert a.tolist() == [4, 1, 0]  # arrival order: 7, 3, 9
    assert e.tolist() == [4, 0, 0]  # unaccepted error in session 3 is ignored


def test_session_counts_keeps_first_calls_in_time_order():
    # Router lets the student take the first `cap` accepted calls; later ones go to teacher.
    s = np.array([1, 1, 1, 1])
    acc = np.array([1, 0, 1, 1], bool)
    bad = np.array([0, 1, 0, 1], bool)
    e, a = session_counts(s, acc, bad, cap=2)
    assert (e.tolist(), a.tolist()) == ([0], [2])


def test_rejects_invalid_counts():
    with pytest.raises(ValueError):
        betting_risk_test([2], [1], ALPHA, CAP)
    with pytest.raises(ValueError):
        betting_risk_test([0], [6], ALPHA, CAP)


def test_no_sessions_no_claim():
    assert betting_risk_test([], [], ALPHA, CAP).pvalue == 1.0


def test_clean_evidence_certifies_only_with_enough_sessions():
    small = betting_risk_test(np.zeros(20), np.full(20, CAP), ALPHA, CAP)
    large = betting_risk_test(np.zeros(300), np.full(300, CAP), ALPHA, CAP)
    assert small.pvalue > DELTA
    assert large.pvalue <= DELTA
    assert large.final_pvalue >= large.pvalue


@pytest.mark.parametrize("clustering", ["none", "beta", "all_or_nothing"])
def test_anytime_valid_at_the_null(clustering):
    """At risk exactly alpha, sup-capital crosses 1/delta in <= delta of runs."""
    rng = np.random.default_rng(42)
    reps, n = 2000, 400
    rejects = 0
    for _ in range(reps):
        a = np.full(n, CAP)
        if clustering == "none":
            e = rng.binomial(CAP, ALPHA, n)
        elif clustering == "beta":
            p = rng.beta(ALPHA * 2, (1 - ALPHA) * 2, n)  # mean alpha, heavy overdispersion
            e = rng.binomial(CAP, p)
        else:
            e = np.where(rng.random(n) < ALPHA, CAP, 0)  # whole sessions fail together
        rejects += betting_risk_test(e, a, ALPHA, CAP).pvalue <= DELTA
    mc = 2 * np.sqrt(DELTA * (1 - DELTA) / reps)
    assert rejects / reps <= DELTA + mc


# --- general mean test, paired (claim B), session-level (coupling) ----------

from graduate.certify.stats import (  # noqa: E402
    betting_mean_test,
    paired_noninferiority_test,
    session_rate_test,
)


def _mc_tol(reps: int) -> float:
    return 2 * np.sqrt(DELTA * (1 - DELTA) / reps)


def test_risk_test_is_the_mean_test_on_its_statistic():
    rng = np.random.default_rng(3)
    a = rng.integers(0, CAP + 1, 500)
    e = rng.binomial(a, 0.01)
    r = betting_risk_test(e, a, ALPHA, CAP)
    m = betting_mean_test(e - ALPHA * a, CAP * (1 - ALPHA), prior_mean=-ALPHA / 2, prior_var=ALPHA)
    assert r.pvalue == m.pvalue


def test_mean_test_rejects_values_above_upper():
    with pytest.raises(ValueError):
        betting_mean_test([0.5, 2.0], upper=1.0)


@pytest.mark.parametrize("prior_mean", [0.0, -1.0])  # -1.0 = wildly optimistic prior
def test_paired_test_valid_at_the_null_even_with_a_wrong_prior(prior_mean):
    rng = np.random.default_rng(11)
    reps, n, margin = 1500, 400, 0.02
    pb, pc = 0.05, 0.03  # p_b - p_c == margin: boundary null
    rejects = 0
    for _ in range(reps):
        # Heavy session clustering: whole sessions are "hard" or "easy".
        hard = rng.random(n) < 0.3
        sb = np.where(hard, pb / 0.3 * 0.5, pb / 0.7 * 0.5)
        sc = np.where(hard, pc / 0.3 * 0.5, pc / 0.7 * 0.5)
        a = np.full(n, CAP)
        u = rng.random((n, CAP))
        b = (u < sb[:, None]).sum(1)
        c = ((u >= sb[:, None]) & (u < (sb + sc)[:, None])).sum(1)
        x = (b - c) - margin * a
        res = betting_mean_test(x, CAP * (1 - margin), prior_mean=prior_mean, prior_var=margin)
        rejects += res.pvalue <= DELTA
    assert rejects / reps <= DELTA + _mc_tol(reps)


def test_paired_test_credits_student_fixes():
    # Student is wrong where teacher is right 4% of calls (> 2% margin), but fixes
    # teacher errors 3% of calls: net 1% worse, so non-inferior at margin 2%.
    rng = np.random.default_rng(5)
    n = 3000
    a = np.full(n, CAP)
    u = rng.random((n, CAP))
    b = (u < 0.04).sum(1)
    c = ((u >= 0.04) & (u < 0.07)).sum(1)
    assert paired_noninferiority_test(b, c, a, margin=0.02, cap=CAP).pvalue <= DELTA
    # Ignoring the fixes (absolute disagreement-style bound on b alone) cannot certify 2%.
    assert betting_risk_test(b, a, 0.02, CAP).pvalue > DELTA


def test_paired_test_input_checks():
    with pytest.raises(ValueError):
        paired_noninferiority_test([2], [1], [2], 0.02, CAP)  # b + c > a


@pytest.mark.parametrize("clustering", ["binomial", "all_or_nothing"])
def test_session_rate_test_valid_at_the_null(clustering):
    rng = np.random.default_rng(21)
    reps, n, beta = 1500, 600, 0.1
    rejects = 0
    for _ in range(reps):
        if clustering == "binomial":
            d = rng.binomial(CAP, beta / CAP, n)
        else:
            d = np.where(rng.random(n) < beta / CAP, CAP, 0)
        rejects += session_rate_test(d, beta, CAP).pvalue <= DELTA
    assert rejects / reps <= DELTA + _mc_tol(reps)


def test_session_rate_test_certifies_rare_divergence():
    rng = np.random.default_rng(2)
    d = rng.binomial(CAP, 0.002, 2000)  # E[D] = 0.01 per session
    assert session_rate_test(d, 0.05, CAP).pvalue <= DELTA


# --- priors (0a.13) ---------------------------------------------------------


def test_default_prior_weight_reproduces_original_behaviour():
    rng = np.random.default_rng(9)
    a = rng.integers(0, CAP + 1, 400)
    e = rng.binomial(a, 0.01)
    base = betting_risk_test(e, a, ALPHA, CAP)
    same = betting_risk_test(
        e, a, ALPHA, CAP, prior_mean=-ALPHA / 2, prior_var=ALPHA, prior_weight=1
    )
    assert base.pvalue == same.pvalue


def test_strong_wrong_prior_cannot_break_validity():
    rng = np.random.default_rng(13)
    reps, n = 1500, 400
    rejects = 0
    for _ in range(reps):
        e = rng.binomial(CAP, ALPHA, n)  # boundary null
        a = np.full(n, CAP)
        # Prior claims risk is zero, with the weight of 1,000 sessions.
        res = betting_risk_test(
            e, a, ALPHA, CAP, prior_mean=-ALPHA * CAP, prior_var=1e-4, prior_weight=1000
        )
        rejects += res.pvalue <= DELTA
    assert rejects / reps <= DELTA + _mc_tol(reps)


def test_right_prior_speeds_up_certification_near_the_margin():
    # Near the margin (true 2.5% vs claim 3%) the plug-in estimates matter, so a
    # right prior with weight shortens the time to certify. Far below the margin
    # bets sit at their limit from the start and priors are irrelevant.
    rng = np.random.default_rng(1)
    a = np.full(20000, CAP)
    e = rng.binomial(a, 0.025)
    x_mean, x_var = (0.025 - ALPHA) * CAP, CAP * 0.025 * 0.975
    plain = betting_risk_test(e, a, ALPHA, CAP)
    informed = betting_risk_test(
        e, a, ALPHA, CAP, prior_mean=x_mean, prior_var=x_var, prior_weight=200
    )

    def first(r):
        return int(np.argmax(r.capital >= 1 / DELTA))

    assert informed.pvalue <= DELTA and plain.pvalue <= DELTA
    assert first(informed) < 0.7 * first(plain)


# --- mixture of bets (0a.13b) -----------------------------------------------

MIX = (0.5, 0.75, 0.9, 0.99)


def test_single_level_tuple_equals_scalar():
    rng = np.random.default_rng(4)
    a = rng.integers(0, CAP + 1, 300)
    e = rng.binomial(a, 0.01)
    assert (
        betting_risk_test(e, a, ALPHA, CAP, c=(0.75,)).pvalue
        == betting_risk_test(e, a, ALPHA, CAP, c=0.75).pvalue
    )


def test_mixture_within_factor_k_of_best_level():
    rng = np.random.default_rng(6)
    a = np.full(5000, CAP)
    e = rng.binomial(a, 0.02)
    best = max(betting_risk_test(e, a, ALPHA, CAP, c=lv).capital.max() for lv in MIX)
    mix = betting_risk_test(e, a, ALPHA, CAP, c=MIX).capital.max()
    assert best / len(MIX) <= mix * (1 + 1e-9) <= best * (1 + 1e-9)


def test_mixture_valid_at_the_null():
    rng = np.random.default_rng(23)
    reps, n = 1500, 600
    rejects = 0
    for _ in range(reps):
        e = np.where(rng.random(n) < ALPHA, CAP, 0)  # whole sessions fail together
        rejects += betting_risk_test(e, np.full(n, CAP), ALPHA, CAP, c=MIX).pvalue <= DELTA
    assert rejects / reps <= DELTA + _mc_tol(reps)


# --- confidence bounds (0a.7) -----------------------------------------------

from graduate.certify.stats import ratio_confidence_interval, ratio_upper_bound  # noqa: E402


def test_upper_bound_covers_truth_under_clustering():
    rng = np.random.default_rng(30)
    reps, n, truth = 300, 500, 0.03
    covered = 0
    for _ in range(reps):
        e = np.where(rng.random(n) < truth, CAP, 0)  # whole sessions fail together
        covered += truth <= ratio_upper_bound(e, np.full(n, CAP), CAP)
    assert covered / reps >= 0.95 - 2 * np.sqrt(0.05 * 0.95 / reps)


def test_interval_brackets_estimate_and_shrinks_with_data():
    rng = np.random.default_rng(31)
    a_small, a_big = np.full(200, CAP), np.full(5000, CAP)
    lo1, hi1 = ratio_confidence_interval(rng.binomial(CAP, 0.2, 200), a_small, CAP)
    lo2, hi2 = ratio_confidence_interval(rng.binomial(CAP, 0.2, 5000), a_big, CAP)
    assert lo1 < 0.2 < hi1 and lo2 < 0.2 < hi2
    assert hi2 - lo2 < hi1 - lo1
