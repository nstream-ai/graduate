import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from graduate.certify.stats import choose_threshold


def test_basic_selection():
    conf = [0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [1, 1, 1, 0, 1]
    got = choose_threshold(conf, correct, alpha=0.0)
    assert got is not None and got.tau == 0.7 and got.accepted == 3 and got.risk == 0.0
    loose = choose_threshold(conf, correct, alpha=0.25)
    assert loose is not None and loose.tau == 0.5  # 1 error in 5 = 0.2 <= 0.25


def test_none_when_nothing_qualifies():
    assert choose_threshold([0.9, 0.8], [0, 0], alpha=0.1) is None
    assert choose_threshold([], [], alpha=0.1) is None


def test_ties_are_accepted_together():
    # Accepting 0.9 alone would be clean, but both 0.9s come in together.
    got = choose_threshold([0.9, 0.9, 0.5], [1, 0, 1], alpha=0.0)
    assert got is None


def test_min_accepted():
    assert choose_threshold([0.9, 0.8, 0.7], [1, 1, 1], 0.0, min_accepted=5) is None


@given(
    st.lists(st.tuples(st.floats(0, 1), st.booleans()), min_size=1, max_size=80),
    st.floats(0.0, 0.5),
)
def test_result_is_consistent_and_loosest(rows, alpha):
    conf = np.array([r[0] for r in rows])
    correct = np.array([r[1] for r in rows])
    got = choose_threshold(conf, correct, alpha)
    for tau in np.unique(conf):
        acc = conf >= tau
        risk = (~correct[acc]).mean()
        if risk <= alpha:
            assert got is not None and tau >= got.tau
    if got is not None:
        acc = conf >= got.tau
        assert acc.sum() == got.accepted
        assert (~correct[acc]).mean() <= alpha + 1e-12


# --- power-aware selection -------------------------------------------------

from scipy.stats import binom  # noqa: E402

from graduate.certify.stats import choose_threshold_for_power, critical_errors  # noqa: E402


def test_critical_errors_known_values():
    # 0 errors in 100 at alpha=3%: P = 0.97^100 = 0.0476 <= 0.05, so k = 0.
    assert critical_errors(np.array([0, 50, 100]), 0.03, 0.05).tolist() == [-1, -1, 0]


@given(st.integers(1, 3000), st.floats(0.005, 0.2), st.floats(0.001, 0.2))
def test_critical_errors_is_the_largest_certifying_count(n, alpha, level):
    k = int(critical_errors(np.array([n]), alpha, level)[0])
    if k >= 0:
        assert binom.cdf(k, n, alpha) <= level
    assert binom.cdf(k + 1, n, alpha) > level


def test_power_choice_is_tighter_than_boundary_choice():
    rng = np.random.default_rng(0)
    n = 6000
    wrong = rng.random(n) < 0.08
    conf = rng.standard_normal(n) + 1.8 * ~wrong
    boundary = choose_threshold(conf, ~wrong, 0.03)
    power = choose_threshold_for_power(conf, ~wrong, 0.03, n_calibration=2000)
    assert boundary is not None and power is not None
    assert power.tau > boundary.tau  # backs off the boundary to make the test passable
    assert 0.0 < power.pass_probability <= 1.0


def test_power_choice_none_when_calibration_too_small():
    conf = np.linspace(0, 1, 200)
    assert choose_threshold_for_power(conf, np.ones(200, bool), 0.03, n_calibration=50) is None


def test_power_choice_accepts_all_when_slice_is_clean():
    conf = np.linspace(0, 1, 500)
    got = choose_threshold_for_power(conf, np.ones(500, bool), 0.03, n_calibration=1000)
    assert got is not None and got.coverage == 1.0
