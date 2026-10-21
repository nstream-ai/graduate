import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import binom

from graduate.certify.stats import clopper_pearson_interval, clopper_pearson_upper, risk_pvalue


def test_rule_of_three():
    # zero errors in n: upper bound ~ 3/n
    assert clopper_pearson_upper(0, 100, 0.05) == pytest.approx(0.0295, abs=1e-3)


def test_no_data_gives_no_claim():
    assert clopper_pearson_upper(0, 0, 0.05) == 1.0
    assert clopper_pearson_interval(0, 0) == (0.0, 1.0)
    assert risk_pvalue(0, 0, 0.03) == 1.0


def test_all_errors():
    assert clopper_pearson_upper(10, 10, 0.05) == 1.0
    assert clopper_pearson_interval(10, 10)[1] == 1.0


@pytest.mark.parametrize("bad", [(-1, 5), (6, 5), (0, -1)])
def test_rejects_bad_counts(bad):
    with pytest.raises(ValueError):
        risk_pvalue(*bad, 0.05)


@given(n=st.integers(1, 500), data=st.data(), alpha=st.floats(0.001, 0.5))
def test_pvalue_monotone_in_errors(n, data, alpha):
    e = data.draw(st.integers(0, n - 1))
    assert risk_pvalue(e, n, alpha) <= risk_pvalue(e + 1, n, alpha) + 1e-12


@given(n=st.integers(1, 500), data=st.data(), delta=st.floats(0.01, 0.2))
def test_upper_bound_matches_pvalue_duality(n, data, delta):
    e = data.draw(st.integers(0, n))
    ub = clopper_pearson_upper(e, n, delta)
    if 0 < ub < 1:
        # testing H0: p >= ub gives p-value == delta exactly at the bound
        assert risk_pvalue(e, n, ub) == pytest.approx(delta, abs=1e-6)


def test_interval_coverage_is_conservative():
    rng = np.random.default_rng(1)
    p, n, reps = 0.03, 200, 4000
    hits = 0
    for e in rng.binomial(n, p, reps):
        lo, hi = clopper_pearson_interval(int(e), n, 0.95)
        hits += lo <= p <= hi
    assert hits / reps >= 0.94


def test_pvalue_is_valid_under_null():
    # Under p = alpha, P(pvalue <= x) <= x for all x (exactness up to discreteness).
    alpha, n = 0.05, 150
    pmf = binom.pmf(np.arange(n + 1), n, alpha)
    pvals = np.array([risk_pvalue(e, n, alpha) for e in range(n + 1)])
    for x in (0.01, 0.05, 0.1, 0.5):
        assert pmf[pvals <= x].sum() <= x + 1e-12
