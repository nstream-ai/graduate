import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from graduate.certify.stats import benjamini_hochberg, benjamini_yekutieli, bonferroni

pvec = st.lists(st.floats(0, 1), min_size=0, max_size=60)


def test_bh_known_example():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    # BH step-up at q=0.05, m=10: thresholds 0.005,0.01,0.015,... -> k=2
    assert benjamini_hochberg(p).sum() == 2


def test_bonferroni_uses_family_size():
    assert bonferroni([0.01] * 3 + [1.0] * 37).sum() == 0  # 0.01 > 0.05/40
    assert bonferroni([0.001, 1.0]).sum() == 1


def test_empty_family():
    for f in (bonferroni, benjamini_hochberg, benjamini_yekutieli):
        assert f([]).size == 0


@pytest.mark.parametrize("bad", [[0.5, float("nan")], [-0.1], [1.2]])
def test_rejects_invalid_pvalues(bad):
    with pytest.raises(ValueError):
        benjamini_hochberg(bad)


@given(pvec)
def test_bh_is_least_strict(p):
    # Theorems: BY rejections are a subset of BH's; Bonferroni's are too.
    # (BY vs Bonferroni is not ordered: BY can reject more at large k.)
    bon, by, bh = bonferroni(p), benjamini_yekutieli(p), benjamini_hochberg(p)
    assert (by <= bh).all()
    assert (bon <= bh).all()


@given(pvec, st.randoms(use_true_random=False))
def test_permutation_invariant(p, r):
    idx = list(range(len(p)))
    r.shuffle(idx)
    shuffled = [p[i] for i in idx]
    assert benjamini_hochberg(shuffled).sum() == benjamini_hochberg(p).sum()


@given(pvec)
def test_rejections_are_the_smallest_pvalues(p):
    passed = benjamini_hochberg(p)
    arr = np.asarray(p)
    if passed.any() and (~passed).any():
        assert arr[passed].max() <= arr[~passed].min()


def test_undefined_slices_count_toward_family_size():
    # One strong slice alone survives; with 39 undefined (p=1) siblings Bonferroni is stricter.
    assert bonferroni([0.01]).all()
    assert not bonferroni([0.01] + [1.0] * 39).any()


def test_holm_dominates_bonferroni_and_controls_fwer() -> None:
    from graduate.certify.stats import holm

    rng = np.random.default_rng(11)
    p = np.array([0.001, 0.012, 0.02, 0.5])
    assert holm(p, 0.05).tolist() == [True, True, True, False]  # 0.02 <= 0.05 / 2
    assert bonferroni(p, 0.05).tolist() == [True, True, False, False]
    assert (holm(p, 0.05) >= bonferroni(p, 0.05)).all()
    false = 0
    for _ in range(2000):
        false += holm(rng.random(10), 0.05).any()  # all nulls true
    assert false / 2000 <= 0.05 + 2 * np.sqrt(0.05 * 0.95 / 2000)
    assert holm(np.zeros(0)).size == 0
