import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from graduate.certify.stats import one_step_per_session


def test_one_per_session():
    rng = np.random.default_rng(0)
    s = np.array([1, 1, 1, 2, 2, 3])
    idx = one_step_per_session(s, rng)
    assert len(idx) == 3 and set(s[idx]) == {1, 2, 3}


def test_groups_keep_one_per_group_session():
    rng = np.random.default_rng(0)
    s = np.array([1, 1, 1, 1])
    g = np.array(["a", "a", "b", "b"])
    idx = one_step_per_session(s, rng, groups=g)
    assert sorted(g[idx]) == ["a", "b"]


def test_empty():
    assert one_step_per_session([], np.random.default_rng(0)).size == 0


def test_reproducible_and_uniform():
    s = np.zeros(4, dtype=int)
    a = one_step_per_session(s, np.random.default_rng(7))
    b = one_step_per_session(s, np.random.default_rng(7))
    assert (a == b).all()
    rng = np.random.default_rng(0)
    picks = np.bincount([one_step_per_session(s, rng)[0] for _ in range(4000)], minlength=4)
    assert picks.min() > 800


@given(st.lists(st.integers(0, 6), min_size=1, max_size=50), st.integers(0, 2**32 - 1))
def test_unique_sessions_covered_exactly_once(sess, seed):
    s = np.array(sess)
    idx = one_step_per_session(s, np.random.default_rng(seed))
    assert sorted(s[idx]) == sorted(set(sess))


# --- certified population: thinning ----------------------------------------

from graduate.certify.stats import thinned_one_per_session  # noqa: E402


def test_thinning_samples_every_population_call_equally():
    rng = np.random.default_rng(0)
    lengths = [1, 2, 4, 7]  # cap 4: session of 7 contributes only its first 4 calls
    s = np.repeat(np.arange(len(lengths)), lengths)
    acc = np.ones(s.size, bool)
    reps = 40_000
    counts = np.zeros(s.size)
    for _ in range(reps):
        counts[thinned_one_per_session(s, acc, 4, rng)] += 1
    population = np.r_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # rows inside the cap
    beyond = np.r_[11, 12, 13]
    assert counts[beyond].sum() == 0
    assert counts[population] / reps == pytest.approx(np.full(population.size, 0.25), abs=0.01)


def test_thinning_is_unbiased_where_one_step_per_session_is_not():
    # Long sessions are all wrong, short ones all right: per-call risk is 8/9.
    rng = np.random.default_rng(1)
    lengths = np.tile([1, 8], 5000)
    s = np.repeat(np.arange(lengths.size), lengths)
    acc = np.ones(s.size, bool)
    wrong = np.repeat(lengths == 8, lengths)
    thin = thinned_one_per_session(s, acc, 8, rng)
    plain = one_step_per_session(s, rng)
    assert wrong[thin].mean() == pytest.approx(8 / 9, abs=0.02)
    assert wrong[plain].mean() == pytest.approx(0.5, abs=0.02)  # session-weighted: biased
