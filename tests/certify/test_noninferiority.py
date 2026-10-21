import numpy as np
import pytest

from graduate.certify.stats import clustered_noninferiority


def test_identical_models_are_noninferior():
    rng = np.random.default_rng(0)
    c = rng.random(400) < 0.9
    sess = np.repeat(np.arange(200), 2)
    r = clustered_noninferiority(c, c, sess, rng, margin=0.02)
    assert r.non_inferior and r.estimate == 0.0 and r.n_sessions == 200


def test_clearly_worse_candidate_fails():
    rng = np.random.default_rng(0)
    t = np.ones(400)
    s = np.r_[np.ones(300), np.zeros(100)]
    r = clustered_noninferiority(s, t, np.arange(400), rng, margin=0.02)
    assert not r.non_inferior and r.estimate == pytest.approx(-0.25)


def test_input_validation():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        clustered_noninferiority([1, 0], [1], [1, 2], rng)
    with pytest.raises(ValueError):
        clustered_noninferiority([], [], [], rng)


def test_matches_naive_bootstrap_distributionally():
    rng = np.random.default_rng(3)
    sess = np.repeat(np.arange(50), 3)
    d_s = rng.random(150) < 0.9
    d_t = rng.random(150) < 0.92
    r = clustered_noninferiority(d_s, d_t, sess, np.random.default_rng(1), n_boot=4000)
    naive = []
    ids = np.unique(sess)
    d = d_s.astype(float) - d_t
    r2 = np.random.default_rng(2)
    for _ in range(4000):
        pick = r2.choice(ids, ids.size)
        naive.append(np.concatenate([d[sess == i] for i in pick]).mean())
    assert r.lower == pytest.approx(np.quantile(naive, 0.05), abs=0.01)
