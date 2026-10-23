"""AC-1 (2026-10-23): task-level bounds and e-BH (graduate.certify.stats.taskbounds)."""

import numpy as np

from graduate.certify.stats.taskbounds import e_bh, e_value, finite_upper, t_upper


def test_e_bh_rejects_by_rank():
    # K = 3, level 0.05: thresholds 60, 30, 20 for ranks 1, 2, 3
    assert e_bh([100.0, 25.0, 1.0], 0.05) == [True, False, False]
    assert e_bh([100.0, 35.0, 1.0], 0.05) == [True, True, False]
    assert e_bh([10.0, 10.0], 0.05) == [False, False]
    assert e_bh([], 0.05) == []


def test_finite_upper_is_at_least_the_mean_and_tightens_with_tasks():
    rng = np.random.default_rng(0)
    small = rng.choice([-1.0, 0.0, 0.0, 0.0, 1.0], size=30)
    large = rng.choice([-1.0, 0.0, 0.0, 0.0, 1.0], size=300)
    assert finite_upper(small, perms=20) >= small.mean()
    assert finite_upper(large, perms=20) < finite_upper(small, perms=20)


def test_one_sided_null_is_not_certified_by_the_point_estimate():
    # the old conditional bound returned the point estimate when no task was better; here every
    # task's harm is 0 or 0.25 (no better task), mean 0.10: the bound must exceed 0.10
    d = np.array([0.25] * 12 + [0.0] * 18)
    assert finite_upper(d, perms=20) > d.mean()
    assert t_upper(d) > d.mean()


def test_all_ties_does_not_read_non_inferior_from_one_task():
    from graduate.certify.session import compare

    r = compare({"t0": [1.0]}, {"t0": [1.0]}, margin=0.10)
    assert r["verdict"].startswith("cannot be decided")


def test_e_value_is_one_with_no_tasks():
    assert e_value(np.array([]), 0.1) == 1.0
