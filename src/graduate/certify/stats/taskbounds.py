"""Task-level bounds on a mean difference (analysis correction AC-1, 2026-10-23).

The unit is the task: d_t in [-1, 1] is the per-task mean difference (reference minus candidate,
over the task's states or trials), so magnitudes count, not signs. Two one-sided upper bounds on
E[d]:

- finite-sample: H0 "E[d] >= delta" is tested with the anytime-valid betting test on
  X_t = d_t - delta (X_t <= 1 - delta); the fixed-time e-value is the mean final capital over a
  fixed set of permutations of the task order (a mean of e-values is an e-value, and each
  permutation's capital is one under exchangeable tasks). The upper bound is the smallest delta
  whose H0 is rejected at the level, found by bisection;
- asymptotic: the task-cluster t bound, mean + t_{1-level, n-1} * sd / sqrt(n).

Multiplicity across slices: e-BH (Wang and Ramdas 2022), valid under arbitrary dependence (slices
share tasks).
"""

from __future__ import annotations

import math

import numpy as np

from graduate.certify.stats.sequential import betting_mean_test

LEVELS = (0.5, 0.75, 0.9)
PERMUTATIONS = 200


def e_value(d, delta: float, perms: int = PERMUTATIONS, seed: int = 0) -> float:
    """Fixed-time e-value against H0: E[d] >= delta, for d_t in [-1, 1]."""
    x = np.asarray(d, dtype=float) - delta
    if x.size == 0 or delta >= 1.0:
        return 1.0
    rng = np.random.default_rng(seed)
    caps = [betting_mean_test(rng.permutation(x), 1.0 - delta, c=LEVELS).capital[-1]
            for _ in range(perms)]  # fmt: skip
    return float(np.mean(caps))


def finite_upper(d, level: float = 0.05, perms: int = PERMUTATIONS, seed: int = 0,
                 tol: float = 1e-3) -> float:  # fmt: skip
    """Smallest delta in [-1, 1] whose H0 "E[d] >= delta" the e-value rejects at `level`."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return 1.0
    lo, hi = float(d.mean()), 1.0 - 1e-9  # the bound is never below the observed mean
    if e_value(d, hi, perms, seed) < 1.0 / level:
        return 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if e_value(d, mid, perms, seed) >= 1.0 / level:
            hi = mid
        else:
            lo = mid
    return float(hi)


def t_upper(d, level: float = 0.05) -> float:
    """Task-cluster t upper bound on E[d] (asymptotic)."""
    from scipy.stats import t as student_t

    d = np.asarray(d, dtype=float)
    n = d.size
    if n < 2:
        return 1.0
    return float(d.mean() + student_t.ppf(1 - level, n - 1) * d.std(ddof=1) / math.sqrt(n))


def e_bh(evalues: list[float], level: float = 0.05) -> list[bool]:
    """e-BH: reject the k largest e-values, k = max{k : e_(k) >= K / (level k)}."""
    k_all = len(evalues)
    if not k_all:
        return []
    order = sorted(range(k_all), key=lambda i: -evalues[i])
    k_star = 0
    for rank, i in enumerate(order, start=1):
        if evalues[i] >= k_all / (level * rank):
            k_star = rank
    keep = set(order[:k_star])
    return [i in keep for i in range(k_all)]
