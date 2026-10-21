"""Confidence-threshold selection on the dev split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom


@dataclass(frozen=True, slots=True)
class ThresholdChoice:
    """A threshold and what it achieved on the data it was chosen on.

    ``pass_probability`` is set only by power-aware selection: the estimated
    chance that the calibration test certifies this threshold.
    """

    tau: float
    accepted: int
    risk: float
    coverage: float
    pass_probability: float | None = None


@dataclass(frozen=True, slots=True)
class _Curve:
    """Risk-coverage curve at every distinct confidence value, loosest last."""

    tau: NDArray[np.float64]
    accepted: NDArray[np.int64]
    errors: NDArray[np.int64]
    n: int


def _curve(conf: ArrayLike, correct: ArrayLike) -> _Curve | None:
    c = np.asarray(conf, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    if c.shape != ok.shape or c.ndim != 1:
        raise ValueError("conf and correct must be 1-D arrays of equal length")
    if np.isnan(c).any():
        raise ValueError("conf must not contain NaN")
    if c.size == 0:
        return None
    order = np.argsort(-c, kind="stable")
    sorted_conf = c[order]
    errors = np.cumsum(~ok[order])
    accepted = np.arange(1, c.size + 1)
    # Ties are accepted together, so only the last row of each tie is a candidate.
    last_of_tie = np.append(sorted_conf[1:] != sorted_conf[:-1], True)
    return _Curve(sorted_conf[last_of_tie], accepted[last_of_tie], errors[last_of_tie], c.size)


def _choice(curve: _Curve, i: int, pass_probability: float | None = None) -> ThresholdChoice:
    return ThresholdChoice(
        tau=float(curve.tau[i]),
        accepted=int(curve.accepted[i]),
        risk=float(curve.errors[i] / curve.accepted[i]),
        coverage=float(curve.accepted[i] / curve.n),
        pass_probability=pass_probability,
    )


def choose_threshold(
    conf: ArrayLike,
    correct: ArrayLike,
    alpha: float,
    min_accepted: int = 1,
) -> ThresholdChoice | None:
    """Loosest threshold whose empirical risk among accepted calls is <= alpha.

    A call is accepted when ``conf >= tau``. Returns None when no threshold
    qualifies; callers must treat that slice as p = 1.0.

    Intended for the dev split only. Choosing at the certification target
    itself puts the true risk near the boundary, where the calibration test
    rarely passes; prefer :func:`choose_threshold_for_power`.
    """
    curve = _curve(conf, correct)
    if curve is None:
        return None
    qualifies = (curve.errors <= alpha * curve.accepted) & (curve.accepted >= min_accepted)
    idx = np.nonzero(qualifies)[0]
    return _choice(curve, int(idx[-1])) if idx.size else None


def critical_errors(n: NDArray[np.int64], alpha: float, level: float) -> NDArray[np.int64]:
    """Largest error count that certifies risk < alpha at ``level`` with n samples.

    -1 where no error count certifies (n too small).
    """
    n = np.asarray(n, dtype=np.int64)
    k = binom.ppf(level, n, alpha).astype(np.int64)
    k = np.where(binom.cdf(k, n, alpha) <= level, k, k - 1)
    return np.where(n > 0, k, -1)


def choose_threshold_for_power(
    conf: ArrayLike,
    correct: ArrayLike,
    alpha: float,
    n_calibration: int,
    level: float = 0.05,
    min_accepted: int = 1,
) -> ThresholdChoice | None:
    """Threshold maximizing expected certified coverage: coverage x P(certified).

    For each candidate threshold, the dev risk (with a Jeffreys-style +0.5
    correction so zero observed errors is not treated as zero risk) predicts
    how many errors the calibration split will show among its expected
    ``coverage * n_calibration`` accepted samples, and so the probability the
    exact test at ``level`` certifies it. ``n_calibration`` is the number of
    independent calibration samples (one per session). ``level`` should be the
    per-slice level the family procedure will roughly apply.

    Uses dev data only, so the calibration test stays valid. Returns None when
    no threshold has a positive chance of certifying.
    """
    if n_calibration < 1:
        raise ValueError("n_calibration must be >= 1")
    curve = _curve(conf, correct)
    if curve is None:
        return None
    coverage = curve.accepted / curve.n
    risk_hat = (curve.errors + 0.5) / (curve.accepted + 1.0)
    n_cal = np.floor(coverage * n_calibration).astype(np.int64)
    k = critical_errors(n_cal, alpha, level)
    p_pass = np.where(k >= 0, binom.cdf(np.maximum(k, 0), n_cal, risk_hat), 0.0)
    score = np.where(curve.accepted >= min_accepted, coverage * p_pass, 0.0)
    if not (score > 0).any():
        return None
    i = int(np.argmax(score))
    return _choice(curve, i, float(p_pass[i]))
