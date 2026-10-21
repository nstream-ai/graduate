"""Paired, session-clustered non-inferiority test (percentile bootstrap)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

_MAX_BATCH_CELLS = 4_000_000


@dataclass(frozen=True, slots=True)
class NonInferiority:
    """Outcome of the test. ``lower`` is the one-sided lower bound on mean(s - t)."""

    non_inferior: bool
    estimate: float
    lower: float
    margin: float
    n_sessions: int


def clustered_noninferiority(
    candidate_correct: ArrayLike,
    teacher_correct: ArrayLike,
    sessions: ArrayLike,
    rng: np.random.Generator,
    margin: float = 0.02,
    level: float = 0.05,
    n_boot: int = 5000,
) -> NonInferiority:
    """Claim candidate is not worse than teacher by more than ``margin``.

    Resamples whole sessions. Non-inferior iff the ``level`` quantile of the
    bootstrap mean difference exceeds ``-margin``. Percentile bootstrap can be
    anti-conservative with few sessions; the simulation suite measures this.
    """
    s = np.asarray(candidate_correct, dtype=np.float64)
    t = np.asarray(teacher_correct, dtype=np.float64)
    sess = np.asarray(sessions)
    if not (s.shape == t.shape == sess.shape) or s.ndim != 1:
        raise ValueError("inputs must be 1-D arrays of equal length")
    if s.size == 0:
        raise ValueError("need at least one observation")

    _, code = np.unique(sess, return_inverse=True)
    code = code.reshape(-1)
    m = int(code.max()) + 1
    sums = np.bincount(code, weights=s - t, minlength=m)
    counts = np.bincount(code, minlength=m).astype(np.float64)

    batch = max(1, _MAX_BATCH_CELLS // m)
    means = np.empty(n_boot)
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        pick = rng.integers(0, m, size=(b, m))
        means[done : done + b] = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
        done += b

    lower = float(np.quantile(means, level))
    return NonInferiority(
        non_inferior=lower > -margin,
        estimate=float(sums.sum() / counts.sum()),
        lower=lower,
        margin=margin,
        n_sessions=m,
    )
