"""Power planning: how much evidence a slice needs before it can be certified."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from graduate.certify.stats.threshold import critical_errors

_DEFAULT_MAX_N = 200_000


@dataclass(frozen=True, slots=True)
class EvidencePlan:
    """Evidence needed for the exact one-step-per-session test to reach ``power``.

    ``accepted_needed`` counts independent accepted samples; ``sessions_needed``
    converts that through the slice's coverage (one sample per session).
    ``more_sessions`` is what remains after the sessions already collected.
    """

    accepted_needed: int
    sessions_needed: int
    more_sessions: int
    power: float


def accepted_samples_needed(
    true_risk: float,
    alpha: float,
    level: float = 0.05,
    power: float = 0.8,
    max_n: int = _DEFAULT_MAX_N,
) -> int | None:
    """Smallest n from which the exact test keeps power >= ``power`` (up to ``max_n``).

    The exact test's power zig-zags in n because of discreteness, so a planner
    that returned the first n to touch ``power`` would promise a lucky spike that
    the next few sessions undo. This returns the n after which power never drops
    below the target. Returns None when ``true_risk >= alpha`` (the claim is
    false, so no amount of evidence should certify it) or when more than
    ``max_n`` samples are needed.
    """
    if not 0.0 <= true_risk < 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("need 0 <= true_risk < 1 and 0 < alpha < 1")
    if not 0.0 < power < 1.0 or not 0.0 < level < 1.0:
        raise ValueError("power and level must be in (0, 1)")
    if true_risk >= alpha:
        return None
    n = np.arange(1, max_n + 1, dtype=np.int64)
    k = critical_errors(n, alpha, level)
    reach = np.where(k >= 0, binom.cdf(np.maximum(k, 0), n, true_risk), 0.0)
    # Power from n onward (min over the tail); monotone, so the first True is the answer.
    stays = np.minimum.accumulate(reach[::-1])[::-1] >= power
    return int(n[np.argmax(stays)]) if stays[-1] else None


def plan_evidence(
    risk_estimate: float,
    coverage: float,
    sessions_collected: int,
    alpha: float,
    level: float = 0.05,
    power: float = 0.8,
) -> EvidencePlan | None:
    """Sessions a slice still needs, given its estimated risk and coverage at tau.

    ``risk_estimate`` should come from data the certificate will not use (dev or
    earlier shadow evidence). ``level`` is the per-slice level the family
    procedure applies (e.g. delta / S for Bonferroni). None means the slice is
    not expected to certify at this threshold.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    if sessions_collected < 0:
        raise ValueError("sessions_collected must be >= 0")
    n = accepted_samples_needed(risk_estimate, alpha, level, power)
    if n is None:
        return None
    sessions = int(np.ceil(n / coverage))
    return EvidencePlan(n, sessions, max(0, sessions - sessions_collected), power)


# --- planner for the anytime-valid betting test (default in `tf certify`) ---

DEFAULT_MIXTURE = (0.5, 0.75, 0.9, 0.99)


@dataclass(frozen=True, slots=True)
class BettingPlan:
    """Predicted evidence for the betting test, by resampling the slice's own sessions.

    ``sessions_needed`` is the smallest n at which P(certified by n) reaches
    ``power`` (None if not reached within ``n_max``); ``p_certify`` is the
    predicted probability of certifying within ``n_max``; ``median_sessions`` the
    median stopping time among simulated runs that certify.
    """

    sessions_needed: int | None
    more_sessions: int | None
    p_certify: float
    median_sessions: float | None
    power: float
    n_max: int


def plan_betting_test(
    x_dev: np.ndarray,
    upper: float,
    sessions_collected: int = 0,
    power: float = 0.8,
    level: float = 0.05,
    c: float | tuple[float, ...] = DEFAULT_MIXTURE,
    n_max: int = 20_000,
    sims: int = 200,
    rng: np.random.Generator | None = None,
    **prior: float,
) -> BettingPlan | None:
    """How many sessions the betting test needs, for any per-session statistic X.

    ``x_dev`` are per-session values of X from data the certificate will not use
    (dev or earlier shadow traffic): e.g. d - alpha * a for claims A/C, or
    (b - c) - margin * a for claim B. Streams are resampled from ``x_dev`` and the
    same test as `tf certify` is run on them. Returns None when the mean of X is
    not negative (the slice is not expected to certify at this margin).

    This is a plug-in estimate: it treats the dev sessions as the population, so
    it is optimistic when dev is small or the slice is near its margin.
    """
    from graduate.certify.stats.sequential import betting_mean_test

    x = np.asarray(x_dev, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x_dev must be a non-empty 1-D array")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if x.mean() >= 0.0:
        return None
    gen = rng if rng is not None else np.random.default_rng(0)
    stops = np.full(sims, np.inf)
    for i in range(sims):
        res = betting_mean_test(x[gen.integers(0, x.size, n_max)], upper, c, **prior)
        hit = np.nonzero(res.capital >= 1.0 / level)[0]
        if hit.size:
            stops[i] = hit[0] + 1
    finite = stops[np.isfinite(stops)]
    q = np.quantile(stops, power) if np.isfinite(np.quantile(stops, power)) else None
    needed = int(np.ceil(q)) if q is not None else None
    return BettingPlan(
        sessions_needed=needed,
        more_sessions=max(0, needed - sessions_collected) if needed is not None else None,
        p_certify=float(finite.size / sims),
        median_sessions=float(np.median(finite)) if finite.size else None,
        power=power,
        n_max=n_max,
    )


def plan_betting_risk(
    errors_dev: np.ndarray,
    accepted_dev: np.ndarray,
    alpha: float,
    cap: int,
    **kwargs: object,
) -> BettingPlan | None:
    """Planner for `betting_risk_test` (claims A and C) from dev per-session counts."""
    e = np.asarray(errors_dev, dtype=np.float64)
    a = np.asarray(accepted_dev, dtype=np.float64)
    return plan_betting_test(e - alpha * a, cap * (1.0 - alpha), **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class BettingPlanRange:
    """Planner output with its own uncertainty, for reporting to users.

    ``low`` / ``median`` / ``high`` are sessions-needed when planning at the
    optimistic, point and pessimistic estimate of the slice's margin (dev mean of
    X shifted by -z, 0, +z standard errors). ``inf`` means "may not certify
    within ``n_max``". On real traffic with 300 dev sessions, z = 1.28 gave
    70-83% coverage of the true value for a nominal 80%; a bootstrap
    of the planner under-covered badly (30-70%), so it is not used.
    """

    low: float
    median: float
    high: float
    z: float


def plan_betting_range(
    x_dev: np.ndarray,
    upper: float,
    z: float = 1.28,
    **kwargs: object,
) -> BettingPlanRange:
    """Sessions-needed range for the betting test from dev per-session values of X."""
    x = np.asarray(x_dev, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x_dev must be 1-D with at least 2 sessions")
    se = float(x.std(ddof=1) / np.sqrt(x.size))
    out = []
    for shift in (-z * se, 0.0, z * se):
        # shifted values must stay inside the test's support (X <= upper)
        p = plan_betting_test(np.minimum(x + shift, upper), upper, **kwargs)  # type: ignore[arg-type]
        out.append(float("inf") if p is None or p.sessions_needed is None else p.sessions_needed)
    return BettingPlanRange(low=out[0], median=out[1], high=out[2], z=z)
