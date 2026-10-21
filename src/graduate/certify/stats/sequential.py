"""Anytime-valid risk test by betting, over independent sessions.

Each session contributes X_i = e_i - alpha * a_i, where a_i is the number of
accepted steps (capped at ``cap``) and e_i the errors among them. The step-level
risk among accepted calls is E[e] / E[a], so "risk < alpha" is exactly
E[X] < 0. Sessions are independent, so steps inside a session may be arbitrarily
correlated without breaking validity.

Under H0 (E[X] >= 0) the capital K_t = prod(1 - lambda_i X_i), with predictable
lambda_i in [0, c / U] and U the largest possible X, is a non-negative
supermartingale. By Ville's inequality P(sup_t K_t >= 1/delta) <= delta, so
p = 1 / max_t K_t is valid however often the evidence is inspected, which is
what shadow mode needs. Reference: Waudby-Smith & Ramdas (2023), "Estimating
means of bounded random variables by betting"; Shafer & Vovk, game-theoretic
probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class BettingTest:
    """Result of the sequential test. ``capital`` is the path, one entry per session."""

    pvalue: float
    capital: NDArray[np.float64]
    n_sessions: int

    @property
    def final_pvalue(self) -> float:
        """Valid p-value using only the end of the path (also anytime-valid)."""
        return float(min(1.0, 1.0 / self.capital[-1])) if self.capital.size else 1.0


def betting_mean_test(
    x: ArrayLike,
    upper: float,
    c: float | tuple[float, ...] = 0.75,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    prior_weight: float = 1.0,
) -> BettingTest:
    """Anytime-valid test of H0: E[X] >= 0 for independent per-session X_i <= ``upper``.

    Only the upper bound matters for validity: bets lambda_i lie in [0, c / upper],
    so 1 - lambda_i X_i >= 1 - c > 0 and, under H0, E[1 - lambda_i X_i] <= 1.
    ``prior_mean`` / ``prior_var`` seed the aGRAPA plug-in estimates and count as
    ``prior_weight`` pseudo-sessions. Like any predictable choice they change
    power, never validity, so prior knowledge (other slices, other deployments)
    can be injected safely; a wrong prior only costs time.

    ``c`` may be a tuple of aggressiveness levels: the capital is then the
    equal-weight average of one aGRAPA bettor per level (a mixture of test
    supermartingales is a test supermartingale, so validity is unchanged). It
    costs at most a factor len(c) of capital versus the best level in hindsight
    and adapts to whether bets are limited by the bound or by the estimates.
    """
    levels = (c,) if isinstance(c, float | int) else tuple(c)
    if not levels or any(not 0.0 < lv < 1.0 for lv in levels):
        raise ValueError("every aggressiveness level c must be in (0, 1)")
    if upper <= 0.0:
        raise ValueError("upper must be > 0 (otherwise H0 is trivially false)")
    if prior_var <= 0.0:
        raise ValueError("prior_var must be > 0")
    if prior_weight <= 0.0:
        raise ValueError("prior_weight must be > 0")
    xs = np.asarray(x, dtype=np.float64)
    if xs.ndim != 1:
        raise ValueError("x must be 1-D")
    if (xs > upper + 1e-12).any():
        raise ValueError("every x must be <= upper")
    n = xs.size
    if n == 0:
        return BettingTest(1.0, np.ones(0), 0)

    # Predictable plug-in estimates from sessions before i, with priors.
    prev_sum = np.concatenate(([0.0], np.cumsum(xs)[:-1]))
    prev_sq = np.concatenate(([0.0], np.cumsum(xs * xs)[:-1]))
    t = np.arange(n, dtype=np.float64)
    w = prior_weight
    mu = (w * prior_mean + prev_sum) / (t + w)
    second = (w * (prior_var + prior_mean * prior_mean) + prev_sq) / (t + w)
    var = np.maximum(second - mu * mu, 1e-12)
    raw = -mu / (var + mu * mu)

    logs = np.stack([np.cumsum(np.log1p(-np.clip(raw, 0.0, lv / upper) * xs)) for lv in levels])
    log_cap = np.logaddexp.reduce(logs, axis=0) - np.log(len(levels))
    capital = np.exp(log_cap)
    return BettingTest(float(min(1.0, np.exp(-log_cap.max()))), capital, n)


def _check_counts(top: NDArray[np.float64], a: NDArray[np.float64], cap: int) -> None:
    if top.shape != a.shape or top.ndim != 1:
        raise ValueError("count arrays must be 1-D and of equal length")
    if (a < 0).any() or (a > cap).any():
        raise ValueError("need 0 <= accepted <= cap per session")


def betting_risk_test(
    errors: ArrayLike,
    accepted: ArrayLike,
    alpha: float,
    cap: int,
    c: float | tuple[float, ...] = 0.75,
    prior_mean: float | None = None,
    prior_var: float | None = None,
    prior_weight: float = 1.0,
) -> BettingTest:
    """Anytime-valid test of H0: per-call risk among accepted >= alpha.

    ``errors`` and ``accepted`` are per-session counts in arrival order, with
    ``0 <= errors <= accepted <= cap``. Equivalent to ``betting_mean_test`` on
    X = e - alpha * a. The same test bounds a per-call *disagreement* rate
    (claim A) when ``errors`` counts disagreements. Priors are on the scale of X
    per session; defaults reproduce the original weakly informative choice.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    e = np.asarray(errors, dtype=np.float64)
    a = np.asarray(accepted, dtype=np.float64)
    _check_counts(e, a, cap)
    if (e < 0).any() or (e > a).any():
        raise ValueError("need 0 <= errors <= accepted per session")
    return betting_mean_test(
        e - alpha * a,
        cap * (1.0 - alpha),
        c,
        prior_mean=-alpha / 2.0 if prior_mean is None else prior_mean,
        prior_var=alpha if prior_var is None else prior_var,
        prior_weight=prior_weight,
    )


def paired_noninferiority_test(
    student_worse: ArrayLike,
    student_better: ArrayLike,
    accepted: ArrayLike,
    margin: float,
    cap: int,
    c: float = 0.75,
) -> BettingTest:
    """Anytime-valid test of H0: p_b - p_c >= margin, per call among accepted (claim B).

    Per session: ``student_worse`` = b (student wrong, teacher right),
    ``student_better`` = c (student right, teacher wrong), ``accepted`` = a.
    Only disagreements need labels: when student and teacher agree, both are
    right or both wrong and contribute 0. Small p certifies "the student is not
    worse than the teacher by ``margin`` or more" and credits cases where the
    student fixes the teacher.
    """
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin must be in [0, 1)")
    b = np.asarray(student_worse, dtype=np.float64)
    cc = np.asarray(student_better, dtype=np.float64)
    a = np.asarray(accepted, dtype=np.float64)
    _check_counts(b, a, cap)
    _check_counts(cc, a, cap)
    if (b < 0).any() or (cc < 0).any() or (b + cc > a).any():
        raise ValueError("need b, c >= 0 and b + c <= accepted per session")
    return betting_mean_test(
        (b - cc) - margin * a,
        cap * (1.0 - margin),
        c,
        prior_mean=-max(margin, 1e-3) / 2.0,
        prior_var=max(margin, 1e-3),
    )


def session_rate_test(
    counts: ArrayLike,
    beta: float,
    cap: int,
    c: float = 0.75,
) -> BettingTest:
    """Anytime-valid test of H0: E[count per session] >= beta.

    With counts = routed disagreements per session (capped), small p certifies
    E[D] < beta, which by the coupling bound gives
    P(hybrid session fails) <= P(teacher-only session fails) + beta.
    """
    if beta <= 0.0 or beta >= cap:
        raise ValueError("need 0 < beta < cap")
    d = np.asarray(counts, dtype=np.float64)
    if d.ndim != 1 or (d < 0).any() or (d > cap).any():
        raise ValueError("counts must be 1-D with 0 <= count <= cap")
    return betting_mean_test(d - beta, cap - beta, c, prior_mean=-beta / 2.0, prior_var=beta)


def parity_test(
    y: ArrayLike,
    accepted: ArrayLike,
    margin: float,
    cap: int,
    c: float | tuple[float, ...] = 0.75,
) -> BettingTest:
    """Anytime-valid test of H0: parity statistic >= margin, per call (first-principles Idea 2).

    Per audited call, draw two student samples S1, S2 and two teacher samples
    T1, T2 and compute y = 2 k(S1,T1)' - k(S1,S2)' - k(T1,T2)' where k' = 1 - k is
    a disagreement in [0, 1] (hard: 1[different]; soft: 1 - similarity). E[y] is
    the kernel two-sample statistic 2 d_ST - d_SS - d_TT: 0 when the student's
    action distribution equals the teacher's, whatever the number of acceptable
    actions. ``y`` here is the per-session sum (in [-2a, 2a]); ``accepted`` = a.
    Small p certifies "indistinguishable from the teacher up to ``margin``".
    """
    if not 0.0 <= margin < 2.0:
        raise ValueError("margin must be in [0, 2)")
    ys = np.asarray(y, dtype=np.float64)
    a = np.asarray(accepted, dtype=np.float64)
    _check_counts(ys, a, cap)
    if (np.abs(ys) > 2 * a + 1e-12).any():
        raise ValueError("per-session y must lie in [-2a, 2a]")
    return betting_mean_test(
        ys - margin * a,
        cap * (2.0 - margin),
        c,
        prior_mean=-max(margin, 1e-3) / 2.0,
        prior_var=max(margin, 1e-3),
    )


_BOUND_GRID = np.unique(np.concatenate([np.geomspace(1e-4, 1.0, 400), [0.0, 1.0]]))


def ratio_upper_bound(
    num: ArrayLike,
    den: ArrayLike,
    cap: int,
    delta: float = 0.05,
    c: float | tuple[float, ...] = (0.5, 0.75, 0.9, 0.99),
) -> float:
    """Upper confidence bound on sum(num) / sum(den) over independent sessions.

    Inverts the anytime-valid betting test of H0: ratio >= r over a fine grid and
    returns the smallest grid value r it certifies (rounded up to the grid, so
    conservative). With num = errors and den = accepted calls this is the
    certificate's "per-call error <= U (1 - delta)". Valid for any within-session
    dependence, because sessions are the unit.
    """
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    if n.shape != d.shape or n.ndim != 1:
        raise ValueError("num and den must be 1-D arrays of equal length")
    if (n < 0).any() or (n > d).any() or (d > cap).any():
        raise ValueError("need 0 <= num <= den <= cap per session")
    for r in _BOUND_GRID[1:]:
        if r >= 1.0:
            return 1.0
        res = betting_mean_test(
            n - r * d, cap * (1.0 - r), c, prior_mean=-r / 2.0, prior_var=max(r, 1e-4)
        )
        if res.pvalue <= delta:
            return float(r)
    return 1.0


def ratio_confidence_interval(
    num: ArrayLike,
    den: ArrayLike,
    cap: int,
    delta: float = 0.05,
    c: float | tuple[float, ...] = (0.5, 0.75, 0.9, 0.99),
) -> tuple[float, float]:
    """Two-sided (1 - delta) interval for sum(num) / sum(den) (delta/2 per side)."""
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    upper = ratio_upper_bound(n, d, cap, delta / 2, c)
    lower = 1.0 - ratio_upper_bound(d - n, d, cap, delta / 2, c)
    return max(0.0, lower), upper
