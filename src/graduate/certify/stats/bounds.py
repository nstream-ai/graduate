"""Exact binomial bounds and tests for error rates on accepted calls."""

from __future__ import annotations

from scipy.stats import beta, binom


def _check_counts(errors: int, n: int) -> None:
    if n < 0 or errors < 0 or errors > n:
        raise ValueError(f"need 0 <= errors <= n, got errors={errors}, n={n}")


def _check_prob(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value}")


def clopper_pearson_upper(errors: int, n: int, delta: float) -> float:
    """One-sided exact (1 - delta) upper confidence bound on the error rate.

    With no observations the only valid bound is 1.0.
    """
    _check_counts(errors, n)
    _check_prob("delta", delta)
    if n == 0 or errors == n:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))


def clopper_pearson_interval(errors: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Two-sided exact confidence interval for a binomial proportion."""
    _check_counts(errors, n)
    _check_prob("level", level)
    if n == 0:
        return 0.0, 1.0
    tail = (1.0 - level) / 2.0
    lower = 0.0 if errors == 0 else float(beta.ppf(tail, errors, n - errors + 1))
    upper = 1.0 if errors == n else float(beta.ppf(1.0 - tail, errors + 1, n - errors))
    return lower, upper


def risk_pvalue(errors: int, n: int, alpha: float) -> float:
    """Exact one-sided p-value for H0: true error rate >= alpha.

    Small p supports "risk is below alpha". P(E <= errors | p = alpha) is the
    worst case over the composite null because the binomial CDF is decreasing
    in p, so the test is valid for every p >= alpha. With n == 0 there is no
    evidence, so p = 1.0.
    """
    _check_counts(errors, n)
    _check_prob("alpha", alpha)
    if n == 0:
        return 1.0
    return float(binom.cdf(errors, n, alpha))
