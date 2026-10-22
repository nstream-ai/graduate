"""Report card: per-slice certification of a candidate against a reference, on held-out sessions.

Input is one row per model call: slice key, session id, a token weight, and one or more *gates*:
boolean arrays saying, per call, that the candidate failed in a given way (disagreed with the
reference, produced an invalid call, added a field, ...), each with its own allowed rate.

For each slice and gate the certified population is the first `cap` calls per session in arrival
order, tested with the anytime-valid betting test of H0: rate >= alpha (a mixture of
bets).

A slice graduates only if **every** gate passes. This is an intersection-union test: the slice's
p-value is the largest gate p-value, which is valid at level delta for the conjunction without
any correction across gates (Berger 1982). Multiplicity across slices is then Benjamini-Hochberg
(default), Holm or Bonferroni.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from graduate.certify.stats import (
    benjamini_hochberg,
    betting_risk_test,
    bonferroni,
    clopper_pearson_interval,
    holm,
    ratio_upper_bound,
    session_counts,
)

MIXTURE = (0.5, 0.75, 0.9, 0.99)
# Planner: how many more conversations a slice needs, with the card's own test.
PLAN_FUTURES = 200  # simulated futures per slice and level
PLAN_POWER = 0.8  # the estimate is where this share of futures has passed
PLAN_MAX_MORE = 20_000
NEAR_MARGIN = "estimates near the margin are unreliable; this is a range, not a promise."


def more_sessions_needed(
    errors: ArrayLike,
    accepted: ArrayLike,
    alpha: float,
    cap: int,
    level: float,
    futures: int = PLAN_FUTURES,
    power: float = PLAN_POWER,
    max_more: int = PLAN_MAX_MORE,
    seed: int = 0,
) -> int | None:
    """Conversations to add until `power` of simulated futures pass the card's betting test at
    `level`. A future keeps the real conversations in order and appends whole conversations
    resampled with replacement (so the within-conversation dependence the test sees is kept;
    resampling single calls would assume independent calls). None: more than `max_more`."""
    e = np.asarray(errors, dtype=np.float64)
    a = np.asarray(accepted, dtype=np.float64)
    n = e.size
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    if (e == e[0]).all() and (a == a[0]).all():
        futures = 1  # every conversation alike (e.g. no disagreement): every future is the same
    horizon = max(64, 2 * n)
    while True:
        firsts = np.full(futures, np.inf)
        for f in range(futures):
            pick = rng.integers(0, n, size=horizon)
            path = betting_risk_test(np.concatenate([e, e[pick]]), np.concatenate([a, a[pick]]),
                                     alpha, cap, c=MIXTURE).capital  # fmt: skip
            hit = np.flatnonzero(path >= 1.0 / level)
            if hit.size:
                firsts[f] = max(0, int(hit[0]) + 1 - n)
        q = float(np.sort(firsts)[int(np.ceil(power * futures)) - 1])  # inf-safe quantile
        if np.isfinite(q):
            return int(np.ceil(q))
        if horizon >= max_more:
            return None
        horizon = min(max_more, horizon * 4)


def slice_plan(counts: tuple, gate: GateResult, cap: int, delta: float, m: int) -> dict | None:
    """The planner's range for one slice (None at or above alpha: more evidence cannot help)."""
    e, a = counts
    if not e.size or not gate.rate < gate.alpha:
        return None
    low = more_sessions_needed(e, a, gate.alpha, cap, delta)
    high = more_sessions_needed(e, a, gate.alpha, cap, delta / max(1, m))
    return {"at_sessions": int(e.size), "rate": gate.rate, "low": low, "high": high,
            "criterion": f"{PLAN_POWER:.0%} of simulated futures pass",
            "near_margin": near_margin(gate.rate, gate.alpha, int(a.sum()))}  # fmt: skip


def plan_text(plan: dict) -> str:
    def n(x: int | None) -> str:
        return f"more than {PLAN_MAX_MORE:,}" if x is None else f"~{max(1, x):,}"

    lo, hi = plan["low"], plan["high"]
    span = f"at least {n(lo)}" + (f", up to {n(hi)}" if hi != lo else "")
    text = f"not enough evidence: {span} more conversations ({plan['criterion']})"
    return text + ("; " + NEAR_MARGIN if plan.get("near_margin") else "")


def above_gate(rate: float, population: int, alpha: float, delta: float) -> bool:
    """The disagreement rate is significantly at or above alpha: the one-sided 1 - delta lower
    Clopper-Pearson bound reaches alpha. Only then can more conversations not help; an observed
    rate above alpha on a thin slice (1 of 5) is still undecided."""
    if population <= 0:
        return False
    errors = round(rate * population)
    return clopper_pearson_interval(errors, population, 1 - 2 * delta)[0] >= alpha


def near_margin(rate: float, alpha: float, population: int) -> bool:
    """The observed rate is within two standard errors of alpha (its own standard error: a rate
    of 0 is not noisy, every simulated future then agrees)."""
    if population <= 0 or rate <= 0:
        return False
    return alpha - rate <= 2 * np.sqrt(rate * (1 - rate) / population)


ADJUST = {"bh": benjamini_hochberg, "holm": holm, "bonferroni": bonferroni}


@dataclass(frozen=True)
class GateResult:
    alpha: float
    rate: float
    upper: float
    pvalue: float
    feasible: bool = True  # False: even zero failures could not pass with this much evidence


@dataclass(frozen=True)
class SliceResult:
    slice: str
    cap: int
    sessions: int
    calls: int  # all calls in the slice
    certified_population: int  # calls inside the per-session cap
    gates: dict[str, GateResult]
    binding: str  # the gate with the largest p-value: what stops (or last allowed) graduation
    pvalue: float  # max over gates (intersection-union)
    certified: bool
    call_share: float  # share of all calls the slice routes (inside the cap), if certified
    token_share: float
    # planner, for a slice not certified with its observed rate under alpha: conversations to
    # add, "low" at the uncorrected level delta (the most BH can give one slice) and "high" at
    # delta / m (the BH worst case), each where PLAN_POWER of simulated futures pass
    plan: dict | None = None

    @property
    def rate(self) -> float:
        return self.gates[self.binding].rate

    @property
    def upper(self) -> float:
        return self.gates[self.binding].upper


@dataclass(frozen=True)
class ReportCard:
    alphas: dict[str, float]
    delta: float
    cap: int | dict[str, int]
    multiplicity: str
    slices: list[SliceResult]
    calls: int
    certified_call_share: float
    certified_token_share: float

    @property
    def alpha(self) -> float:
        return min(self.alphas.values())

    def render(self, top: int = 25) -> str:
        multi = len(self.alphas) > 1
        head = (
            f"{'slice':<52}{'cap':>4}{'sess':>6}{'calls':>7}{'rate':>7}{'upper':>7}"
            f"{'detect':>7}{'p':>8}{'share':>7}  verdict"
        )
        rows = [head]
        for r in sorted(self.slices, key=lambda r: -r.calls)[:top]:
            verdict = "GRADUATED" if r.certified else "-"
            if multi and not r.certified:
                verdict += f"  (binding: {r.binding})"
            if r.plan:
                verdict += f"  [{plan_text(r.plan)}]"
            elif not r.certified and r.rate >= self.alphas[r.binding]:
                a = self.alphas[r.binding]
                if above_gate(r.rate, r.certified_population, a, self.delta):
                    verdict += (
                        f"  [disagreement {r.rate:.0%} is significantly above alpha {a:g}: "
                        "more conversations will not help]"
                    )
                else:
                    verdict += (
                        f"  [cannot be decided with this evidence: disagreement {r.rate:.0%} is "
                        f"not significantly above alpha {a:g}]"
                    )
            detect = 3 / r.sessions if r.sessions else 1.0  # smallest rate n can rule out
            rows.append(
                f"{r.slice[:51]:<52}{r.cap:>4}{r.sessions:>6}{r.calls:>7}{r.rate:>7.3f}"
                f"{r.upper:>7.3f}{detect:>7.3f}{r.pvalue:>8.3f}{r.call_share:>7.3f}  {verdict}"
            )
        caps = "per slice (dev)" if isinstance(self.cap, dict) else f"{self.cap}"
        gates = ", ".join(f"{g} <= {a}" for g, a in self.alphas.items())
        rows.append(
            f"\ngates: {gates}; delta {self.delta}; cap {caps} per session; "
            f"{self.multiplicity} across {len(self.slices)} slices; {self.calls:,} calls"
        )
        rows.append(
            f"certified coverage: {self.certified_call_share:.1%} of calls, "
            f"{self.certified_token_share:.1%} of tokens"
        )
        return "\n".join(rows)


def within_cap(session_ids: Sequence[str], cap: int) -> NDArray[np.bool_]:
    """True for the first `cap` rows of each session (rows in arrival order)."""
    seen: Counter[str] = Counter()
    keep = np.zeros(len(session_ids), dtype=bool)
    for i, s in enumerate(session_ids):
        keep[i] = seen[s] < cap
        seen[s] += 1
    return keep


def choose_caps(
    slice_keys: Sequence[str], session_ids: Sequence[str], quantile: float = 0.95
) -> dict[str, int]:
    """Per-slice router cap from *dev* data: a high quantile of calls per session in the slice.

    The betting test's bet size scales with 1 / cap, so a cap far above what sessions actually
    contribute wastes power. Must be chosen on data not used to certify.
    """
    counts: dict[str, Counter[str]] = {}
    for k, s in zip(slice_keys, session_ids, strict=True):
        counts.setdefault(k, Counter())[s] += 1
    return {
        k: max(1, int(np.ceil(np.quantile(list(c.values()), quantile)))) for k, c in counts.items()
    }


def report_card(
    slice_keys: Sequence[str],
    session_ids: Sequence[str],
    wrong: ArrayLike | None = None,
    tokens: ArrayLike | None = None,
    *,
    gates: Mapping[str, ArrayLike] | None = None,
    accepted: ArrayLike | None = None,
    alpha_scale: Mapping[str, float] | None = None,
    alpha: float | Mapping[str, float] = 0.05,
    delta: float = 0.05,
    cap: int | Mapping[str, int] = 20,
    default_cap: int = 20,
    multiplicity: str = "bh",
    min_sessions: int = 1,
    plan: bool = True,
) -> ReportCard:
    """Rows must be in arrival order (sessions in time order, calls in session order).

    Pass either `wrong` (one gate, named "disagree") or `gates` (name -> per-call failure flags).
    `accepted` marks the calls the router would hand to the candidate (e.g. a validity shield or
    a confidence threshold); the others go to the reference model, so they are outside the
    certified population and do not count as coverage, but stay in the denominators.
    `alpha_scale` multiplies the allowed rates of named slices (e.g. 1 - U for slices whose
    labels came from a judge with miss-rate bound U).
    `alpha` is one allowed rate for every gate or a per-gate mapping. `cap` is one cap for all
    slices or a per-slice mapping (from `choose_caps` on dev data); slices missing from the
    mapping use `default_cap`. Coverage counts only calls inside the cap.
    """
    if (wrong is None) == (gates is None):
        raise ValueError("pass exactly one of `wrong` or `gates`")
    gate_rows = {"disagree": wrong} if gates is None else dict(gates)
    flags = {g: np.asarray(v, dtype=bool) for g, v in gate_rows.items()}
    alphas = (
        {g: float(alpha[g]) for g in flags}
        if isinstance(alpha, Mapping)
        else dict.fromkeys(flags, float(alpha))
    )
    keys = np.asarray(slice_keys, dtype=object)
    sess = np.asarray(session_ids, dtype=object)
    tok = np.ones(len(keys)) if tokens is None else np.asarray(tokens, dtype=np.float64)
    if any(len(f) != len(keys) for f in flags.values()) or not len(keys) == len(sess) == len(tok):
        raise ValueError("all inputs must have one entry per call")
    if multiplicity not in ADJUST:
        raise ValueError(f"multiplicity must be one of {sorted(ADJUST)}")
    total_calls, total_tokens = len(keys), float(tok.sum()) or 1.0
    acc = np.ones(len(keys), dtype=bool) if accepted is None else np.asarray(accepted, bool)
    if len(acc) != len(keys):
        raise ValueError("accepted must have one entry per call")
    pre = []
    for name in dict.fromkeys(keys.tolist()):
        k = cap.get(name, default_cap) if isinstance(cap, Mapping) else cap
        idx = np.flatnonzero((keys == name) & acc)
        if idx.size == 0:
            continue
        _, codes = np.unique(sess[idx], return_inverse=True)
        results: dict[str, GateResult] = {}
        counts: dict[str, tuple] = {}
        n_sessions, population = 0, 0
        for g, f in flags.items():
            e, a = session_counts(codes, np.ones(idx.size, dtype=bool), f[idx], k)
            keep = a > 0
            e, a = e[keep], a[keep]
            n_sessions, population = int(e.size), int(a.sum())
            a_eff = alphas[g] * (alpha_scale or {}).get(name, 1.0)
            if e.size >= min_sessions:
                p = betting_risk_test(e, a, a_eff, k, c=MIXTURE).pvalue
                upper = ratio_upper_bound(e, a, k, delta)
            else:
                p, upper = 1.0, 1.0
            rate = float(e.sum() / a.sum()) if a.sum() else float("nan")
            # The evidence floor (about 3 / n): with these sessions, could a perfect candidate pass?
            best = ratio_upper_bound(np.zeros_like(a), a, k, delta) if a.size else 1.0
            results[g] = GateResult(a_eff, rate, float(upper), float(p), feasible=best <= a_eff)
            counts[g] = (e, a)
        binding = max(results, key=lambda g: results[g].pvalue)
        routed = idx[within_cap(sess[idx].tolist(), k)]
        pre.append((name, idx, routed, k, n_sessions, population, results, binding, counts))
    pvals = np.array([x[6][x[7]].pvalue for x in pre]) if pre else np.zeros(0)
    passed = ADJUST[multiplicity](pvals, delta)
    out = [
        SliceResult(
            slice=str(name),
            cap=int(k),
            sessions=n_sessions,
            calls=int(idx.size),
            certified_population=population,
            gates=results,
            binding=binding,
            pvalue=results[binding].pvalue,
            certified=bool(ok),
            call_share=routed.size / total_calls,
            token_share=float(tok[routed].sum()) / total_tokens,
            plan=(
                slice_plan(counts[binding], results[binding], k, delta, len(pre))
                if plan and not ok
                else None
            ),
        )
        for (name, idx, routed, k, n_sessions, population, results, binding, counts), ok in zip(
            pre, passed, strict=True
        )
    ]
    return ReportCard(
        alphas=alphas,
        delta=delta,
        cap=dict(cap) if isinstance(cap, Mapping) else cap,
        multiplicity=multiplicity,
        slices=out,
        calls=total_calls,
        certified_call_share=sum(r.call_share for r in out if r.certified),
        certified_token_share=sum(r.token_share for r in out if r.certified),
    )
