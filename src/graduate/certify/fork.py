"""Claim B by effect: certify a candidate from fork outcomes, per slice and per session.

A fork starts from a state the reference model saw in a real conversation. Three branches run to
the end of the conversation in a replayable environment:

  T1  the reference answers the turn and finishes the conversation
  T2  the same again (a second continuation: the reference's own effect-level noise)
  S   the candidate answers the turn, the reference finishes

Each branch ends in a task outcome (pass / fail). Per slice (by the candidate's proposed action):

- signed net harm h = P(T1 pass and S fail) - P(T1 fail and S pass), with the stricter of a
  task-clustered bootstrap bound and an exact bound conditional on the discordant states;
- certified iff Benjamini-Hochberg across slices rejects h >= delta_turn, the bound <= delta_turn,
  n >= 3 / delta_turn, the slice is non-vacuous (P(T1 pass) >= 2 delta_turn) and the reference
  agrees with itself (h_T = T2 vs T1 within 1.645 clustered standard errors of 0);
- never-route iff the harm bound exceeds half the reference's own success there;
- the minimum detectable harm with this evidence, delta_min = (1.645 + 0.84) * sqrt(change_T / n)
  (and at least 3 / n): a slice whose delta_min exceeds delta_turn "cannot be decided with this
  evidence", which is not a failure;
- per session: delta_turn = session margin / cap, so the harm a session can accumulate over its
  <= cap routed turns is bounded by cap * the per-turn bound.

Rows (one per branch): {"state", "branch": "T1"|"T2"|"S", "task_id", "slice", "success": bool|None,
"termination", optional "proposed_slice"}. Rows with success None or an excluded termination are
left out, never scored.
"""

from __future__ import annotations

import math
from collections import defaultdict

BRANCHES = ("T1", "T2", "S")
EXCLUDED = ("infrastructure_error", "replay_error", "error", "budget_stop")
Z_POWER = 1.645 + 0.84  # one-sided alpha 5%, power 80%


def pooled(slice_key: str) -> str:
    """domain/kind/tool with the position bucket dropped (phase B: position is a covariate)."""
    return "/".join(slice_key.split("/")[:3])


def pair_rows(rows: list[dict]) -> list[dict]:
    """One record per state with every branch's pass/fail; states with an excluded branch drop."""
    by_state: dict = defaultdict(dict)
    meta = {}
    for r in rows:
        by_state[r["state"]][r["branch"]] = r
        if r["branch"] == "S":
            meta[r["state"]] = r
    out = []
    for sid, br in by_state.items():
        if set(BRANCHES) - set(br):
            continue
        if any(br[b].get("termination") in EXCLUDED or br[b].get("success") is None for b in br):
            continue
        s = meta[sid]
        out.append(
            {
                "state": sid,
                "task_id": s["task_id"],
                "slice": s["slice"],
                "proposed_slice": s.get("proposed_slice") or s["slice"],
                "agree_args": s.get("first_agree_args"),
                **{b: bool(br[b]["success"]) for b in BRANCHES},
            }
        )
    return out


def _cluster_boot(diff, tasks, seed: int, n_boot: int = 5000):
    """Bootstrap of mean(diff) resampling whole tasks (forks of one task are correlated)."""
    import numpy as np

    _, code = np.unique(np.asarray(tasks), return_inverse=True)
    code = code.reshape(-1)
    m = int(code.max()) + 1
    sums = np.bincount(code, weights=diff, minlength=m)
    counts = np.bincount(code, minlength=m).astype(float)
    pick = np.random.default_rng(seed).integers(0, m, size=(n_boot, m))
    return sums[pick].sum(axis=1) / counts[pick].sum(axis=1)


def states_needed(change_t: float, delta: float) -> int:
    """States needed per slice at 80% power: n >= max(3 / delta, (1.645 + 0.84)^2 * change_T /
    delta^2), the exact zero-event bound, or the one-sided normal bound when the reference's own
    flips (change_T) add variance; those flips are variance, not bias."""
    return math.ceil(max(3 / delta, Z_POWER**2 * change_t / delta**2))


def slice_stats(pairs: list[dict], delta: float = 0.05, level: float = 0.05, seed: int = 0) -> dict:
    """Per-slice effect statistics for one group of paired states.

    Primary: signed net harm h = mean(worse - better) of S against T1, one-sided task-clustered
    upper bound <= delta (delta_turn = session margin / router cap). The teacher's own flip rate
    change_T is not in the bound: it sizes the slice, checks validity (h_T, T2 against T1, must be
    within noise of 0) and marks slices unresolvable at this budget."""
    import numpy as np

    from graduate.certify.stats.bounds import (
        clopper_pearson_interval,
        clopper_pearson_upper,
        risk_pvalue,
    )

    n = len(pairs)
    t1 = np.array([p["T1"] for p in pairs], dtype=float)
    t2 = np.array([p["T2"] for p in pairs], dtype=float)
    s = np.array([p["S"] for p in pairs], dtype=float)
    tasks = [p["task_id"] for p in pairs]
    ch_t, ch_s = int((t1 != t2).sum()), int((t1 != s).sum())
    worse, better = int(((t1 == 1) & (s == 0)).sum()), int(((t1 == 0) & (s == 1)).sum())
    boot = _cluster_boot(t1 - s, tasks, seed)  # bootstrap of h
    boot_t = _cluster_boot(t1 - t2, tasks, seed + 1)  # bootstrap of h_T
    boot_upper = float(np.quantile(boot, 1 - level))
    # exact, state-level: given the d = worse + better discordant states,
    # worse ~ Binomial(d, q) and h = (d / n)(2q - 1); a Clopper-Pearson upper bound on q bounds h
    d = worse + better
    # no discordant state: the harm is at most the discordance rate, whose exact zero-event bound
    # is not 0
    exact_upper = (
        (d / n) * (2 * clopper_pearson_upper(worse, d, level) - 1) if d
        else clopper_pearson_upper(0, n, level)
    )  # fmt: skip
    upper = max(boot_upper, exact_upper)  # the stricter certifies
    binding = "bootstrap" if boot_upper >= exact_upper else "exact"
    h_t_ci = (float(np.quantile(boot_t, 0.025)), float(np.quantile(boot_t, 0.975)))
    h_t, h_t_se = float((t1 - t2).mean()), float(boot_t.std())
    need = states_needed(ch_t / n, delta)
    # the smallest harm this evidence could detect (one-sided 5%, power 80%)
    detectable = max(3 / n, Z_POWER * math.sqrt(ch_t / n / n)) if n else 1.0
    return {
        "n_states": n,
        "n_tasks": len(set(tasks)),
        "pass_T1": float(t1.mean()),
        "pass_T2": float(t2.mean()),
        "pass_S": float(s.mean()),
        "worse": worse,
        "better": better,
        "harm": (worse - better) / n,
        "harm_upper": upper,
        "harm_upper_bootstrap": boot_upper,
        "harm_upper_exact": exact_upper,
        "binding": binding,
        # one-sided bootstrap p-value for H0: h >= delta (for BH across slices)
        "p_harm": float((1 + (boot >= delta).sum()) / (1 + boot.size)),
        "delta": delta,
        "change_T": ch_t / n,
        "change_T_ci": clopper_pearson_interval(ch_t, n),
        "change_S": ch_s / n,
        "change_S_ci": clopper_pearson_interval(ch_s, n),
        # diagnostic: the candidate adds no more randomness than the teacher
        "change_S_within_floor": ch_s / n <= ch_t / n + 0.02,
        "h_T": h_t,
        "h_T_ci95": h_t_ci,
        "h_T_se_clustered": h_t_se,
        # validity of the reference noise floor: h_T (T2 against T1) must be within noise, tested
        # one-sided 5% each way with tasks as clusters, not compared as a point
        "valid_h_T": abs(h_t) <= 1.645 * h_t_se if h_t_se > 0 else h_t == 0.0,
        "states_needed": need,
        "detectable_harm": detectable,
        "resolvable": n >= need,
        # reference only: exact state-level bound on P(worse) alone
        "p_worse_exact": risk_pvalue(worse, n, delta),
    }


def fork_card(
    pairs: list[dict],
    session_margin: float = 0.10,
    cap: int = 2,
    level: float = 0.05,
    min_states: int = 30,  # smaller slices stay out of the BH family (reported as untested)
    serving: dict | None = None,
) -> dict:
    """Card by the reference's logged slice and by the candidate's proposed slice (what a router
    sees). Verdicts: certified; excluded by evidence (never route); vacuous; cannot be decided with
    this evidence (detectable harm above delta_turn: not a failure); not certified. `serving`
    records the reference's serving configuration: its self-noise floor is a property of it."""
    from graduate.certify.stats.multiplicity import benjamini_hochberg

    delta = session_margin / cap
    card: dict = {"session_margin": session_margin, "cap": cap, "delta_turn": delta,
                  "level": level, "n_states": len(pairs)}  # fmt: skip
    for key in ("slice", "proposed_slice"):
        groups: dict = defaultdict(list)
        for p in pairs:
            groups[p[key]].append(p)
        table = {
            k: slice_stats(v, delta, level)
            for k, v in sorted(groups.items())
            if len(v) >= min_states
        }
        names = list(table)
        bh = benjamini_hochberg([table[k]["p_harm"] for k in names], q=level) if names else []
        for k, ok in zip(names, bh, strict=True):
            t = table[k]
            t["p_T"] = t["pass_T1"]
            # non-vacuity: a slice is eligible only where the reference itself succeeds >= 2 delta;
            # "no worse than a reference that rarely succeeds" is true of anything
            t["eligible"] = t["p_T"] >= 2 * delta
            # never route: harm is likely (h > 0) and could exceed half the reference's own success
            # (upper bound > p_T / 2): the router refuses the slice
            t["never_route"] = t["harm_upper"] > t["p_T"] / 2 and t["harm"] > 0
            t["certified"] = (
                bool(ok)
                and t["eligible"]
                and t["harm_upper"] <= delta
                and t["n_states"] >= 3 / delta
                and t["valid_h_T"]
            )
            t["session_harm_upper"] = min(1.0, cap * t["harm_upper"])  # <= cap routed turns
            t["decidable"] = t["detectable_harm"] <= delta
            t["verdict"] = (
                "certified" if t["certified"]
                else "excluded by evidence (never route)" if t["never_route"]
                else f"vacuous (reference succeeds {t['p_T']:.0%})" if not t["eligible"]
                else (f"cannot be decided with this evidence (detectable harm "
                      f"{t['detectable_harm']:.1%} > {delta:.1%})") if not t["decidable"]
                else "not certified"
            )  # fmt: skip
        card[f"by_{key}"] = table
        card[f"untested_{key}"] = sorted(k for k in groups if k not in table)
    card["all"] = slice_stats(pairs, delta, level) if pairs else None
    if card["all"]:  # the reference's own effect-level noise: the floor of every claim here
        card["reference_floor"] = {k: card["all"][k] for k in ("h_T", "h_T_ci95", "change_T")}
    card["serving"] = serving or {}
    # the largest router cap for which the per-turn evidence still supports the session margin by
    # the union bound: floor(margin / pooled per-turn upper bound); beyond it the session claim
    # derived from per-turn evidence is not informative
    turn_upper = card["all"]["harm_upper"] if card["all"] else None
    card["cap_max_informative"] = (
        None if turn_upper is None
        else math.floor(session_margin / turn_upper) if turn_upper > 0
        else None  # no harm bound above zero: no limit from this evidence
    )  # fmt: skip
    limit = card["cap_max_informative"]
    card["composition"] = (
        f"union bound over <= {cap} routed turns per session (session harm <= cap x per-turn "
        "bound); with this per-turn bound the session claim is informative up to cap "
        f"{limit if limit is not None else 'unbounded'}; a dependence-aware composition is "
        "research, not used here"
    )
    card["requires"] = "an environment that can be reset to a logged state (replayable forks)"
    return card


def card_pairs(rows: list[dict], args) -> list[dict]:
    pairs = pair_rows(rows)
    if getattr(args, "pool_positions", False):
        for p in pairs:
            p["slice"], p["proposed_slice"] = pooled(p["slice"]), pooled(p["proposed_slice"])
    return pairs


def routing_table(card: dict) -> dict:
    """The hybrid agent's routing shape; a certified slice is used at any confidence (tau 0)."""
    cert = [k for k, t in card["by_proposed_slice"].items() if t["certified"]]
    never = [k for k, t in card["by_proposed_slice"].items() if t.get("never_route")]
    return {
        "tau": {k: 0.0 for k in cert},
        "certified": cert,
        "never": never,  # never-route slices: refused regardless of confidence
        "nodes": {},
        "measure": "mean",
        "source": "fork_compare",
        "margin": f"h <= {card['delta_turn']:g}",
        "session_margin": card["session_margin"],
        "cap": card["cap"],
    }
