"""Session-level claim B: is a candidate (or a hybrid) no worse than the reference, task by task?

Both run the same tasks, possibly several trials each. Per task, regret = reference success rate
- candidate success rate. The claim "regret <= margin" holds when the finite-sample task-level
betting bound on the per-task regrets is <= margin; the
task-cluster t bound alone reads "supported (asymptotic only)" and the bootstrap bound is
printed. Tasks, not trials, are the unit. Fewer than 3 / margin tasks: "cannot be decided". A regret
lower bound above the margin reads "worse than the margin". Tasks in one file only are not
paired and are counted.

The minimum detectable regret is the smallest true regret this many tasks could detect at
one-sided 5% with 80% power, (1.645 + 0.84) * SE, with SE from the per-task differences (and at
least 3 / n): a comparison whose margin is below it "cannot be decided with this evidence".

Non-vacuity: "no worse than a reference that rarely succeeds" is true of anything. A comparison is
eligible only when the reference succeeds on at least 2 x margin of the tasks; otherwise the
verdict is "vacuous". Run `compare` per domain as well as pooled.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

Z_POWER = 1.645 + 0.84


def per_task(outcomes: Iterable[tuple[str, bool]]) -> dict[str, list[float]]:
    """(task id, success) pairs -> task id -> list of outcomes."""
    out: dict[str, list[float]] = defaultdict(list)
    for task, ok in outcomes:
        out[str(task)].append(float(ok))
    return out


def compare(
    reference: Mapping[str, Sequence[float]],
    candidate: Mapping[str, Sequence[float]],
    margin: float = 0.10,
    level: float = 0.05,
    seed: int = 0,
    n_boot: int = 10000,
) -> dict:
    from graduate.certify.stats.taskbounds import finite_upper, t_upper

    ids = sorted(t for t in reference if reference[t] and candidate.get(t))
    n = len(ids)
    # tasks in one file only are not paired; they are counted so the caller can say so
    only_ref = sorted(t for t in reference if reference[t] and not candidate.get(t))
    only_cand = sorted(t for t in candidate if candidate[t] and not reference.get(t))
    if not n:
        return {"tasks": 0, "verdict": "no paired tasks", "tasks_only_reference": len(only_ref),
                "tasks_only_candidate": len(only_cand)}  # fmt: skip
    d = [sum(reference[t]) / len(reference[t]) - sum(candidate[t]) / len(candidate[t])
         for t in ids]  # fmt: skip
    mean = sum(d) / n
    rng = random.Random(seed)
    boot = sorted(sum(rng.choice(d) for _ in d) / n for _ in range(n_boot))
    up_boot = boot[min(n_boot - 1, int((1 - level) * n_boot))]  # printed, not used to decide
    worse, better = sum(x > 0 for x in d), sum(x < 0 for x in d)
    # task-level bounds on the per-task regret (magnitudes, not signs); the
    # conditional exact bound (0 when no task differs) was removed
    up_finite = finite_upper(d, level, seed=seed)
    up_t = t_upper(d, level)
    upper = up_finite
    # the other side: a lower bound on the regret (the same finite-sample bound on -d); above the
    # margin, the candidate is shown worse, whatever the number of tasks
    low_finite = -finite_upper([-x for x in d], level, seed=seed)
    # in simulation the t bound held for sessions except with one trial per task at 30-60 tasks
    # (6.4-6.8% at a nominal 5%); there the "supported (asymptotic only)" label is suppressed
    trials = min(len(reference[t]) for t in ids)
    t_valid = n >= 100 or trials >= 2
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 1.0
    detectable = max(3 / n, Z_POWER * sd / math.sqrt(n))
    p_ref = sum(sum(reference[t]) / len(reference[t]) for t in ids) / n
    non_vacuous = p_ref >= 2 * margin
    verdict = (
        f"worse than the margin (regret lower bound {low_finite:+.0%} > {margin:.0%})"
        if low_finite > margin
        else f"cannot be decided with this evidence ({n} tasks < 3 / margin)" if n < 3 / margin
        else f"vacuous (reference succeeds {p_ref:.0%} < 2 x margin)" if not non_vacuous
        else "non-inferior" if up_finite <= margin
        else "supported (asymptotic only)" if up_t <= margin and t_valid
        else "cannot be decided with this evidence" if detectable > margin
        else "non-inferiority not shown (not proven worse)"
    )  # fmt: skip
    return {
        "tasks": n, "margin": margin, "regret": mean, "regret_upper": upper,
        "regret_upper_finite": up_finite, "regret_upper_t": up_t,
        "regret_upper_bootstrap": up_boot, "regret_lower_finite": low_finite,
        "tasks_only_reference": len(only_ref), "tasks_only_candidate": len(only_cand),
        "non_vacuous": non_vacuous,
        "tasks_worse": worse, "tasks_better": better, "detectable_regret": detectable,
        "reference_success": sum(sum(reference[t]) / len(reference[t]) for t in ids) / n,
        "candidate_success": sum(sum(candidate[t]) / len(candidate[t]) for t in ids) / n,
        "verdict": verdict,
    }  # fmt: skip
