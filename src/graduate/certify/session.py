"""Session-level claim B: is a candidate (or a hybrid) no worse than the reference, task by task?

Both run the same tasks, possibly several trials each. Per task, regret = reference success rate
- candidate success rate. The claim "regret <= margin" holds when the one-sided upper bound on the
mean regret over tasks is <= margin, taking the stricter of a task-clustered bootstrap bound and
an exact bound conditional on the discordant tasks (tasks where the two differ). Tasks, not
trials, are the unit: repeating trials narrows the interval little when tasks bind.

The minimum detectable regret is the smallest true regret this many tasks could detect at
one-sided 5% with 80% power, (1.645 + 0.84) * SE, with SE from the per-task differences (and at
least 3 / n): a comparison whose margin is below it "cannot be decided with this evidence".

Non-vacuity: "no worse than a reference that rarely succeeds" is true of anything. A comparison is
eligible only when the reference succeeds on at least 2 x margin of the tasks; otherwise the
verdict is "vacuous". Run `compare` per domain as well as pooled. Both bounds are returned and the
binding one (the larger, which certifies) is named.
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
    from scipy.stats import beta

    ids = sorted(t for t in reference if reference[t] and candidate.get(t))
    n = len(ids)
    if not n:
        return {"tasks": 0, "verdict": "no paired tasks"}
    d = [sum(reference[t]) / len(reference[t]) - sum(candidate[t]) / len(candidate[t])
         for t in ids]  # fmt: skip
    mean = sum(d) / n
    rng = random.Random(seed)
    boot = sorted(sum(rng.choice(d) for _ in d) / n for _ in range(n_boot))
    up_boot = boot[min(n_boot - 1, int((1 - level) * n_boot))]
    worse, better = sum(x > 0 for x in d), sum(x < 0 for x in d)
    disc = worse + better
    if disc == 0:
        up_exact = 0.0
    elif worse == disc:
        up_exact = disc / n
    else:
        up_exact = (disc / n) * (2 * float(beta.ppf(1 - level, worse + 1, disc - worse)) - 1)
    upper = max(up_boot, up_exact)
    binding = "bootstrap" if up_boot >= up_exact else "exact"
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 1.0
    detectable = max(3 / n, Z_POWER * sd / math.sqrt(n))
    p_ref = sum(sum(reference[t]) / len(reference[t]) for t in ids) / n
    non_vacuous = p_ref >= 2 * margin
    verdict = (
        f"vacuous (reference succeeds {p_ref:.0%} < 2 x margin)" if not non_vacuous
        else "non-inferior" if upper <= margin
        else "cannot be decided with this evidence" if detectable > margin
        else "not non-inferior"
    )  # fmt: skip
    return {
        "tasks": n, "margin": margin, "regret": mean, "regret_upper": upper,
        "regret_upper_bootstrap": up_boot, "regret_upper_exact": up_exact, "binding": binding,
        "non_vacuous": non_vacuous,
        "tasks_worse": worse, "tasks_better": better, "detectable_regret": detectable,
        "reference_success": sum(sum(reference[t]) / len(reference[t]) for t in ids) / n,
        "candidate_success": sum(sum(candidate[t]) / len(candidate[t]) for t in ids) / n,
        "verdict": verdict,
    }  # fmt: skip
