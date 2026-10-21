"""Session-aware sampling to avoid overstating evidence from correlated steps.

Certified population: per slice, the router lets the student handle at
most ``cap`` calls per session; later calls go to the teacher. The population a
certificate covers is therefore "the first ``cap`` accepted calls of each
session, in time order", and the per-call risk is E[errors] / E[accepted] over
that population. Rows passed in must be in time order within each session.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def one_step_per_session(
    sessions: ArrayLike,
    rng: np.random.Generator,
    groups: ArrayLike | None = None,
) -> NDArray[np.intp]:
    """Indices of one uniformly random row per (group, session), sorted ascending.

    ``groups`` is typically the slice id, so each slice keeps one step per session.
    """
    s = np.asarray(sessions)
    if s.ndim != 1:
        raise ValueError("sessions must be 1-D")
    if s.size == 0:
        return np.zeros(0, dtype=np.intp)
    _, s_code = np.unique(s, return_inverse=True)
    if groups is None:
        key = s_code.reshape(-1)
    else:
        g = np.asarray(groups)
        if g.shape != s.shape:
            raise ValueError("groups must match sessions in shape")
        _, g_code = np.unique(g, return_inverse=True)
        key = g_code.reshape(-1) * (int(s_code.max()) + 1) + s_code.reshape(-1)
    order = np.lexsort((rng.random(s.size), key))
    sorted_key = key[order]
    first = np.append(True, sorted_key[1:] != sorted_key[:-1])
    picked: NDArray[np.intp] = np.sort(order[first]).astype(np.intp, copy=False)
    return picked


def _capped_population(
    sessions: ArrayLike, accepted: ArrayLike, cap: int
) -> tuple[NDArray[np.intp], NDArray[np.intp], int]:
    """Rows in the certified population and each row's session arrival rank.

    Returns (rank per input row, population row indices sorted by session, m).
    """
    if cap < 1:
        raise ValueError("cap must be >= 1")
    s = np.asarray(sessions)
    acc = np.asarray(accepted, dtype=bool)
    if s.shape != acc.shape or s.ndim != 1:
        raise ValueError("sessions and accepted must be 1-D arrays of equal length")
    if s.size == 0:
        return np.zeros(0, np.intp), np.zeros(0, np.intp), 0
    _, first, code = np.unique(s, return_index=True, return_inverse=True)
    arrival = np.argsort(np.argsort(first))  # session code -> arrival rank
    rank = arrival[code.reshape(-1)].astype(np.intp)
    idx = np.nonzero(acc)[0]
    order = idx[np.argsort(rank[idx], kind="stable")]  # by session, time order kept
    r = rank[order]
    position = np.arange(order.size) - np.searchsorted(r, r, side="left")
    return rank, order[position < cap], int(first.size)


def session_counts(
    sessions: ArrayLike,
    accepted: ArrayLike,
    wrong: ArrayLike,
    cap: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Per-session (errors, accepted) counts over the certified population.

    Sessions are returned in arrival order (first appearance). Sessions with
    nothing accepted are kept with a = 0.
    """
    bad = np.asarray(wrong, dtype=bool)
    if bad.shape != np.asarray(sessions).shape:
        raise ValueError("wrong must match sessions in shape")
    rank, keep, m = _capped_population(sessions, accepted, cap)
    a = np.bincount(rank[keep], minlength=m).astype(np.int64)
    e = np.bincount(rank[keep], weights=bad[keep], minlength=m).astype(np.int64)
    return e, a


def thinned_one_per_session(
    sessions: ArrayLike,
    accepted: ArrayLike,
    cap: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """At most one population row per session, each population call equally likely.

    A session with a' population calls is kept with probability a' / cap, then
    one of its population calls is chosen uniformly, so every call in the
    population is sampled with probability 1 / cap. Sessions are independent, so
    the sampled rows are i.i.d. draws of a per-call outcome and the exact
    binomial test bounds the per-call risk. Plain one-step-per-session sampling
    instead weights sessions equally, which is biased when long sessions differ.
    """
    rank, keep, m = _capped_population(sessions, accepted, cap)
    if keep.size == 0:
        return np.zeros(0, np.intp)
    r = rank[keep]
    pick = keep[np.lexsort((rng.random(keep.size), r))]
    first = np.append(True, rank[pick][1:] != rank[pick][:-1])
    chosen = pick[first]
    size = np.bincount(r, minlength=m)[rank[chosen]]
    kept = chosen[rng.random(chosen.size) < size / cap]
    out: NDArray[np.intp] = np.sort(kept).astype(np.intp, copy=False)
    return out
