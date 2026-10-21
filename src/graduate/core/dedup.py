"""Duplicate and near-duplicate call states, within training data and across splits.

The *state* of a call is what the model saw just before answering. Two notions:
  exact  hash of the full input (static context + every message), after canonicalization
  near   MinHash/LSH over word 5-gram shingles of the recent context (last user turn and last
         tool result), with digits normalized, so templated sessions that differ only in ids
         or amounts are found (Broder 1997; Lee et al. 2022 for why it matters in training)

Agent traffic repeats legitimately: a held-out state that also appears in training is not
contamination when the held-out data is *later* traffic, because production will repeat it too.
It is contamination when the split is not time-ordered. So the leakage report is always printed,
and certification can be re-run without leaked states to show how much the result depends on
them; nothing is dropped silently.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from graduate.core.canonical import stable_hash
from graduate.core.schema import Call, Session

_PRIME = (1 << 31) - 1
_WORD = re.compile(r"[a-z0-9]+")


def exact_state(session: Session, call: Call) -> str:
    return stable_hash(
        [
            session.system or "",
            [t.model_dump() for t in session.tools],
            [m.model_dump(exclude={"reasoning"}) for m in session.call_input(call)],
        ]
    )


def recent_text(session: Session, call: Call) -> str:
    hist = session.call_input(call)
    user = next((m.content for m in reversed(hist) if m.role == "user"), "")
    tool = next((m.content for m in reversed(hist) if m.role == "tool"), "")
    return f"{user}\n{tool}"


def _normalize(text: str) -> list[str]:
    return _WORD.findall(re.sub(r"\d", "0", text.lower()))


def shingles(text: str, k: int = 5) -> set[int]:
    words = _normalize(text)
    if len(words) < k:
        grams = [" ".join(words)] if words else []
    else:
        grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
    return {
        int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") for g in grams
    }


class MinHasher:
    """MinHash with affine permutations modulo the Mersenne prime 2^31 - 1 (vectorized; the
    products fit in int64)."""

    def __init__(self, num_perm: int = 64, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _PRIME, num_perm, dtype=np.int64)[:, None]
        self.b = rng.integers(0, _PRIME, num_perm, dtype=np.int64)[:, None]
        self.num_perm = num_perm

    def signature(self, items: set[int]) -> NDArray[np.int64]:
        if not items:
            return np.full(self.num_perm, _PRIME, dtype=np.int64)
        x = np.fromiter((i % _PRIME for i in items), dtype=np.int64)[None, :]
        return np.asarray(((self.a * x + self.b) % _PRIME).min(axis=1), dtype=np.int64)


def jaccard_estimate(s1: NDArray[np.int64], s2: NDArray[np.int64]) -> float:
    return float(np.mean(s1 == s2))


@dataclass(frozen=True)
class LeakageReport:
    held_out_calls: int
    exact_in_train: int
    near_in_train: int
    train_calls: int
    train_exact_duplicates: int  # train calls whose exact state appears earlier in train
    leaked: frozenset[tuple[str, int]]  # (session_id, call index) of held-out calls with a twin

    def render(self) -> str:
        def pct(a: int, b: int) -> str:
            return f"{a:,} ({100 * a / b:.1f}%)" if b else "0"

        return "\n".join(
            [
                f"train calls with an exact duplicate state in train   "
                f"{pct(self.train_exact_duplicates, self.train_calls)}",
                f"held-out calls whose exact state is in train          "
                f"{pct(self.exact_in_train, self.held_out_calls)}",
                f"held-out calls with a near-duplicate state in train   "
                f"{pct(self.near_in_train, self.held_out_calls)}",
            ]
        )


def _calls(sessions: Iterable[Session]) -> list[tuple[Session, Call]]:
    return [(s, c) for s in sessions for c in s.calls]


def leakage(
    train: Sequence[Session],
    held_out: Sequence[Session],
    *,
    threshold: float = 0.9,
    num_perm: int = 64,
    bands: int = 16,
) -> LeakageReport:
    """Exact and near-duplicate states of held-out calls in the training data."""
    if num_perm % bands:
        raise ValueError("num_perm must be divisible by bands")
    rows = num_perm // bands
    tr, ho = _calls(train), _calls(held_out)
    seen: set[str] = set()
    train_dups = 0
    for s, c in tr:
        h = exact_state(s, c)
        train_dups += h in seen
        seen.add(h)
    mh = MinHasher(num_perm)
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    sigs = []
    for i, (s, c) in enumerate(tr):
        sig = mh.signature(shingles(recent_text(s, c)))
        sigs.append(sig)
        for band in range(bands):
            buckets[(band, sig[band * rows : (band + 1) * rows].tobytes())].append(i)
    exact = near = 0
    leaked = set()
    for s, c in ho:
        is_exact = exact_state(s, c) in seen
        sig = mh.signature(shingles(recent_text(s, c)))
        cands = {
            j
            for band in range(bands)
            for j in buckets.get((band, sig[band * rows : (band + 1) * rows].tobytes()), [])
        }
        is_near = is_exact or any(jaccard_estimate(sig, sigs[j]) >= threshold for j in cands)
        exact += is_exact
        near += is_near
        if is_near:
            leaked.add((s.session_id, c.index))
    return LeakageReport(len(ho), exact, near, len(tr), train_dups, frozenset(leaked))


def dedupe_sessions(sessions: Iterable[Session]) -> tuple[list[Session], int]:
    """Drop sessions whose full content (static context + messages) repeats an earlier one."""
    out, seen, dropped = [], set(), 0
    for s in sessions:
        h = stable_hash(
            [
                s.system or "",
                [t.model_dump() for t in s.tools],
                [m.model_dump() for m in s.messages],
            ]
        )
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        out.append(s)
    return out, dropped
