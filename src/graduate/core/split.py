"""Session-level splits for training and certification.

Splits are by session, never by call (calls in a session are dependent). When timestamps exist
the split is time-ordered: train on the past, certify on later traffic. Without timestamps a
stable hash order is used and the split says so (`time_ordered=False`), because a random split
overstates how well a student does on future traffic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from graduate.core.canonical import stable_hash
from graduate.core.schema import Session

DEFAULT_FRACTIONS = {"train": 0.7, "dev": 0.1, "calibration": 0.2}


@dataclass(frozen=True)
class Split:
    parts: dict[str, list[Session]]
    time_ordered: bool

    def __getitem__(self, name: str) -> list[Session]:
        return self.parts[name]


def split_sessions(
    sessions: Sequence[Session],
    fractions: dict[str, float] | None = None,
    salt: str = "",
    group: Callable[[Session], str] | None = None,
) -> Split:
    """`group` keeps related sessions together (e.g. repeated trials of one benchmark task, which
    would otherwise leak between train and held-out); parts are then filled with whole groups, so
    their sizes match `fractions` only approximately."""
    fractions = fractions or DEFAULT_FRACTIONS
    if abs(sum(fractions.values()) - 1) > 1e-9 or any(f < 0 for f in fractions.values()):
        raise ValueError("fractions must be non-negative and sum to 1")
    ids = [s.session_id for s in sessions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate session_id")
    time_ordered = bool(sessions) and all(s.ts is not None for s in sessions)
    if time_ordered:
        order = sorted(sessions, key=lambda s: (s.ts, s.session_id))
    else:
        order = sorted(sessions, key=lambda s: stable_hash([salt, s.session_id]))
    names = list(fractions)
    if group is not None:
        return Split(_split_groups(order, fractions, group, salt), time_ordered)
    parts: dict[str, list[Session]] = {}
    start, cum = 0, 0.0
    for i, name in enumerate(names):
        cum += fractions[name]
        end = len(order) if i == len(names) - 1 else round(cum * len(order))
        parts[name] = list(order[start:end])
        start = end
    return Split(parts, time_ordered)


def _split_groups(
    order: list[Session], fractions: dict[str, float], group: Callable[[Session], str], salt: str
) -> dict[str, list[Session]]:
    groups: dict[str, list[Session]] = {}
    for s in order:  # groups keep the order of their first session
        groups.setdefault(str(group(s)), []).append(s)
    if not any(s.ts for s in order):
        groups = dict(sorted(groups.items(), key=lambda kv: stable_hash([salt, "group", kv[0]])))
    names, n = list(fractions), len(order)
    bounds, cum = [], 0.0
    for name in names:
        cum += fractions[name]
        bounds.append(cum * n)
    parts: dict[str, list[Session]] = {name: [] for name in names}
    seen = 0
    for members in groups.values():
        mid = seen + len(members) / 2  # a group goes to the part holding its midpoint
        i = next((j for j, b in enumerate(bounds) if mid < b), len(names) - 1)
        parts[names[i]].extend(members)
        seen += len(members)
    return parts


@dataclass(frozen=True)
class Round:
    index: int
    train: list[Session]
    held_out: list[Session]
    time_ordered: bool


def rolling_rounds(
    sessions: Sequence[Session], rounds: int = 3, window: float = 0.15, salt: str = ""
) -> list[Round]:
    """Rotating held-out windows: round r trains on everything before window r and certifies on
    window r, so each round's exam is traffic the student has never seen (Karpathy's private,
    rotating eval). Windows are the last `rounds` slices of size `window` in time order
    (stable hash order, flagged, when there are no timestamps)."""
    if rounds < 1 or not 0 < window * rounds < 1:
        raise ValueError("need rounds >= 1 and 0 < window * rounds < 1")
    order = split_sessions(sessions, {"all": 1.0}, salt)
    seq, n = order["all"], len(sessions)
    out = []
    for r in range(rounds):
        start = round(n * (1 - window * (rounds - r)))
        end = round(n * (1 - window * (rounds - r - 1)))
        out.append(Round(r, list(seq[:start]), list(seq[start:end]), order.time_ordered))
    return out


def group_holdout(
    sessions: Sequence[Session], held_out_groups: set[str], key: str = "agent_node"
) -> tuple[list[Session], list[Session]]:
    """Hold out whole groups (e.g. an agent or app absent from training): a diagnostic of
    whether the student learned the interface grammar or memorized the training agents. Not a
    certificate: certificates are for traffic like the calibration window."""
    train: list[Session] = []
    test: list[Session] = []
    for s in sessions:
        (test if str(getattr(s, key)) in held_out_groups else train).append(s)
    return train, test
