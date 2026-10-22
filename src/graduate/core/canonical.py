"""Canonical actions: when two model calls count as "the same action".

An action is either a set of tool calls (order-insensitive: parallel calls have no order) or a
text turn. Arguments are normalized conservatively: keys sorted, surrounding whitespace removed
from strings, integral floats equal to ints. Nothing semantic (dates, units, synonyms) is
normalized here; that is the job of a per-slice equivalence plugin.

Agreement levels, from weakest to strongest:
  decision  tool call vs text
  tools     the same multiset of tool names
  args      the same multiset of (name, normalized arguments)
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from graduate.core.schema import ToolCall

Level = Literal["decision", "tools", "args"]
UNPARSED = "<unparsed>"


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list | tuple):
        return [normalize(v) for v in value]
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(normalize(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Action:
    kind: Literal["tool", "text"]
    tools: tuple[tuple[str, str], ...] = ()  # sorted (name, canonical args json)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(n for n, _ in self.tools)


def action_of(tool_calls: Iterable[ToolCall]) -> Action:
    items = sorted(
        (tc.name, canonical_json(tc.arguments) if tc.arguments is not None else UNPARSED)
        for tc in tool_calls
    )
    return Action("tool", tuple(items)) if items else Action("text")


def agree(a: Action, b: Action, level: Level = "args") -> bool:
    """Do two actions agree at `level`? Unparsed arguments never agree at level "args"."""
    if a.kind != b.kind:
        return False
    if a.kind == "text" or level == "decision":
        return True
    if level == "tools":
        return Counter(a.tool_names) == Counter(b.tool_names)
    if any(args == UNPARSED for _, args in a.tools + b.tools):
        return False
    return Counter(a.tools) == Counter(b.tools)
