"""Failure types of a proposed action against the reference, for per-failure-type gates.

A student that is good on average can still be bad in one way that matters (e.g. it adds
fields the reference never sends). Each failure type becomes its own gate in the report card,
and a slice graduates only if every gate passes.

Failure types (a call can have several):
  wrong_decision   tool call vs text reply
  wrong_tool       the multiset of tool names differs
  missing_field    an argument the reference sent is absent
  extra_field      an argument the reference did not send ("hallucinated field")
  wrong_value      a shared argument has a different (normalized) value
  unparsed         arguments could not be parsed on either side

Calls are paired by tool name (parallel calls are a multiset); unpaired calls count as
wrong_tool.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

from graduate.core.canonical import normalize
from graduate.core.schema import ToolCall, ToolSchema

FAILURE_TYPES = (
    "wrong_decision",
    "wrong_tool",
    "missing_field",
    "extra_field",
    "wrong_value",
    "unparsed",
)


def failures(proposed: Sequence[ToolCall], reference: Sequence[ToolCall]) -> set[str]:
    if bool(proposed) != bool(reference):
        return {"wrong_decision"}
    out: set[str] = set()
    by_name: dict[str, list[ToolCall]] = defaultdict(list)
    for r in reference:
        by_name[r.name].append(r)
    for p in proposed:
        if not by_name.get(p.name):
            out.add("wrong_tool")
            continue
        r = by_name[p.name].pop(0)
        if p.arguments is None or r.arguments is None:
            out.add("unparsed")
            continue
        pa, ra = normalize(p.arguments), normalize(r.arguments)
        if set(ra) - set(pa):
            out.add("missing_field")
        if set(pa) - set(ra):
            out.add("extra_field")
        if any(pa[k] != ra[k] for k in set(pa) & set(ra)):
            out.add("wrong_value")
    if any(by_name.values()):
        out.add("wrong_tool")
    return out


def verbose(
    proposed_text: str, reference_text: str, *, ratio: float = 1.5, slack: int = 40
) -> bool:
    """Length gate: the reply is much longer than the reference's (more than `ratio` times plus
    `slack` characters). Plain SFT can make small students verbose and repetitive (Ministral 3,
    arXiv 2601.08584), which costs tokens and latency even when the action is right."""
    return len(proposed_text.strip()) > ratio * len(reference_text.strip()) + slack


def repetitive(text: str, *, n: int = 8, max_share: float = 0.3) -> bool:
    """Loop gate: a large share of word n-grams repeat (degenerate, looping output)."""
    words = text.split()
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return bool(grams) and 1 - len(set(grams)) / len(grams) > max_share


FREE_TEXT_HINTS = (
    "summary", "description", "reason", "message", "note", "comment", "detail", "explanation",
    "issue", "text", "content", "body", "feedback", "instructions", "notes", "narrative",
)  # fmt: skip

# Characters that render like ASCII punctuation or spaces but are different code points.
LOOKALIKES = re.compile(
    "[\\u00a0\\u2000-\\u200f\\u2010-\\u2015\\u2212\\u2028\\u2029\\u202f\\u2060\\ufeff]"
)


def free_text_fields(schema: ToolSchema) -> set[str]:
    """Arguments whose value is prose (graded by meaning), by name and schema hints. Everything
    else (ids, codes, domains, enums, numbers) is graded exactly."""
    props = schema.parameters.get("properties") if isinstance(schema.parameters, dict) else None
    out: set[str] = set()
    for name, spec in (props or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        if spec.get("type", "string") != "string" or "enum" in spec or "format" in spec:
            continue
        text = f"{name} {spec.get('description', '')}".lower()
        if any(h in text for h in FREE_TEXT_HINTS):
            out.add(name)
    return out


def differing_fields(proposed: ToolCall, reference: ToolCall) -> set[str]:
    pa, ra = normalize(proposed.arguments or {}), normalize(reference.arguments or {})
    return {k for k in set(pa) | set(ra) if pa.get(k) != ra.get(k)}


def has_lookalikes(value: object) -> bool:
    return isinstance(value, str) and bool(LOOKALIKES.search(value))


def judge_eligible(proposed: ToolCall, reference: ToolCall, schema: ToolSchema | None) -> bool:
    """A same-tool disagreement the judge may settle: every differing argument is free text.
    Identifier differences (including look-alike characters) stay exact mismatches."""
    if schema is None or proposed.name != reference.name:
        return False
    diff = differing_fields(proposed, reference)
    return bool(diff) and diff <= free_text_fields(schema)
