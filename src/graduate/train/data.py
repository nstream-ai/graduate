"""Supervised examples from sessions: one example per teacher model call (F8.1, per-call SFT).

The input is everything the model saw (static context + history); the target is the call's
output, merged into one assistant message. Messages use the OpenAI chat format so any Hugging
Face chat template can render them. Pure Python: no torch needed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from graduate.core import Message, Session

Chat = list[dict[str, Any]]


@dataclass(frozen=True)
class SFTExample:
    session_id: str
    call_index: int
    messages: Chat  # input, including the system message
    tools: list[dict[str, Any]]
    target: dict[str, Any]  # one assistant message
    meta: dict[str, Any] = field(default_factory=dict)


def _tool_call(tc: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": tc.name, "arguments": tc.arguments or {}}}


def to_chat(messages: Iterable[Message], *, reasoning: bool = False) -> Chat:
    out: Chat = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.role == "assistant":
            if m.tool_calls:
                d["tool_calls"] = [_tool_call(t) for t in m.tool_calls]
            if reasoning and m.reasoning:
                d["reasoning_content"] = m.reasoning
            if out and out[-1]["role"] == "assistant":  # merge the parts of one model call
                prev = out[-1]
                prev["content"] = (prev["content"] + d["content"]).strip()
                prev["tool_calls"] = prev.get("tool_calls", []) + d.get("tool_calls", [])
                if not prev["tool_calls"]:
                    del prev["tool_calls"]
                continue
        elif m.role == "tool":
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.name:
                d["name"] = m.name
        out.append(d)
    return out


def tools_spec(session: Session) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            },
        }
        for t in session.tools
    ]


def build_examples(
    sessions: Iterable[Session],
    *,
    include_failed: bool = False,
    include_repaired: bool = True,
    reasoning: bool = False,
) -> tuple[list[SFTExample], dict[str, int]]:
    """Examples for every teacher call. Calls whose target has an unparsable tool call are
    skipped by default (a student must not learn malformed calls); repaired ones are kept and
    can be excluded. Returns the examples and counts of what was skipped."""
    out: list[SFTExample] = []
    skipped = {"student_produced": 0, "failed_target": 0, "repaired_target": 0,
               "unknown_tool_target": 0}  # fmt: skip
    for s in sessions:
        system = [{"role": "system", "content": s.system}] if s.system else []
        tools = tools_spec(s)
        for c in s.calls:
            if c.produced_by != "teacher":
                skipped["student_produced"] += 1
                continue
            tcs = s.call_tool_calls(c)
            if not include_failed and any(t.parse == "failed" for t in tcs):
                skipped["failed_target"] += 1
                continue
            if not include_repaired and any(t.parse == "repaired" for t in tcs):
                skipped["repaired_target"] += 1
                continue
            # A call to a tool the agent does not have (τ² telecom: the teacher calling the
            # user's device tools) is a teacher mistake, not something to teach.
            names = {t.name for t in s.tools}
            if not include_failed and names and any(t.name not in names for t in tcs):
                skipped["unknown_tool_target"] += 1
                continue
            (target,) = to_chat(s.call_output(c), reasoning=reasoning)
            out.append(
                SFTExample(
                    session_id=s.session_id,
                    call_index=c.index,
                    messages=system + to_chat(s.call_input(c)),
                    tools=tools,
                    target=target,
                    meta={"node": s.agent_node, "source": s.source},
                )
            )
    return out, skipped
