"""The conversation in the OpenAI chat format: the messages and tools a candidate is shown.

Each model call's parts are merged into one assistant message, so any OpenAI-compatible endpoint
(or chat template) sees what the logged model saw.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graduate.core import Message, Session

Chat = list[dict[str, Any]]


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
