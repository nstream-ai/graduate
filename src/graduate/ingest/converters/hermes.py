"""ShareGPT conversations (`from` / `value`) with Hermes `<tool_call>` blocks in the text."""

from __future__ import annotations

import re
from typing import Any

from graduate.core import Message, Outcome, Session
from graduate.ingest.converters.common import Builder, as_list, tool_schemas
from graduate.ingest.repair import tool_call_from_text

TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
TOOL_RESPONSE = re.compile(r"^\s*<tool_response>\s*(.*?)\s*</tool_response>\s*$", re.S)
ROLES = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}


def convert_sharegpt(
    conversations: Any,
    *,
    source: str,
    session_id: Any = None,
    tools: Any = None,
    outcome: Outcome | None = None,
    meta: dict[str, Any] | None = None,
) -> Session:
    b = Builder()
    for turn in as_list(conversations):
        who = ROLES.get(str(turn.get("from", "")).lower())
        value = b.clean(turn.get("value"))
        if who is None:
            raise ValueError(f"unknown speaker {turn.get('from')!r} in {source}")
        if who == "system":
            b.system(value)
        elif who == "user":
            b.add(Message(role="user", content=value))
        elif who == "assistant":
            calls = [tool_call_from_text(raw) for raw in TOOL_CALL.findall(value)]
            b.add(
                Message(
                    role="assistant", content=TOOL_CALL.sub("", value).strip(), tool_calls=calls
                )
            )
        else:
            m = TOOL_RESPONSE.match(value)
            b.add(Message(role="tool", content=m.group(1) if m else value))
    return b.build(
        source=source, session_id=session_id, tools=tool_schemas(tools), outcome=outcome, meta=meta
    )
