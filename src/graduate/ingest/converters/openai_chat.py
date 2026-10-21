"""OpenAI-style chat messages, including the variants seen in open datasets:

* `tool_calls` with arguments as a JSON string or as an object (Nemotron, SWE-rebench)
* flat `{"name", "arguments"}` entries in `tool_calls` (tau2-bench)
* legacy `function_call` + role `function`, one assistant message per call (Toucan)
* roles `tool_call` / `tool_response` with a Python-repr dict as content (Toucan SFT)
* `reasoning_content` next to the content (GPT-OSS, Nemotron); `<think>` blocks in content
"""

from __future__ import annotations

from typing import Any

from graduate.core import Message, Outcome, Session, ToolCall
from graduate.ingest.converters.common import Builder, as_list, tool_schemas
from graduate.ingest.repair import make_tool_call, tool_call_from_text

TOOL_RESULT_ROLES = {"tool", "function", "tool_response", "observation", "ipython"}
USER_ROLES = {"user", "human"}
ASSISTANT_ROLES = {"assistant", "ai", "gpt", "model"}


def _tool_calls(m: dict[str, Any]) -> list[ToolCall]:
    out = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", tc)
        out.append(make_tool_call(fn.get("name"), fn.get("arguments"), id=tc.get("id")))
    fc = m.get("function_call")
    if fc:
        out.append(make_tool_call(fc.get("name"), fc.get("arguments")))
    return out


def convert_messages(
    messages: Any,
    *,
    source: str,
    session_id: Any = None,
    tools: Any = None,
    outcome: Outcome | None = None,
    meta: dict[str, Any] | None = None,
) -> Session:
    b = Builder()
    for m in as_list(messages):
        role = str(m.get("role", "")).lower()
        content = b.clean(m.get("content"))
        if role in ("system", "developer"):
            b.system(content)
        elif role in USER_ROLES:
            b.add(Message(role="user", content=content))
        elif role == "tool_call":
            b.add(Message(role="assistant", tool_calls=[tool_call_from_text(content)]))
        elif role in ASSISTANT_ROLES:
            b.add(
                Message(
                    role="assistant",
                    content=content,
                    reasoning=b.clean(m.get("reasoning_content")) or None,
                    tool_calls=_tool_calls(m),
                )
            )
        elif role in TOOL_RESULT_ROLES:
            b.add(
                Message(
                    role="tool",
                    content=content,
                    tool_call_id=m.get("tool_call_id"),
                    name=m.get("name"),
                )
            )
        else:
            raise ValueError(f"unknown role {role!r} in {source}")
    return b.build(
        source=source, session_id=session_id, tools=tool_schemas(tools), outcome=outcome, meta=meta
    )
