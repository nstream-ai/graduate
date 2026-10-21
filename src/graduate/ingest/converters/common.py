"""Pieces shared by all converters: tool schemas, system prompt, session assembly."""

from __future__ import annotations

import json
from typing import Any

from graduate.core import (
    Flags,
    Message,
    Outcome,
    Session,
    ToolSchema,
    calls_from_messages,
    stable_hash,
)
from graduate.ingest.repair import split_think, strip_template_tokens


def as_list(value: Any) -> list[Any]:
    """Some datasets store lists as a JSON string, or as a list of JSON strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else []
    return [json.loads(v) if isinstance(v, str) else v for v in value]


def tool_schemas(raw: Any) -> list[ToolSchema]:
    out = []
    for t in as_list(raw):
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if not isinstance(fn, dict) or "name" not in fn:
            continue
        params = fn.get("parameters")
        out.append(
            ToolSchema(
                name=str(fn["name"]),
                description=str(fn.get("description") or ""),
                parameters=params if isinstance(params, dict) else {},
            )
        )
    return out


class Builder:
    """Accumulates messages for one session, cleaning content and counting what was changed."""

    def __init__(self) -> None:
        self.system_parts: list[str] = []
        self.messages: list[Message] = []
        self.template_tokens = 0

    def clean(self, text: Any) -> str:
        cleaned, n = strip_template_tokens(text if isinstance(text, str) else "")
        self.template_tokens += n
        return cleaned

    def system(self, text: Any) -> None:
        t = self.clean(text).strip()
        if t:
            self.system_parts.append(t)

    def add(self, message: Message) -> None:
        if message.role == "assistant" and message.reasoning is None:
            text, reasoning = split_think(message.content)
            message.content, message.reasoning = text, reasoning
        self.messages.append(message)

    def build(
        self,
        *,
        source: str,
        session_id: Any,
        tools: list[ToolSchema],
        outcome: Outcome | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Session:
        system = "\n\n".join(self.system_parts) or None
        tcs = [tc for m in self.messages for tc in m.tool_calls]
        sid = (
            str(session_id)
            if session_id
            else stable_hash([source, [m.model_dump() for m in self.messages]])
        )
        return Session(
            session_id=sid,
            source=source,
            system=system,
            tools=tools,
            messages=self.messages,
            calls=calls_from_messages(self.messages),
            outcome=outcome or Outcome(),
            flags=Flags(
                template_tokens_stripped=self.template_tokens,
                static_context_missing=system is None and not tools,
                tool_calls_failed=sum(tc.parse == "failed" for tc in tcs),
                tool_calls_repaired=sum(tc.parse == "repaired" for tc in tcs),
            ),
            meta=meta or {},
        )
