"""Trace stores: OpenTelemetry GenAI spans and MLflow traces, turned into OpenAI chat messages.

A trace store keeps one span per model call, and each model-call span carries the whole history
the model was given. The last model-call span of an agent therefore holds that agent's session:
its input messages plus its output are the history, and every earlier assistant turn in it is a
call. One session per agent (`gen_ai.agent.name`), so a multi-agent trace gives one session per
agent node.

Message shapes, sniffed per value (MLflow re-maps other tracers onto the same keys):
- OpenAI chat: `[{"role", "content", "tool_calls"}]` or `{"messages": [...]}`;
- OTel GenAI parts: `[{"role", "parts": [{"type": "text" | "tool_call" | "tool_call_response" |
  "reasoning", ...}]}]`;
- Gemini: `{"contents": [{"role": "user" | "model", "parts": [{"text"} | {"functionCall"} |
  {"functionResponse"}]}], "system_instruction", "tools"}`.
Attribute values may be JSON strings, Python-literal strings (some exporters write `str(dict)`),
or already parsed; OTLP JSON keeps them as `[{"key", "value": {"stringValue": ...}}]`.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from typing import Any

from graduate.core import Session, Usage
from graduate.ingest.converters.openai_chat import convert_messages


def value(v: Any) -> Any:
    """An attribute value, parsed when it is a JSON or Python-literal string."""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{\"'" and s:
            try:
                return json.loads(s)
            except ValueError:
                try:
                    return ast.literal_eval(s)
                except (ValueError, SyntaxError):
                    return v
        return v
    return v


def _otlp_attrs(attrs: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for a in attrs:
        v = a.get("value") or {}
        out[a.get("key")] = next(iter(v.values()), None) if isinstance(v, dict) else v
    return out


def spans(record: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    """(name, attributes, trace id) of every span in a record, in start order where known.

    Accepts an OTLP JSON export (`resourceSpans`), a span tree (`root_span` with `children`), an
    MLflow trace (`data.spans`), a row with a `spans` list, or one span."""
    if isinstance(record.get("trace"), (str, dict)):  # an exported table with a trace column
        record = value(record["trace"])
    out: list[tuple[Any, str, dict[str, Any], str]] = []

    def add(s: dict[str, Any], start: Any = None) -> None:
        attrs = value(s.get("attributes") or {})
        if isinstance(attrs, list):
            attrs = _otlp_attrs(attrs)
        ctx = s.get("context") or {}
        tid = str(s.get("trace_id") or s.get("traceId") or ctx.get("trace_id") or "")
        key = s.get("start_time") or s.get("startTimeUnixNano") or start
        out.append((key, str(s.get("name") or ""), attrs if isinstance(attrs, dict) else {}, tid))

    if "resourceSpans" in record:
        for rs in record["resourceSpans"]:
            for ss in rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []:
                for s in ss.get("spans", []):
                    add(s)
    elif "root_span" in record:
        n = 0

        def walk(s: dict[str, Any]) -> None:
            nonlocal n
            add(s, n)
            n += 1
            for c in s.get("children") or []:
                walk(c)

        walk(record["root_span"])
    else:
        listed = (record.get("data") or {}).get("spans") or record.get("spans")
        for i, s in enumerate(value(listed) if listed is not None else [record]):
            add(s, i)
    if all(k is not None for k, *_ in out):
        out.sort(key=lambda t: str(t[0]) if not isinstance(t[0], (int, float)) else f"{t[0]:020}")
    return [(name, attrs, tid) for _, name, attrs, tid in out]


def _text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def _args(v: Any) -> str:
    v = value(v)
    return v if isinstance(v, str) else json.dumps(v if v is not None else {}, ensure_ascii=False)


def from_parts(msgs: Any) -> list[dict[str, Any]]:
    """OTel GenAI parts messages to OpenAI chat messages."""
    out: list[dict[str, Any]] = []
    for m in msgs or []:
        role = str(m.get("role", "")).lower()
        texts, calls, reasoning = [], [], []
        for p in m.get("parts") or []:
            kind = p.get("type")
            if kind == "text":
                texts.append(_text(p.get("content")))
            elif kind in ("reasoning", "thinking"):
                reasoning.append(_text(p.get("content")))
            elif kind == "tool_call":
                calls.append(
                    {
                        "id": p.get("id"),
                        "type": "function",
                        "function": {"name": p.get("name"), "arguments": _args(p.get("arguments"))},
                    }
                )
            elif kind == "tool_call_response":
                res = p.get("response", p.get("result"))
                out.append({"role": "tool", "tool_call_id": p.get("id"), "content": _text(res)})
        if role in ("assistant", "model"):
            out.append(
                {
                    "role": "assistant",
                    "content": "\n".join(texts),
                    "tool_calls": calls,
                    **({"reasoning_content": "\n".join(reasoning)} if reasoning else {}),
                }
            )
        elif texts:
            out.append(
                {
                    "role": "system" if role == "system" else role or "user",
                    "content": "\n".join(texts),
                }
            )
    return out


def from_gemini(contents: Any) -> list[dict[str, Any]]:
    """Gemini `contents` to OpenAI chat messages."""
    out: list[dict[str, Any]] = []
    for c in contents or []:
        role = "assistant" if c.get("role") == "model" else "user"
        texts, calls = [], []
        for p in c.get("parts") or []:
            fc = p.get("functionCall") or p.get("function_call")
            fr = p.get("functionResponse") or p.get("function_response")
            if fc:
                calls.append(
                    {
                        "id": fc.get("id"),
                        "type": "function",
                        "function": {"name": fc.get("name"), "arguments": _args(fc.get("args"))},
                    }
                )
            elif fr:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": fr.get("id"),
                        "name": fr.get("name"),
                        "content": _text(fr.get("response")),
                    }
                )
            elif "text" in p and not p.get("thought"):
                texts.append(_text(p["text"]))
        if role == "assistant":
            out.append({"role": "assistant", "content": "\n".join(texts), "tool_calls": calls})
        elif texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


def gemini_tools(tools: Any) -> list[dict[str, Any]]:
    out = []
    for t in tools or []:
        decls = t.get("function_declarations") or t.get("functionDeclarations")
        out.extend(decls if decls else [t])
    return out


def sniff(v: Any) -> tuple[list[dict[str, Any]], list[Any], str | None]:
    """(OpenAI messages, tools, system) from a request in any of the three shapes."""
    v = value(v)
    if isinstance(v, dict) and "contents" in v:  # Gemini request
        si = v.get("system_instruction") or v.get("systemInstruction") or {}
        system = "\n".join(_text(p.get("text")) for p in si.get("parts") or []) or None
        cfg = v.get("config") or {}
        return from_gemini(v["contents"]), gemini_tools(v.get("tools") or cfg.get("tools")), system
    if isinstance(v, dict) and "messages" in v:  # OpenAI chat request
        return sniff(v["messages"])[0], v.get("tools") or [], None
    if isinstance(v, list) and v and isinstance(v[0], dict):
        if any("parts" in m for m in v):
            return from_parts(v), [], None
        return [m for m in v if isinstance(m, dict) and m.get("role")], [], None
    return [], [], None


def output(v: Any) -> list[dict[str, Any]]:
    """The model's answer as OpenAI chat messages, from any of the three shapes."""
    v = value(v)
    if isinstance(v, dict) and v.get("choices"):
        return [v["choices"][0].get("message") or {}]
    if isinstance(v, dict) and v.get("candidates"):
        return from_gemini([v["candidates"][0].get("content") or {}])
    if isinstance(v, dict) and v.get("role"):
        return [v]
    if isinstance(v, list):
        return sniff(v)[0]
    return []


def session(
    history: list[dict[str, Any]],
    answer: list[dict[str, Any]],
    *,
    source: str,
    session_id: Any,
    tools: Any,
    system: str | None,
    agent: str | None,
    usage: Usage | None,
    meta: dict[str, Any],
) -> Session:
    if not history and not answer:
        raise ValueError(
            "the model-call span has no messages: content capture is off or redacted "
            "(e.g. OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, or an empty llm_request)"
        )
    msgs = ([{"role": "system", "content": system}] if system else []) + history + answer
    s = convert_messages(msgs, source=source, session_id=session_id, tools=tools, meta=meta)
    s.agent_node = agent
    if s.calls and usage:
        s.calls[-1].usage = usage  # the span's usage is the last call's
    return s


def last_per_agent(model_spans: list[tuple[str | None, dict[str, Any]]]) -> Iterator[Any]:
    """The last model-call span of each agent, in the order the agents first appear."""
    last: dict[str | None, dict[str, Any]] = {}
    for agent, attrs in model_spans:
        last[agent] = attrs
    yield from last.items()
