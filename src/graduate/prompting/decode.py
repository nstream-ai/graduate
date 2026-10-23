"""Turn a student's generated text back into an action, for the report card.

Handles the common open-model formats: Hermes / Qwen `<tool_call>{...}</tool_call>` blocks, the
Qwen3-Coder / Qwen3.5 XML form (`<tool_call><function=name><parameter=k>v</parameter>...`, vLLM's
`qwen3_coder` parser) and a bare JSON object with "name" and "arguments". Anything else is a text
reply. Unparsable calls are kept as failed calls, so they count against the student (L0 gate).

XML parameter values are untyped text; given the tool schemas they are typed as the schema says
(as vLLM's parser does), so "19122" stays a string for a string field and becomes 3 for an
integer one. Without schemas only JSON arrays, objects, booleans and null are decoded.
"""

from __future__ import annotations

import json
import re
from typing import Any

from graduate.core import ToolCall
from graduate.ingest import tool_call_from_text

TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
THINK = re.compile(r"<think>.*?</think>", re.S)
FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?(?:</parameter>|(?=<parameter=)|$)", re.S)


def _schemas(tools: Any) -> dict[str, dict[str, Any]]:
    """name -> JSON-schema properties, from OpenAI-style tool dicts or objects with .parameters."""
    out: dict[str, dict[str, Any]] = {}
    for t in tools or []:
        if isinstance(t, dict):
            f = t.get("function", t)
            name, params = f.get("name"), f.get("parameters") or {}
        else:
            name, params = getattr(t, "name", None), getattr(t, "parameters", None) or {}
        if name:
            out[name] = params.get("properties") or {}
    return out


def _typed(value: str, schema: dict[str, Any] | None) -> Any:
    typ = (schema or {}).get("type")
    if isinstance(typ, list):
        typ = next((x for x in typ if x != "null"), None)
    if typ == "string":
        return value
    try:
        v = json.loads(value)
    except ValueError:
        return value
    if typ is None and not (isinstance(v, list | dict | bool) or v is None):
        return value  # a bare number without a schema stays text (ids, zip codes)
    return v


def _xml_call(raw: str, schemas: dict[str, dict[str, Any]]) -> ToolCall:
    m = FUNCTION.search(raw)
    if not m:
        return ToolCall(name="?", raw=raw, parse="failed")
    name = m.group(1).strip()
    props = schemas.get(name, {})
    args = {k.strip(): _typed(v, props.get(k.strip())) for k, v in PARAM.findall(m.group(2))}
    closed = "</function>" in raw
    return ToolCall(name=name, arguments=args, raw=raw, parse="ok" if closed else "failed")


def _gemma_calls(body: str) -> tuple[str, list[ToolCall]]:
    from graduate.prompting import gemma

    calls = []
    for name, args in gemma.CALL.findall(body):
        try:
            calls.append(ToolCall(name=name, arguments=gemma.parse_arguments(args), raw=args))
        except (ValueError, IndexError):
            calls.append(ToolCall(name=name, raw=args, parse="failed"))
    if "<|tool_call>" in body and not calls:  # opened but never closed: a truncated call
        m = re.search(r"<\|tool_call>call:([^{\s]+)", body)
        calls.append(ToolCall(name=m.group(1) if m else "?", raw=body, parse="failed"))
    reply = re.sub(r"<\|tool_call>.*?(<tool_call\|>|$)", "", body, flags=re.S)
    return reply.replace("<turn|>", "").strip(), calls


def parse_completion(text: str, tools: Any = None) -> tuple[str, list[ToolCall]]:
    """(reply text, tool calls) from generated text: Hermes/Qwen `<tool_call>` blocks (JSON or
    Qwen3.5 XML), Gemma 4 `<|tool_call>call:name{...}<tool_call|>`, or a bare JSON call. `tools`
    (the turn's tool schemas) types XML parameter values."""
    if "<|tool_call>" in text or "<|channel>" in text:
        from graduate.prompting.gemma import strip_thoughts

        return _gemma_calls(strip_thoughts(text))
    body = THINK.sub("", text)
    if (
        "</think>" in body
    ):  # thinking opened by the prompt (enable_thinking): only the close is here
        body = body.rsplit("</think>", 1)[1]
    body = body.strip()
    raws = TOOL_CALL.findall(body)
    if raws:
        schemas = _schemas(tools)
        calls = [
            _xml_call(r, schemas) if r.lstrip().startswith("<function=") else tool_call_from_text(r)
            for r in raws
        ]
        return TOOL_CALL.sub("", body).strip(), calls
    stripped = body.strip()
    if stripped.startswith("{") and '"name"' in stripped:
        call = tool_call_from_text(stripped)
        if call.parse != "failed":
            return "", [call]
    return body, []
