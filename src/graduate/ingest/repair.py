"""Lenient parsing with an audit trail. Every repair is named and recorded on the tool call, so
training and evaluation can include or exclude repaired calls explicitly. Repairs seen in open
datasets (research/report/trace-variation.md):

  double_encoded       arguments are a JSON string containing JSON
  literal_newline      a literal backslash-n around the JSON
  python_literal       Python dict syntax (single quotes, True/None)
  json_literals_in_py  Python syntax mixed with JSON true/false/null
  name_in_arguments    the tool name nested inside the arguments object
  missing_arguments    no arguments given; taken as {}
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from graduate.core.schema import ParseStatus, ToolCall

_PY = {"true": "True", "false": "False", "null": "None"}
_TEMPLATE_TOKENS = re.compile(
    r"<\|(?:im_start|im_end|im_system|im_middle|im_user|im_assistant|endoftext|eot_id|"
    r"start_header_id|end_header_id|begin_of_text|end_of_text)\|>"
)
_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)


def _load(text: str) -> tuple[Any, list[str]] | None:
    """Try JSON, then Python literals; return (value, repairs) or None."""
    try:
        return json.loads(text), []
    except (ValueError, RecursionError):
        pass
    for candidate, repair in (
        (text, "python_literal"),
        (re.sub(r"\b(true|false|null)\b", lambda m: _PY[m.group(1)], text), "json_literals_in_py"),
    ):
        try:
            return ast.literal_eval(candidate), [repair]
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return None


def parse_value(raw: Any) -> tuple[Any, list[str]] | None:
    if not isinstance(raw, str):
        return raw, []
    text = raw.strip()
    repairs: list[str] = []
    if text.startswith("\\n") or text.endswith("\\n"):
        text = text.removeprefix("\\n").removesuffix("\\n").strip()
        repairs.append("literal_newline")
    loaded = _load(text)
    if loaded is None:
        return None
    value, more = loaded
    repairs += more
    if isinstance(value, str):  # double-encoded JSON
        inner = _load(value.strip())
        if inner is not None and isinstance(inner[0], dict):
            value = inner[0]
            repairs += ["double_encoded", *inner[1]]
    return value, repairs


def parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, ParseStatus, list[str]]:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, "repaired", ["missing_arguments"]
    parsed = parse_value(raw)
    if parsed is None or not isinstance(parsed[0], dict):
        return None, "failed", []
    value, repairs = parsed
    return value, ("repaired" if repairs else "ok"), repairs


def make_tool_call(name: Any, arguments: Any, *, id: str | None = None, raw: str = "") -> ToolCall:
    args, status, repairs = parse_arguments(arguments)
    if not raw:
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments, default=str)
    return ToolCall(id=id, name=str(name), arguments=args, raw=raw, parse=status, repairs=repairs)


def tool_call_from_text(raw: str, *, id: str | None = None) -> ToolCall:
    """A call written as text: {"name": ..., "arguments": ...} as JSON or Python syntax."""
    parsed = parse_value(raw)
    if parsed is None or not isinstance(parsed[0], dict):
        name = re.search(r"""["']name["']\s*:\s*["']([^"']+)["']""", raw)
        return ToolCall(id=id, name=name.group(1) if name else "?", raw=raw, parse="failed")
    obj, repairs = parsed
    args = obj.get("arguments", obj.get("parameters"))
    if "name" not in obj and isinstance(args, dict) and "name" in args:
        args = dict(args)
        obj = {"name": args.pop("name")}
        repairs = [*repairs, "name_in_arguments"]
    tc = make_tool_call(obj.get("name", "?"), args, id=id, raw=raw)
    if repairs:
        tc.repairs = repairs + tc.repairs
        if tc.parse == "ok":
            tc.parse = "repaired"
    return tc


def strip_template_tokens(text: str) -> tuple[str, int]:
    """Remove chat-template control tokens that leaked into content; return (text, count)."""
    cleaned, n = _TEMPLATE_TOKENS.subn("", text)
    return cleaned, n


def split_think(text: str) -> tuple[str, str | None]:
    """Move a leading <think>...</think> block into the reasoning field."""
    m = _THINK.match(text)
    if not m:
        return text, None
    return text[m.end() :], m.group(1).strip()
