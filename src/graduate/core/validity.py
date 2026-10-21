"""Step validity (evaluation level L0): is a proposed call well-formed for the tools it names?

Checked against the session's tool schemas with a small, tolerant JSON-Schema subset: `type`,
`required`, `enum`, `properties`, `items`, `additionalProperties: false`. Tool schemas in real
traces are often sloppy, so unknown keywords are ignored rather than rejected, and a missing or
empty schema only checks that the tool exists.

Issue codes:
  unknown_tool         the tool is not in the session's tool list
  unparsed_arguments   the arguments could not be parsed
  missing_required     a required argument is absent
  unknown_argument     an argument the schema does not define (with additionalProperties false,
                       or always when `strict=True`): the "hallucinated field" case
  wrong_type           an argument has the wrong JSON type
  not_in_enum          a value outside the declared enum
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from graduate.core.schema import ToolCall, ToolSchema

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}


def _type_ok(value: Any, declared: Any) -> bool:
    names = declared if isinstance(declared, list) else [declared]
    for name in names:
        py = _TYPES.get(str(name))
        if py is None:
            return True  # unknown type keyword: tolerate
        if isinstance(value, bool) and name in ("integer", "number"):
            continue
        if isinstance(value, py):
            return True
        if name == "integer" and isinstance(value, float) and value.is_integer():
            return True
    return False


def _check(value: Any, schema: dict[str, Any], path: str, strict: bool, out: list[str]) -> None:
    if not isinstance(schema, dict):
        return
    if "type" in schema and not _type_ok(value, schema["type"]):
        out.append(f"wrong_type:{path}")
        return
    if "enum" in schema and isinstance(schema["enum"], list) and value not in schema["enum"]:
        out.append(f"not_in_enum:{path}")
    if isinstance(value, dict):
        raw = schema.get("properties")
        props: dict[str, Any] = raw if isinstance(raw, dict) else {}
        for name in schema.get("required") or []:
            if name not in value:
                out.append(f"missing_required:{path}{name}")
        closed = schema.get("additionalProperties") is False or (strict and bool(props))
        for k, v in value.items():
            if k in props:
                _check(v, props[k], f"{path}{k}.", strict, out)
            elif closed:
                out.append(f"unknown_argument:{path}{k}")
    elif isinstance(value, list | tuple) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value):
            _check(v, schema["items"], f"{path}{i}.", strict, out)


def call_issues(call: ToolCall, tools: Sequence[ToolSchema], *, strict: bool = False) -> list[str]:
    """Issue codes for one tool call (empty list = valid)."""
    by_name = {t.name: t for t in tools}
    if call.name not in by_name:
        return ["unknown_tool"]
    if call.arguments is None:
        return ["unparsed_arguments"]
    out: list[str] = []
    params = by_name[call.name].parameters
    if params:
        _check(call.arguments, {"type": "object", **params}, "", strict, out)
    return [i.rstrip(".") for i in out]


def is_valid(
    calls: Sequence[ToolCall], tools: Sequence[ToolSchema], *, strict: bool = False
) -> bool:
    return all(not call_issues(c, tools, strict=strict) for c in calls)
