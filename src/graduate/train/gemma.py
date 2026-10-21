"""Gemma 4 tool-call syntax: `<|tool_call>call:NAME{key:VALUE,...}<tool_call|>`.

Strings are delimited by `<|"|>` on both sides, keys are bare, and numbers, true/false/null are
written raw (from the `format_argument` macro in Gemma 4's chat template). Reasoning appears as
`<|channel>thought ... <channel|>` and is not part of the action.
"""

from __future__ import annotations

import re
from typing import Any

QUOTE = '<|"|>'
CALL = re.compile(r"<\|tool_call>call:([^{\s]+)(\{.*?\})<tool_call\|>", re.S)
THOUGHT = re.compile(r"<\|channel>.*?<channel\|>", re.S)


class _Parser:
    def __init__(self, text: str) -> None:
        self.s, self.i = text, 0

    def value(self) -> Any:
        s = self.s
        if s.startswith(QUOTE, self.i):
            end = s.index(QUOTE, self.i + len(QUOTE))
            out = s[self.i + len(QUOTE) : end]
            self.i = end + len(QUOTE)
            return out
        ch = s[self.i]
        if ch == "{":
            return self.mapping()
        if ch == "[":
            self.i += 1
            items = []
            while s[self.i] != "]":
                items.append(self.value())
                if s[self.i] == ",":
                    self.i += 1
            self.i += 1
            return items
        m = re.compile(r"[^,}\]]+").match(s, self.i)
        if m is None:
            raise ValueError(f"unexpected character {ch!r} at {self.i}")
        self.i = m.end()
        raw = m.group(0).strip()
        literals: dict[str, Any] = {"true": True, "false": False, "null": None}
        if raw in literals:
            return literals[raw]
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw

    def key(self) -> str:
        if self.s.startswith(QUOTE, self.i):
            return str(self.value())
        end = self.s.index(":", self.i)
        k = self.s[self.i : end]
        self.i = end
        return k

    def mapping(self) -> dict[str, Any]:
        assert self.s[self.i] == "{"
        self.i += 1
        out: dict[str, Any] = {}
        while self.s[self.i] != "}":
            k = self.key()
            if self.s[self.i] != ":":
                raise ValueError(f"expected ':' at {self.i}")
            self.i += 1
            out[k] = self.value()
            if self.s[self.i] == ",":
                self.i += 1
        self.i += 1
        return out


def parse_arguments(body: str) -> dict[str, Any]:
    """Parse `{key:value,...}` written by Gemma 4's `format_argument`."""
    p = _Parser(body)
    out = p.mapping()
    if p.i != len(body):
        raise ValueError("trailing text after arguments")
    return out


def strip_thoughts(text: str) -> str:
    return THOUGHT.sub("", text)
