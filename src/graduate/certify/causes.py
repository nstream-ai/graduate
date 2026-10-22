"""Why a candidate's call disagrees with the reference's: one cause per disagreeing call.

Causes are assigned first match in this order, so each disagreeing call has exactly one cause
and the shares sum to 100%:

  replied instead of acting   the reference called a tool, the candidate replied in text
  acted instead of replying   the reference replied in text, the candidate called a tool
  wrong tool                  different tool names, or a different number of calls
  missing or extra argument   same tools, different argument names
  format-only argument        same tools and names; every value equal after NORMALISER
  wrong argument value        anything else

Format-only differences are reported, not forgiven: "W1000" for "#W1000" may fail in a real tool.
`graduate certify --normalise-args` counts them as agreement, and the card says so.
"""

from __future__ import annotations

import json
import re
from typing import Any

CAUSES = (
    "replied instead of acting",
    "acted instead of replying",
    "wrong tool",
    "missing or extra argument",
    "format-only argument",
    "wrong argument value",
)
# Protocol constant: changing it changes every card, so it carries a version printed on the card.
NORMALISER_VERSION = "1"
NORMALISER = (
    "strings trimmed and lower-cased, with spaces and the characters # $ - _ , . : / removed; "
    "numbers and numeric strings compared as numbers"
)
_STRIP = re.compile(r"[\s#$\-_,.:/]")
_NUMBER = re.compile(r"[-+]?\d+(\.\d+)?")


def normalise(v: Any) -> Any:
    """NORMALISER, applied recursively to lists and objects."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if _NUMBER.fullmatch(s.replace(",", "")):
            return float(s.replace(",", ""))
        t = _STRIP.sub("", s.lower())
        return float(t) if t.isdigit() else t  # "555-123-2002" and 5551232002 compare equal
    if isinstance(v, list):
        return [normalise(x) for x in v]
    if isinstance(v, dict):
        return {str(k): normalise(x) for k, x in v.items()}
    return v


def _args(c: dict[str, Any]) -> dict[str, Any]:
    a = c.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return {"_raw": a}
    return a if isinstance(a, dict) else {}


def _key(c: dict[str, Any]) -> tuple[str, str]:
    return c["name"], json.dumps(_args(c), sort_keys=True, default=str)


def cause(candidate: list[dict[str, Any]], reference: list[dict[str, Any]]) -> str:
    """The cause of a disagreement (call only for calls that disagree): lists of tool calls,
    each {"name", "arguments"}; an empty list is a text reply."""
    if reference and not candidate:
        return CAUSES[0]
    if candidate and not reference:
        return CAUSES[1]
    cand, ref = sorted(candidate, key=_key), sorted(reference, key=_key)
    if len(cand) != len(ref) or [c["name"] for c in cand] != [c["name"] for c in ref]:
        return CAUSES[2]
    pairs = [(_args(c), _args(r)) for c, r in zip(cand, ref, strict=True)]
    if any(set(c) != set(r) for c, r in pairs):
        return CAUSES[3]
    if all(normalise(c) == normalise(r) for c, r in pairs):
        return CAUSES[4]
    return CAUSES[5]


def first_difference(candidate: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict:
    """The example shown on the card: the first differing call and argument."""
    cand, ref = sorted(candidate, key=_key), sorted(reference, key=_key)
    out: dict[str, Any] = {
        "reference": [c["name"] for c in ref] or "text reply",
        "candidate": [c["name"] for c in cand] or "text reply",
    }
    for c, r in zip(cand, ref, strict=False):
        if c["name"] != r["name"]:
            break
        ca, ra = _args(c), _args(r)
        for k in sorted(set(ca) | set(ra)):
            if ca.get(k, "<missing>") != ra.get(k, "<missing>"):
                out.update(tool=c["name"], argument=k, reference_value=ra.get(k, "<missing>"),
                           candidate_value=ca.get(k, "<missing>"))  # fmt: skip
                return out
    return out


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_DIGITS = re.compile(r"\d")


def redact(v: Any) -> Any:
    """For card.md (likely pasted into a public issue): digits become 0 and emails become
    x@x.x, so the format stays visible ("#W0000" vs "W0000") but the value does not."""
    if isinstance(v, str):
        return _DIGITS.sub("0", _EMAIL.sub("x@x.x", v))
    if isinstance(v, list):
        return [redact(x) for x in v]
    if isinstance(v, dict):
        return {k: redact(x) for k, x in v.items()}
    if isinstance(v, int | float) and not isinstance(v, bool):
        return redact(str(v))
    return v
