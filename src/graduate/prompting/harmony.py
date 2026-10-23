"""OpenAI's harmony response format (openai/gpt-oss-20b, gpt-oss-120b), read back as actions.

Format, per the openai-harmony documentation (https://cookbook.openai.com/articles/openai-harmony,
https://github.com/openai/harmony) and the gpt-oss model card / chat template
(https://huggingface.co/openai/gpt-oss-20b): an assistant turn is a sequence of messages

    <|start|>assistant<|channel|>CHANNEL[ to=functions.NAME][ <|constrain|>json]<|message|>BODY END

where CHANNEL is `analysis` (chain of thought, never shown), `commentary` (tool calls, and
recipient-less "preambles" addressed to the user) or `final` (the reply), and END is `<|end|>`
(message done, turn continues), `<|call|>` (a tool call; the turn stops) or `<|return|>` (the
final reply; the turn stops). The recipient may also stand before the channel
(`<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>`), and the content
type may be written without `<|constrain|>`.

A `/completions` call returns only the generated part: it may start mid-message (the prompt ends
in `<|start|>assistant`, so the text begins with `<|channel|>`, or even with a body), and the
stop token is usually stripped by the server. Both are handled: a missing END closes the message
at the end of the text. Analysis is dropped; each `to=functions.X` message is a call whose body
is JSON arguments (repaired as other text calls are; unparsable -> a failed call, which counts
against the student as in `decode.py`); the reply is the final-channel text, or the commentary
preambles when there is no final message.
"""

from __future__ import annotations

import re

from graduate.core import ToolCall
from graduate.ingest.repair import make_tool_call

MARKERS = ("<|message|>", "<|channel|>", "<|call|>", "<|return|>")
END = re.compile(r"<\|(?:end|call|return)\|>")
CHANNEL = re.compile(r"<\|channel\|>\s*(\w+)")
RECIPIENT = re.compile(r"\bto=([^\s<]+)")


def is_harmony(text: str) -> bool:
    """Harmony special tokens are present (none of them occurs in Qwen or Gemma output)."""
    return any(m in text for m in MARKERS)


def _messages(text: str) -> list[tuple[str | None, str | None, str, bool]]:
    """(channel, recipient, body, closed) for each message in the text."""
    out = []
    for seg in text.split("<|start|>"):
        if not seg.strip():
            continue
        if "<|message|>" in seg:
            header, body = seg.split("<|message|>", 1)
        else:  # started inside a body (the prompt opened the message), or a lone header
            is_header = seg.lstrip().startswith(("<|channel|>", "assistant"))
            header, body = (seg, "") if is_header else ("", seg)
        m = END.search(body)
        closed = m is not None
        body = body[: m.start()] if m else body
        ch, to = CHANNEL.search(header), RECIPIENT.search(header)
        out.append((ch.group(1) if ch else None, to.group(1) if to else None, body, closed))
    return out


def parse_harmony(text: str) -> tuple[str, list[ToolCall]]:
    """(reply text, tool calls) from harmony-format generated text."""
    finals, preambles, calls = [], [], []
    for channel, recipient, body, _closed in _messages(text):
        if recipient:
            name = recipient.removeprefix("functions.")
            if not body.strip():
                calls.append(ToolCall(name=name, raw=body, parse="failed"))
            else:
                calls.append(make_tool_call(name, body.strip(), raw=body))
        elif channel == "analysis":
            continue
        elif channel == "commentary":
            preambles.append(body.strip())
        elif channel in (None, "final"):
            finals.append(body.strip())
    reply = "\n".join(x for x in (finals or preambles) if x)
    return reply, calls
