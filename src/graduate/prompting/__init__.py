"""Prompting and parsing: the conversation shown to a candidate, and its reply read back as
actions (OpenAI tool calls, Qwen XML calls typed by the tool schema, Gemma's call syntax)."""

from graduate.prompting.chat import Chat, to_chat, tools_spec
from graduate.prompting.decode import parse_completion

__all__ = ["Chat", "parse_completion", "to_chat", "tools_spec"]
