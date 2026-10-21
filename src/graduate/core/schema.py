"""Trajectory schema v0.3: one record per agent session.

Context kinds (foundations F2) are kept apart:
  static   `Session.system` and `Session.tools` (compiled into weights by context distillation)
  dynamic  `Session.messages` (history, tool results, user turns)
  hidden   `Message.reasoning` (stored, used for training, never scored)

A *model call* is one assistant turn: a maximal run of consecutive assistant messages. Several
formats split one call into a reasoning message, a text message and one message per tool call,
so calls are reconstructed from the messages (`calls_from_messages`). The input of call k is
`messages[:call.start]`; its output is `messages[call.start:call.end]`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.3"

Role = Literal["user", "assistant", "tool"]
ParseStatus = Literal["ok", "repaired", "failed"]
Termination = Literal["finished", "step_limit", "error", "abandoned", "escalated", "unknown"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class ToolSchema(_Model):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolCall(_Model):
    """One tool invocation. `raw` keeps the original text; `arguments` is the parsed form."""

    id: str | None = None
    name: str
    arguments: dict[str, Any] | None = None
    raw: str = ""
    parse: ParseStatus = "ok"
    repairs: list[str] = Field(default_factory=list)


class Message(_Model):
    role: Role
    content: str = ""
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name on tool-result messages


class Usage(_Model):
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


class Call(_Model):
    """One model call, as a span of assistant messages."""

    index: int  # position of the call in the session, from 0
    start: int  # first output message
    end: int  # one past the last output message
    produced_by: Literal["teacher", "student"] = "teacher"
    model: str | None = None
    usage: Usage | None = None
    latency_ms: float | None = None


class ModelRef(_Model):
    provider: str | None = None
    model: str
    version: str | None = None


class Outcome(_Model):
    termination: Termination = "unknown"
    success: bool | None = None
    reward: float | None = None
    source: str | None = None  # e.g. "dataset:resolved", "env:end_state", "judge"


class Flags(_Model):
    """Data-quality facts recorded at ingest (F1): never silently corrected."""

    template_tokens_stripped: int = 0
    static_context_missing: bool = False
    reasoning_carried_over: bool | None = None
    compacted: bool | None = None
    tool_calls_failed: int = 0
    tool_calls_repaired: int = 0


class Session(_Model):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    source: str  # dataset / format the record came from
    teacher: ModelRef | None = None
    agent_node: str | None = None
    ts: datetime | None = None
    system: str | None = None
    tools: list[ToolSchema] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    calls: list[Call] = Field(default_factory=list)
    outcome: Outcome = Field(default_factory=Outcome)
    flags: Flags = Field(default_factory=Flags)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _calls_are_assistant_spans(self) -> Session:
        prev_end = 0
        for i, c in enumerate(self.calls):
            if c.index != i or not prev_end <= c.start < c.end <= len(self.messages):
                raise ValueError(f"call {i}: bad span [{c.start}, {c.end})")
            if any(m.role != "assistant" for m in self.messages[c.start : c.end]):
                raise ValueError(f"call {i}: span contains non-assistant messages")
            prev_end = c.end
        return self

    def call_input(self, call: Call) -> list[Message]:
        return self.messages[: call.start]

    def call_output(self, call: Call) -> list[Message]:
        return self.messages[call.start : call.end]

    def call_tool_calls(self, call: Call) -> list[ToolCall]:
        return [tc for m in self.call_output(call) for tc in m.tool_calls]

    def call_text(self, call: Call) -> str:
        return "".join(m.content for m in self.call_output(call))


def calls_from_messages(messages: list[Message]) -> list[Call]:
    """One call per maximal run of consecutive assistant messages."""
    calls: list[Call] = []
    i = 0
    while i < len(messages):
        if messages[i].role == "assistant":
            j = i
            while j < len(messages) and messages[j].role == "assistant":
                j += 1
            calls.append(Call(index=len(calls), start=i, end=j))
            i = j
        else:
            i += 1
    return calls
