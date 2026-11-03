"""Ingest report: what came in, what was repaired, what is missing. Printed after every ingest
so data problems are visible before anything is scored (e.g. '8,412 of 12,000 traces usable')."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from graduate.core import Session


@dataclass
class IngestReport:
    sessions: int = 0
    calls: int = 0
    tool_calls: int = 0
    tool_calls_repaired: int = 0
    tool_calls_failed: int = 0
    repairs: Counter[str] = field(default_factory=Counter)
    sessions_with_template_tokens: int = 0
    sessions_missing_static_context: int = 0
    sessions_without_calls: int = 0
    calls_with_reasoning: int = 0
    calls_with_usage: int = 0
    termination: Counter[str] = field(default_factory=Counter)
    with_outcome: int = 0

    def add(self, s: Session) -> None:
        self.sessions += 1
        self.calls += len(s.calls)
        self.sessions_without_calls += not s.calls
        self.sessions_with_template_tokens += s.flags.template_tokens_stripped > 0
        self.sessions_missing_static_context += s.flags.static_context_missing
        self.termination[s.outcome.termination] += 1
        self.with_outcome += s.outcome.success is not None or s.outcome.reward is not None
        for c in s.calls:
            out = s.call_output(c)
            self.calls_with_reasoning += any(m.reasoning for m in out)
            self.calls_with_usage += c.usage is not None
        for m in s.messages:
            for tc in m.tool_calls:
                self.tool_calls += 1
                self.tool_calls_repaired += tc.parse == "repaired"
                self.tool_calls_failed += tc.parse == "failed"
                self.repairs.update(tc.repairs)

    def render(self) -> str:
        def pct(a: int, b: int) -> str:
            return f"{a:,} ({100 * a / b:.1f}%)" if b else "0"

        n = self.sessions
        lines = [
            f"sessions                     {self.sessions:,}",
            f"model calls                  {self.calls:,}",
            f"tool calls                   {self.tool_calls:,}",
            f"  repaired                   {pct(self.tool_calls_repaired, self.tool_calls)}",
            f"  unparsable (kept, flagged) {pct(self.tool_calls_failed, self.tool_calls)}",
            f"  repairs                    {dict(self.repairs.most_common())}",
            f"template tokens stripped     {pct(self.sessions_with_template_tokens, n)}",
            f"static context missing       {pct(self.sessions_missing_static_context, n)}",
            f"sessions with no model call  {pct(self.sessions_without_calls, self.sessions)}",
            f"calls with reasoning         {pct(self.calls_with_reasoning, self.calls)}",
            f"calls with token usage       {pct(self.calls_with_usage, self.calls)}",
            f"sessions with an outcome     {pct(self.with_outcome, self.sessions)}",
            f"termination                  {dict(self.termination.most_common())}",
        ]
        return "\n".join(lines)


def report(sessions: Iterable[Session]) -> IngestReport:
    r = IngestReport()
    for s in sessions:
        r.add(s)
    return r


REDACTED = {"", "{}", "[redacted]", "<redacted>", "redacted", "[removed]", "present"}


def completeness(sessions: Iterable[Session]) -> dict[str, int | float]:
    """What a replayed prompt may be missing, counted per conversation: no system prompt, no tool
    list, tool calls without a result, redacted or empty model inputs, a logged input much longer
    than the reconstructed prompt (missing context), and a tool list more than 3x the tools ever
    called (the whole catalogue rather than the turn's tools)."""
    n = no_system = no_tools = unanswered = redacted = short = catalogue = 0
    for s in sessions:
        n += 1
        no_system += not s.system and not any(c.system for c in s.calls)
        no_tools += not s.tools and not any(c.tools for c in s.calls)
        calls = {tc.id for m in s.messages for tc in m.tool_calls if tc.id}
        answered = {m.tool_call_id for m in s.messages if m.role == "tool"}
        n_calls = sum(len(m.tool_calls) for m in s.messages)
        n_results = sum(m.role == "tool" for m in s.messages)
        unanswered += bool(calls - answered) if calls else n_results < n_calls
        redacted += any(m.role in ("user", "tool") and m.content.strip().lower() in REDACTED
                        for m in s.messages)  # fmt: skip
        last = s.calls[-1] if s.calls else None
        if last and last.usage and last.usage.input_tokens:
            chars = len(s.system or "") + len(str([t.model_dump() for t in s.tools])) + sum(
                len(m.content) + len(str([t.arguments for t in m.tool_calls]))
                for m in s.call_input(last)
            )  # fmt: skip
            short += last.usage.input_tokens > 2 * max(1, chars / 4)  # about 4 characters a token
        called = {tc.name for m in s.messages for tc in m.tool_calls}
        catalogue += bool(called) and len(s.tools) > 3 * len(called)
    return {"conversations": n, "no_system_prompt": no_system, "no_tool_list": no_tools,
            "tool_calls_without_result": unanswered, "redacted_or_empty_input": redacted,
            "logged_input_much_longer_than_replayed": short,
            "tool_list_over_3x_tools_called": catalogue}  # fmt: skip


def completeness_lines(c: dict[str, int | float], exact_dups: int, near_dups: int) -> list[str]:
    """The card's completeness block: one line per problem found, or one line saying none."""
    n = int(c["conversations"]) or 1
    text = {
        "no_system_prompt": "have no system prompt",
        "no_tool_list": "have no tool list (the candidate is shown no tools)",
        "tool_calls_without_result": "have a tool call without a result",
        "redacted_or_empty_input": "have a redacted or empty user or tool message",
        "logged_input_much_longer_than_replayed": "logged more than twice the input tokens the "
        "replayed prompt has: context is missing (redacted request, or files the trace omits)",
        "tool_list_over_3x_tools_called": "list more than 3x the tools they ever call: the tool "
        "list may be the whole catalogue, not the turn's",
    }
    out = [f"- {c[k]:,} of {n:,} conversations {v}" for k, v in text.items() if c[k]]
    if exact_dups or near_dups:
        out.append(
            f"- {exact_dups:,} conversations repeat an earlier one exactly and "
            f"{near_dups:,} nearly (90% similar): repeats count the same evidence twice"
        )
    return out or [f"- all {n:,} conversations complete (system prompt, tools, results)"]
