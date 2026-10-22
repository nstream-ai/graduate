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
