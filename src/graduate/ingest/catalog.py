"""Tool catalog audit: problems in the tool definitions and in how the teacher uses them.

Found the hard way in the first student run: two agents named the same handoff tool
`transfer_to_human_agent` and `transfer_to_human_agents`, and the student mixed them up 52% of
the time. These checks run at ingest so such traps are visible before training:

  near_identical_names   names that differ only slightly (plural, case, separators, edit distance)
  schema_variants        the same tool name with different parameter schemas
  shared_names           the same tool name used by several agents (tool sets)
  schema_changes         a tool's schema changing over time (when sessions have timestamps):
                         a tool update, after which old traces teach the old signature
  teacher_invalid_calls  teacher calls that violate their own session's schema: either the
                         schema changed or the teacher is wrong; the trace alone cannot tell
  unknown_tool_calls     teacher calls to tools missing from the session's tool list
  inconsistent_targets   the same input state with different teacher actions in different
                         sessions: conflicting training targets (several may be acceptable)
"""

from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from graduate.core import Session, action_of, call_issues, canonical_json, exact_state


def _norm_name(name: str) -> str:
    n = re.sub(r"[-_\s.]+", "_", name.strip().lower())
    return re.sub(r"s$", "", n)


@dataclass
class CatalogReport:
    tools: int = 0
    agents: int = 0
    near_identical_names: list[tuple[str, str]] = field(default_factory=list)
    schema_variants: dict[str, int] = field(default_factory=dict)
    shared_names: dict[str, int] = field(default_factory=dict)
    schema_changes: dict[str, int] = field(default_factory=dict)
    teacher_invalid_calls: Counter[str] = field(default_factory=Counter)
    unknown_tool_calls: Counter[str] = field(default_factory=Counter)
    inconsistent_states: int = 0
    states_seen_twice: int = 0

    @property
    def problems(self) -> int:
        return (
            len(self.near_identical_names)
            + len(self.schema_variants)
            + len(self.schema_changes)
            + sum(self.teacher_invalid_calls.values())
            + sum(self.unknown_tool_calls.values())
            + self.inconsistent_states
        )

    def render(self) -> str:
        lines = [
            f"tools {self.tools:,} across {self.agents:,} tool sets (agents)",
            f"near-identical tool names      {self.near_identical_names or 'none'}",
            f"same name, different schemas   {self.schema_variants or 'none'}",
            f"same name in several agents    {self.shared_names or 'none'}",
            f"schema changed over time       {self.schema_changes or 'none (or no timestamps)'}",
            f"teacher calls violating schema {dict(self.teacher_invalid_calls) or 'none'}",
            f"teacher calls to unknown tools {dict(self.unknown_tool_calls) or 'none'}",
            f"same state, different teacher actions: {self.inconsistent_states:,} of "
            f"{self.states_seen_twice:,} repeated states",
        ]
        return "\n".join(lines)


def audit(sessions: Iterable[Session], *, name_similarity: float = 0.9) -> CatalogReport:
    r = CatalogReport()
    schemas: dict[str, Counter[str]] = defaultdict(Counter)
    agents_by_tool: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    first_seen: dict[tuple[str, str], object] = {}
    actions_by_state: dict[str, set[object]] = defaultdict(set)
    count_by_state: Counter[str] = Counter()
    agent_sets: set[tuple[str, ...]] = set()
    for s in sessions:
        tset = tuple(sorted(t.name for t in s.tools))
        agent_sets.add(tset)
        for t in s.tools:
            h = canonical_json(t.parameters)
            schemas[t.name][h] += 1
            agents_by_tool[t.name].add(tset)
            if s.ts is not None:
                key = (t.name, h)
                if key not in first_seen or s.ts < first_seen[key]:  # type: ignore[operator]
                    first_seen[key] = s.ts
        for c in s.calls:
            for tc in s.call_tool_calls(c):
                issues = call_issues(tc, s.tools)
                if issues == ["unknown_tool"]:
                    r.unknown_tool_calls[tc.name] += 1
                elif issues:
                    r.teacher_invalid_calls[issues[0].split(":")[0]] += 1
            state = exact_state(s, c)
            count_by_state[state] += 1
            actions_by_state[state].add(action_of(s.call_tool_calls(c)))
    names = sorted(schemas)
    r.tools, r.agents = len(names), len(agent_sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            close = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= name_similarity
            if _norm_name(a) == _norm_name(b) or close:
                r.near_identical_names.append((a, b))
    r.schema_variants = {n: len(v) for n, v in schemas.items() if len(v) > 1}
    r.shared_names = {n: len(v) for n, v in agents_by_tool.items() if len(v) > 1}
    by_tool_versions: dict[str, int] = Counter(name for name, _ in first_seen)
    r.schema_changes = {n: k for n, k in by_tool_versions.items() if k > 1}
    repeated = [st for st, n in count_by_state.items() if n > 1]
    r.states_seen_twice = len(repeated)
    r.inconsistent_states = sum(len(actions_by_state[st]) > 1 for st in repeated)
    return r
