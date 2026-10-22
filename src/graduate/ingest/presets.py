"""Field mappings for known datasets and harness exports. A preset turns one source record into a
Session; adding a new source usually means adding a preset, not a converter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from graduate.core import Outcome, Session, Termination, Usage
from graduate.ingest.converters.hermes import convert_sharegpt
from graduate.ingest.converters.openai_chat import convert_messages

Preset = Callable[[dict[str, Any], str], Session]


def _censor(session: Session, step_limit: int | None) -> Session:
    """Sessions that stop exactly at the harness step limit are censored, not finished."""
    if step_limit and len(session.calls) >= step_limit:
        session.outcome.termination = "step_limit"
    return session


def toucan(r: dict[str, Any], source: str) -> Session:
    return convert_messages(
        r["messages"],
        source=source,
        session_id=r.get("uuid"),
        tools=r.get("available_tools", r.get("tools")),
        meta={"subset": r.get("subset_name"), "target_tools": r.get("target_tools")},
    )


def nemotron(r: dict[str, Any], source: str, step_limit: int | None = None) -> Session:
    s = convert_messages(
        r["messages"], source=source, session_id=r.get("uuid"), tools=r.get("tools")
    )
    return _censor(s, step_limit)


def swe_rebench(r: dict[str, Any], source: str) -> Session:
    exit_status = str(r.get("exit_status") or "")
    termination: Termination = "finished" if exit_status == "submit" else "unknown"
    s = convert_messages(
        r["trajectory"],
        source=source,
        session_id=r.get("trajectory_id"),
        tools=r.get("tools"),
        outcome=Outcome(
            termination=termination,
            success=None if r.get("resolved") is None else bool(r["resolved"]),
            source="dataset:resolved",
        ),
        meta={"instance_id": r.get("instance_id"), "exit_status": exit_status},
    )
    return _censor(s, 100)


def hermes(r: dict[str, Any], source: str) -> Session:
    return convert_sharegpt(
        r["conversations"],
        source=source,
        session_id=r.get("id"),
        tools=r.get("tools"),
        meta={"category": r.get("category"), "subcategory": r.get("subcategory")},
    )


TAU2_TERMINATION: dict[str, Termination] = {
    "user_stop": "finished",
    "agent_stop": "finished",
    "max_steps": "step_limit",
    "too_many_errors": "error",
}


def tau2(sim: dict[str, Any], source: str) -> Session:
    """One tau2-bench simulation.

    Raw result files hold the policy but not the agent's system prompt or the tool definitions;
    add both to each simulation (`system`, `tools`) when you export them. Without them the session
    is flagged static_context_missing. The deterministic database check is kept apart from the full
    reward (which also includes LLM-judged assertions). An infrastructure error is not an outcome
    of the agent, so it never counts as a failure."""
    # The agent's view only. In dual-control domains (telecom) the user calls tools on their own
    # device; those calls and their results are not shown to the agent, so they must not enter
    # its history (they would be orphan tool results and a context the agent never sees).
    if "task_id" not in sim and "termination_reason" not in sim:
        raise ValueError("not a tau2 simulation (no task_id or termination_reason field)")
    msgs = [
        m for m in sim["messages"]
        if not (m.get("role") == "tool" and m.get("requestor") == "user")
        and not (m.get("role") == "user" and m.get("tool_calls") and not m.get("content"))
    ]  # fmt: skip
    info = sim.get("reward_info") or {}
    reward = info.get("reward")
    db = (info.get("db_check") or {}).get("db_match")
    reason = str(sim.get("termination_reason"))
    infra = reason == "infrastructure_error"
    messages = ([{"role": "system", "content": sim["system"]}] if sim.get("system") else []) + msgs
    s = convert_messages(
        messages,
        source=source,
        session_id=sim.get("id"),
        tools=sim.get("tools"),
        outcome=Outcome(
            termination="error" if infra else TAU2_TERMINATION.get(reason, "unknown"),
            success=None if reward is None or infra else reward >= 1.0,
            reward=None if infra else reward,
            source="env:end_state",
        ),
        meta={"task_id": sim.get("task_id"), "trial": sim.get("trial"), "db_match": db,
              "termination_reason": reason, "domain": sim.get("domain")},
    )  # fmt: skip
    # tau2 opens every conversation with a fixed greeting written by its orchestrator, not by any
    # model (no usage, zero cost). It stays as context but is not a model call: counting it would
    # hand any student a free, always-agreeing call.
    if s.calls and msgs and msgs[0].get("role") == "assistant" and not msgs[0].get("usage"):
        first = s.calls[0]
        if first.start == 0 and not msgs[0].get("cost"):
            s.calls = [c.model_copy(update={"index": c.index - 1}) for c in s.calls[1:]]
            s.meta["scripted_greeting"] = True
    for call in s.calls:  # session messages map one to one to `msgs` (the system is kept apart)
        raw = msgs[call.start].get("raw_data") or {}
        if (raw.get("graduate") or {}).get("by") == "student":  # a hybrid run's student turn
            call.produced_by = "student"
        usage = msgs[call.start].get("usage") or {}
        if usage:
            call.usage = Usage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            )
    return s


PRESETS: dict[str, Preset] = {
    "toucan": toucan,
    "nemotron": nemotron,
    "nemotron-swe": lambda r, src: nemotron(r, src, step_limit=100),
    "swe-rebench": swe_rebench,
    "hermes": hermes,
    "tau2": tau2,
    "openai": lambda r, src: convert_messages(
        r["messages"], source=src, session_id=r.get("id"), tools=r.get("tools")
    ),
}
