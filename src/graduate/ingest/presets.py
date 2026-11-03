"""Field mappings for known datasets and harness exports. A preset turns one source record into a
Session; adding a new source usually means adding a preset, not a converter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from graduate.core import Outcome, Session, Termination, Usage
from graduate.ingest.converters import spans as sp
from graduate.ingest.converters.common import tool_schemas
from graduate.ingest.converters.hermes import convert_sharegpt
from graduate.ingest.converters.openai_chat import convert_messages

Preset = Callable[[dict[str, Any], str], Session | list[Session]]


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


def openai(r: dict[str, Any], source: str) -> Session:
    """One conversation in the OpenAI chat format. `session_id` (else `id`) names it; an
    assistant message may carry its own `agent_node`, `system` and `tools` (per-turn values in
    multi-agent traces), which override the conversation's for that call."""
    s = convert_messages(
        r["messages"], source=source, session_id=r.get("session_id") or r.get("id"),
        tools=r.get("tools"),
    )  # fmt: skip
    turns, prev = [], None  # per call: the first assistant message of its run
    for m in r["messages"]:
        role = str(m.get("role", "")).lower()
        if role == "assistant" and prev != "assistant":
            turns.append(m)
        if role not in ("system", "developer"):
            prev = role
    if len(turns) == len(s.calls):
        for c, m in zip(s.calls, turns, strict=True):
            if m.get("agent_node"):
                c.agent_node = str(m["agent_node"])
            if m.get("system") is not None:
                c.system = str(m["system"])
            if m.get("tools") is not None:
                c.tools = tool_schemas(m["tools"])
    return s


def _int(v: Any) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _system_text(v: Any) -> str | None:
    v = sp.value(v)
    if isinstance(v, list):  # OTel parts: [{"type": "text", "content": ...}]
        return "\n".join(str(p.get("content") or p.get("text") or "") for p in v
                         if isinstance(p, dict)) or None  # fmt: skip
    return str(v) if v else None


def otel_genai(r: dict[str, Any], source: str) -> list[Session]:
    """OpenTelemetry GenAI spans (Google ADK, Cloud Trace, Langfuse, Phoenix, Datadog exports):
    one record is one trace (an OTLP export, a span tree, or a row with a `spans` list). One
    session per agent: its last model-call span holds the history. ADK's legacy `call_llm` spans
    (`gcp.vertex.agent.llm_request` in Gemini format) are read too."""
    model_spans, agent, sid = [], None, r.get("session_id")
    trace = None
    for name, a, tid in sp.spans(r):
        trace = trace or tid
        agent = a.get("gen_ai.agent.name") or agent
        sid = sid or a.get("gen_ai.conversation.id") or a.get("gcp.vertex.agent.session_id")
        legacy = name.startswith("call_llm") and "gcp.vertex.agent.llm_request" in a
        if "gen_ai.input.messages" in a or "gen_ai.output.messages" in a or legacy:
            model_spans.append((a.get("gen_ai.agent.name") or agent, a))
    if not model_spans:
        raise ValueError("no model-call span (gen_ai.input.messages or call_llm) in the record")
    sid = str(sid or trace or "trace")
    out = []
    for node, a in sp.last_per_agent(model_spans):
        if "gen_ai.input.messages" in a or "gen_ai.output.messages" in a:
            raw = a.get("gen_ai.input.messages")
            if isinstance(raw, str) and raw.strip() and isinstance(sp.value(raw), str):
                raise ValueError("gen_ai.input.messages is not valid JSON (cut short on export?)")
            history, _, _ = sp.sniff(raw)
            answer = sp.output(a.get("gen_ai.output.messages"))
            tools, system = sp.value(a.get("gen_ai.tool.definitions")), _system_text(
                a.get("gen_ai.system_instructions"))  # fmt: skip
        else:  # ADK legacy: the Gemini request and response
            history, tools, system = sp.sniff(a.get("gcp.vertex.agent.llm_request"))
            answer = sp.output(a.get("gcp.vertex.agent.llm_response"))
        usage = Usage(
            input_tokens=_int(a.get("gen_ai.usage.input_tokens")),
            output_tokens=_int(a.get("gen_ai.usage.output_tokens")),
            cached_tokens=_int(a.get("gen_ai.usage.cache_read.input_tokens")),
            reasoning_tokens=_int(a.get("gen_ai.usage.reasoning.output_tokens")),
        )
        out.append(sp.session(
            history, answer, source=source,
            session_id=sid if len({n for n, _ in model_spans}) == 1 else f"{sid}/{node}",
            tools=tools if isinstance(tools, list) else [], system=system, agent=node,
            usage=usage, meta={"model": a.get("gen_ai.request.model")},
        ))  # fmt: skip
    return out


def mlflow(r: dict[str, Any], source: str) -> Session:
    """An MLflow trace (`info` + `data.spans`, attribute values as JSON strings). The last
    `CHAT_MODEL` span holds the turn's whole history; `mlflow.spanInputs` / `spanOutputs` are
    read first, `mlflow.chat.messages` only for old traces. The inputs may hold OpenAI chat, OTel
    parts or a Gemini request (MLflow maps other tracers onto the same keys): the shape is
    sniffed."""
    if isinstance(r.get("trace"), (str, dict)):
        r = sp.value(r["trace"])
    info = r.get("info") or {}
    all_spans = sp.spans(r)
    chat = [a for _, a, _ in all_spans
            if sp.value(a.get("mlflow.spanType")) in ("CHAT_MODEL", "LLM")]  # fmt: skip
    if not chat:
        raise ValueError("no CHAT_MODEL span in the trace")
    a = {k: (None if sp.value(v) == "" else v) for k, v in chat[-1].items()}  # "" placeholders
    history, tools, system = sp.sniff(a.get("mlflow.spanInputs"))
    answer = sp.output(a.get("mlflow.spanOutputs"))
    if not history and a.get("mlflow.chat.messages"):  # traces from before MLflow 3
        old = sp.sniff(a["mlflow.chat.messages"])[0]
        if old and old[-1].get("role") == "assistant" and not answer:
            old, answer = old[:-1], old[-1:]
        history = old
    tools = tools or sp.value(a.get("mlflow.chat.tools")) or []
    tu = sp.value(a.get("mlflow.chat.tokenUsage")) or {}
    out_usage = sp.value(a.get("mlflow.spanOutputs")) or {}
    ou = out_usage.get("usage") or {} if isinstance(out_usage, dict) else {}
    usage = Usage(
        input_tokens=_int(tu.get("input_tokens", ou.get("prompt_tokens"))),
        output_tokens=_int(tu.get("output_tokens", ou.get("completion_tokens"))),
        cached_tokens=_int(tu.get("cache_read_input_tokens",
                                  (ou.get("prompt_tokens_details") or {}).get("cached_tokens"))),
    )  # fmt: skip
    root = sp.value(all_spans[0][1].get("mlflow.spanInputs")) if all_spans else None
    meta_sid = ((root or {}).get("request") or {}).get("metadata", {}).get("session_id") \
        if isinstance(root, dict) else None  # fmt: skip
    tm = info.get("trace_metadata") or {}
    sid = meta_sid or tm.get("mlflow.trace.session") or info.get("trace_id") or all_spans[0][2]
    return sp.session(
        history, answer, source=source, session_id=str(sid), tools=tools, system=system,
        agent=None, usage=usage, meta={"model": sp.value(a.get("mlflow.llm.model"))},
    )  # fmt: skip


PRESETS: dict[str, Preset] = {
    "toucan": toucan,
    "nemotron": nemotron,
    "nemotron-swe": lambda r, src: nemotron(r, src, step_limit=100),
    "swe-rebench": swe_rebench,
    "hermes": hermes,
    "tau2": tau2,
    "openai": openai,
    "otel-genai": otel_genai,
    "mlflow": mlflow,
}
