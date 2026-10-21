from datetime import UTC, datetime, timedelta

from graduate.core import Message, Session, ToolCall, ToolSchema, calls_from_messages
from graduate.ingest import audit

OBJ = {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"],
       "additionalProperties": False}  # fmt: skip


def sess(sid, tools, call, *, user="where is my order", ts=None):
    msgs = [Message(role="user", content=user), Message(role="assistant", tool_calls=[call])]
    return Session(session_id=sid, source="t", tools=tools, messages=msgs, ts=ts,
                   calls=calls_from_messages(msgs))  # fmt: skip


def test_catalog_audit_finds_the_known_traps() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    a = [ToolSchema(name="transfer_to_human_agents", parameters=OBJ)]
    b = [ToolSchema(name="transfer_to_human_agent", parameters=OBJ)]
    v2 = [ToolSchema(name="transfer_to_human_agents", parameters={**OBJ, "required": []})]
    ss = [
        sess("1", a, ToolCall(name="transfer_to_human_agents", arguments={"id": "x"}), ts=t0),
        sess("2", b, ToolCall(name="transfer_to_human_agent", arguments={"id": "x"}), ts=t0),
        sess(
            "3", v2, ToolCall(name="transfer_to_human_agents", arguments={}), ts=t0 + timedelta(1)
        ),
        sess("4", a, ToolCall(name="transfer_to_human_agents", arguments={"id": 5}), ts=t0),
        sess("5", a, ToolCall(name="cancel_order", arguments={"id": "x"}), ts=t0),
        sess("6", a, ToolCall(name="transfer_to_human_agents", arguments={"id": "y"}), ts=t0),
    ]
    r = audit(ss)
    assert ("transfer_to_human_agent", "transfer_to_human_agents") in r.near_identical_names
    assert r.schema_variants == {"transfer_to_human_agents": 2}
    assert r.schema_changes == {"transfer_to_human_agents": 2}
    assert r.teacher_invalid_calls == {"wrong_type": 1}
    assert r.unknown_tool_calls == {"cancel_order": 1}
    # sessions 1, 4, 5, 6 share the state (same tools, same user turn) with different actions
    assert r.inconsistent_states == 1 and r.states_seen_twice == 1
    assert r.problems > 0 and "near-identical" in r.render()
