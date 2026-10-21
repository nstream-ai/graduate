"""Exam-spec checks: deduplication and leakage, step validity, failure types, rotating windows."""

from datetime import UTC, datetime, timedelta

import pytest

from graduate.core import (
    Message,
    Session,
    ToolCall,
    ToolSchema,
    call_issues,
    calls_from_messages,
    dedupe_sessions,
    failures,
    group_holdout,
    is_valid,
    leakage,
    rolling_rounds,
)

ORDER = ToolSchema(
    name="get_order",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "detail": {"type": "string", "enum": ["short", "full"]},
            "items": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
)


def tc(name: str, args: dict | None) -> ToolCall:
    return ToolCall(name=name, arguments=args, parse="ok" if args is not None else "failed")


def sess(sid: str, user: str, *, ts: datetime | None = None, node: str = "a") -> Session:
    msgs = [Message(role="user", content=user), Message(role="assistant", content="ok")]
    return Session(
        session_id=sid,
        source="t",
        ts=ts,
        agent_node=node,
        tools=[ORDER],
        messages=msgs,
        calls=calls_from_messages(msgs),
    )


# ---------- 1. dedup + leakage ----------


def test_exact_and_near_duplicates_across_splits() -> None:
    base = (
        "please check the status of my order number 4471 placed last tuesday for two blue "
        "lamps and a desk, it has not arrived and the tracking page shows no update since"
    )
    train = [sess("t1", base), sess("t2", "something entirely unrelated about weather in paris")]
    held = [
        sess("h1", base),  # exact
        sess("h2", base.replace("4471", "9020")),  # same template, different id: near
        sess("h3", "how do i reset the password on my account after the email changed"),
    ]
    r = leakage(train, held)
    assert (r.held_out_calls, r.exact_in_train, r.near_in_train) == (3, 1, 2)
    assert ("h3", 0) not in r.leaked and ("h2", 0) in r.leaked
    assert "near-duplicate" in r.render()


def test_dedupe_sessions_drops_exact_repeats_only() -> None:
    a, b = sess("a", "hello there"), sess("b", "hello there")
    c = sess("c", "hello there!")
    kept, dropped = dedupe_sessions([a, b, c])
    assert [s.session_id for s in kept] == ["a", "c"] and dropped == 1


# ---------- 2. step validity ----------


@pytest.mark.parametrize(
    ("call", "issues"),
    [
        (tc("get_order", {"order_id": "A1"}), []),
        (tc("get_order", {"order_id": "A1", "detail": "full", "items": [1, 2]}), []),
        (tc("cancel_order", {"order_id": "A1"}), ["unknown_tool"]),
        (tc("get_order", None), ["unparsed_arguments"]),
        (tc("get_order", {}), ["missing_required:order_id"]),
        (tc("get_order", {"order_id": 7}), ["wrong_type:order_id"]),
        (tc("get_order", {"order_id": "A1", "detail": "huge"}), ["not_in_enum:detail"]),
        (tc("get_order", {"order_id": "A1", "items": [1, "x"]}), ["wrong_type:items.1"]),
        (tc("get_order", {"order_id": "A1", "coupon": "X"}), ["unknown_argument:coupon"]),
    ],
)
def test_call_issues(call: ToolCall, issues: list[str]) -> None:
    assert call_issues(call, [ORDER]) == issues


def test_open_schemas_are_tolerated_unless_strict() -> None:
    open_schema = ToolSchema(name="f", parameters={"properties": {"a": {"type": "number"}}})
    call = tc("f", {"a": 1, "b": 2})
    assert is_valid([call], [open_schema])
    assert call_issues(call, [open_schema], strict=True) == ["unknown_argument:b"]
    assert is_valid([tc("f", {"a": True})], [ToolSchema(name="f")])  # empty schema: name only
    assert call_issues(tc("f", {"a": True}), [open_schema]) == ["wrong_type:a"]


# ---------- 3. failure types ----------


@pytest.mark.parametrize(
    ("proposed", "reference", "expected"),
    [
        ([tc("f", {"a": 1})], [tc("f", {"a": 1.0})], set()),
        ([], [tc("f", {"a": 1})], {"wrong_decision"}),
        ([tc("g", {"a": 1})], [tc("f", {"a": 1})], {"wrong_tool"}),
        ([tc("f", {})], [tc("f", {"a": 1})], {"missing_field"}),
        ([tc("f", {"a": 1, "b": 2})], [tc("f", {"a": 1})], {"extra_field"}),
        ([tc("f", {"a": 2})], [tc("f", {"a": 1})], {"wrong_value"}),
        ([tc("f", {"a": 2, "c": 1})], [tc("f", {"a": 1, "b": 1})],
         {"wrong_value", "missing_field", "extra_field"}),
        ([tc("f", {"a": 1})], [tc("f", {"a": 1}), tc("g", {})], {"wrong_tool"}),
        ([tc("f", None)], [tc("f", {"a": 1})], {"unparsed"}),
        ([tc("g", {}), tc("f", {"a": 1})], [tc("f", {"a": 1}), tc("g", {})], set()),
    ],
)  # fmt: skip
def test_failures(proposed: list, reference: list, expected: set) -> None:
    assert failures(proposed, reference) == expected


# ---------- 6. rotating windows and group holdout ----------


def test_rolling_rounds_move_forward_and_never_overlap() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    ss = [sess(f"s{i:02d}", "x", ts=t0 + timedelta(days=i)) for i in range(20)]
    rounds = rolling_rounds(ss, rounds=3, window=0.2)
    assert [len(r.held_out) for r in rounds] == [4, 4, 4]
    assert [len(r.train) for r in rounds] == [8, 12, 16]
    for r in rounds:
        last_train = max(s.ts for s in r.train)
        assert all(s.ts > last_train for s in r.held_out)  # exam is always later traffic
    assert rounds[0].time_ordered
    with pytest.raises(ValueError):
        rolling_rounds(ss, rounds=5, window=0.25)


def test_group_holdout() -> None:
    ss = [sess("a1", "x", node="a"), sess("b1", "x", node="b"), sess("a2", "x", node="a")]
    train, test = group_holdout(ss, {"b"})
    assert [s.session_id for s in train] == ["a1", "a2"] and [s.session_id for s in test] == ["b1"]


def test_verbosity_and_loop_gates() -> None:
    from graduate.core import repetitive, verbose

    assert not verbose("It shipped yesterday.", "Your order shipped yesterday.")
    assert verbose("It shipped. " * 30, "It shipped.")
    looping = "let me check the order status again " * 12
    assert repetitive(looping) and not repetitive("Your order 7 shipped on Monday via UPS.")


def test_free_text_fields_lookalikes_and_judge_eligibility() -> None:
    from graduate.core import free_text_fields, has_lookalikes, judge_eligible

    handoff = ToolSchema(name="transfer", parameters={"type": "object", "properties": {
        "summary": {"type": "string", "description": "Short summary for the agent"},
        "user_id": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "high"]},
    }})  # fmt: skip
    assert free_text_fields(handoff) == {"summary"}
    ref = tc("transfer", {"summary": "User wants a refund", "user_id": "U1"})
    para = tc("transfer", {"summary": "Customer asks for money back", "user_id": "U1"})
    other_id = tc("transfer", {"summary": "User wants a refund", "user_id": "U2"})
    assert judge_eligible(para, ref, handoff) and not judge_eligible(other_id, ref, handoff)
    assert not judge_eligible(ref, ref, handoff)  # nothing differs
    assert has_lookalikes("api.beer\u2011tasting.club") and not has_lookalikes(
        "api.beer-tasting.club"
    )
