from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from graduate.core import (
    Call,
    Message,
    Session,
    ToolCall,
    action_of,
    agree,
    calls_from_messages,
    canonical_json,
    position_bucket,
    routing_slice,
    slice_of,
    split_sessions,
)


def tc(name: str, args: dict | None) -> ToolCall:
    return ToolCall(name=name, arguments=args, parse="ok" if args is not None else "failed")


def session(sid: str = "s", ts: datetime | None = None) -> Session:
    msgs = [
        Message(role="user", content="hi"),
        Message(role="assistant", reasoning="think"),
        Message(role="assistant", tool_calls=[tc("a", {"x": 1})]),
        Message(role="tool", content="ok"),
        Message(role="assistant", content="done"),
    ]
    return Session(
        session_id=sid, source="t", ts=ts, messages=msgs, calls=calls_from_messages(msgs)
    )


def test_calls_merge_consecutive_assistant_messages() -> None:
    s = session()
    assert [(c.start, c.end) for c in s.calls] == [(1, 3), (4, 5)]
    assert [t.name for t in s.call_tool_calls(s.calls[0])] == ["a"]
    assert s.call_text(s.calls[1]) == "done"
    assert len(s.call_input(s.calls[1])) == 4


def test_bad_spans_are_rejected() -> None:
    s = session()
    with pytest.raises(ValidationError):
        Session(**{**s.model_dump(), "calls": [Call(index=0, start=0, end=2).model_dump()]})
    with pytest.raises(ValidationError):
        Session(**{**s.model_dump(), "calls": [Call(index=1, start=1, end=3).model_dump()]})


def test_json_round_trip() -> None:
    s = session(ts=datetime(2026, 1, 1, tzinfo=UTC))
    assert Session.model_validate_json(s.model_dump_json()) == s


json_scalars = st.one_of(st.integers(-5, 5), st.text(max_size=4), st.booleans(), st.none())
json_dicts = st.dictionaries(st.text(max_size=4), json_scalars, max_size=5)


@given(json_dicts)
def test_canonical_json_ignores_key_order_and_padding(d: dict) -> None:
    reordered = dict(reversed(list(d.items())))
    padded = {k: (f"  {v} " if isinstance(v, str) else v) for k, v in reordered.items()}
    assert canonical_json(d) == canonical_json(padded)


def test_integral_floats_equal_ints_but_strings_do_not() -> None:
    assert canonical_json({"n": 2.0}) == canonical_json({"n": 2})
    assert canonical_json({"n": "2"}) != canonical_json({"n": 2})


@given(st.lists(st.tuples(st.sampled_from("abc"), json_dicts), min_size=1, max_size=4))
def test_parallel_calls_are_order_insensitive(calls: list) -> None:
    a = action_of(tc(n, d) for n, d in calls)
    b = action_of(tc(n, d) for n, d in reversed(calls))
    assert agree(a, b, "args")


def test_agreement_levels() -> None:
    a = action_of([tc("f", {"x": 1})])
    b = action_of([tc("f", {"x": 2})])
    text = action_of([])
    assert agree(a, b, "tools") and not agree(a, b, "args")
    assert agree(a, b, "decision") and not agree(a, text, "decision")
    assert agree(text, action_of([]), "args")
    assert not agree(action_of([tc("f", {"x": 1}), tc("f", {"x": 1})]), a, "tools")


def test_unparsed_arguments_never_agree_on_args() -> None:
    bad = action_of([tc("f", None)])
    assert not agree(bad, bad, "args")
    assert agree(bad, bad, "tools")


def test_slices() -> None:
    s = session()
    first, last = (slice_of(s, c) for c in s.calls)
    assert (first.call_type, first.tool, first.position) == ("tool", "a", "1")
    assert (last.call_type, last.tool, last.position) == ("text", "-", "2-3")
    assert [position_bucket(i) for i in (0, 1, 2, 3, 9, 10, 29, 30)] == [
        "1", "2-3", "2-3", "4-10", "4-10", "11-30", "11-30", "31+",
    ]  # fmt: skip


def test_split_time_ordered_and_disjoint() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    ss = [session(f"s{i}", t0 + timedelta(days=i)) for i in range(10)]
    sp = split_sessions(list(reversed(ss)))
    assert sp.time_ordered
    assert [s.session_id for s in sp["train"]] == [f"s{i}" for i in range(7)]
    assert [s.session_id for s in sp["calibration"]] == ["s8", "s9"]


def test_split_without_timestamps_is_stable_and_flagged() -> None:
    ss = [session(f"s{i}") for i in range(50)]
    a, b = split_sessions(ss), split_sessions(list(reversed(ss)))
    assert not a.time_ordered
    assert [s.session_id for s in a["dev"]] == [s.session_id for s in b["dev"]]
    ids = [s.session_id for part in a.parts.values() for s in part]
    assert sorted(ids) == sorted(s.session_id for s in ss)


def test_split_rejects_duplicates_and_bad_fractions() -> None:
    with pytest.raises(ValueError):
        split_sessions([session("x"), session("x")])
    with pytest.raises(ValueError):
        split_sessions([session("x")], {"train": 0.5})


def test_routing_slice_uses_the_proposed_action() -> None:
    proposed = action_of([tc("b", {}), tc("a", {})])
    sl = routing_slice("n", proposed, 4)
    assert (sl.call_type, sl.tool, sl.position, str(sl)) == (
        "parallel_tool", "a+b", "4-10", "n/parallel_tool/a+b/4-10",
    )  # fmt: skip


def test_split_by_group_keeps_trials_together() -> None:
    ss = [session(f"t{t}-trial{k}") for t in range(40) for k in range(3)]
    sp = split_sessions(ss, group=lambda s: s.session_id.split("-")[0])
    where = {}
    for name, part in sp.parts.items():
        for s in part:
            assert where.setdefault(s.session_id.split("-")[0], name) == name
    assert sum(len(p) for p in sp.parts.values()) == len(ss)
    assert 0.5 < len(sp["train"]) / len(ss) < 0.85 and sp["calibration"]
