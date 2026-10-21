"""Fixtures copy the shapes found in open datasets (research/report/trace-variation.md)."""

import json
from pathlib import Path

import pytest

from graduate.core import action_of, agree
from graduate.ingest import (
    PRESETS,
    convert_messages,
    convert_sharegpt,
    ingest,
    parse_arguments,
    read_sessions,
    report,
    strip_template_tokens,
    tool_call_from_text,
    write_sessions,
)

DATA = Path(__file__).resolve().parents[4] / "research" / "data" / "open"
TOOLS = [{"type": "function", "function": {"name": "get_price", "parameters": {"type": "object"}}}]


# ---------- repairs ----------


@pytest.mark.parametrize(
    ("raw", "status", "repairs"),
    [
        ({"a": 1}, "ok", []),
        ('{"a": 1}', "ok", []),
        ("{'a': 1, 'b': True}", "repaired", ["python_literal"]),
        ("{'a': true}", "repaired", ["json_literals_in_py"]),
        ('"{\\"a\\": 1}"', "repaired", ["double_encoded"]),
        ('\\n{"a": 1}\\n', "repaired", ["literal_newline"]),
        (None, "repaired", ["missing_arguments"]),
        ("", "repaired", ["missing_arguments"]),
    ],
)
def test_parse_arguments_repairs(raw: object, status: str, repairs: list[str]) -> None:
    args, st, rep = parse_arguments(raw)
    assert st == status and rep == repairs
    assert args == ({"a": 1, "b": True} if "b" in str(raw) else {"a": True} if "true" in str(raw)
                    else {} if not raw else {"a": 1})  # fmt: skip


@pytest.mark.parametrize("raw", ["[1, 2]", "not json", "{'a': }"])
def test_parse_arguments_failures_are_flagged(raw: str) -> None:
    assert parse_arguments(raw) == (None, "failed", [])


def test_hermes_name_nested_in_arguments_is_repaired() -> None:
    raw = '\\n{"arguments": {"queries": [\'q "1"\'], "name": "ExpertQAExtractor"}}\\n'
    t = tool_call_from_text(raw)
    assert t.name == "ExpertQAExtractor" and t.arguments == {"queries": ['q "1"']}
    assert t.parse == "repaired" and "name_in_arguments" in t.repairs
    assert t.raw == raw


def test_unparsable_call_keeps_raw_and_best_effort_name() -> None:
    t = tool_call_from_text('{"name": "f", "arguments": {broken')
    assert (t.name, t.parse, t.arguments) == ("f", "failed", None)


def test_strip_template_tokens() -> None:
    leaked = "<|im_system|>tool_declare<|im_middle|>[]"
    assert strip_template_tokens(leaked) == ("tool_declare[]", 2)
    assert strip_template_tokens("plain <|not_a_token|>") == ("plain <|not_a_token|>", 0)


# ---------- OpenAI-style variants ----------


def test_tool_calls_string_and_object_arguments_agree() -> None:
    def sess(args: object):
        return convert_messages(
            [
                {"role": "user", "content": "price?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "get_price", "arguments": args}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "42"},
            ],
            source="t", tools=TOOLS,
        )  # fmt: skip

    a, b = sess('{"inst": "BTC"}'), sess({"inst": " BTC "})
    acts = [action_of(s.call_tool_calls(s.calls[0])) for s in (a, b)]
    assert agree(*acts, "args")
    assert a.tools[0].name == "get_price" and a.messages[2].tool_call_id == "c1"


def test_legacy_function_call_split_across_messages_is_one_call() -> None:
    """GPT-OSS in Toucan: a reasoning message, then one message per function call."""
    s = convert_messages(
        [
            {"role": "system", "content": "# Tools ..."},
            {"role": "user", "content": "BTC price and candles"},
            {"role": "assistant", "content": "", "reasoning_content": "need both"},
            {"role": "assistant", "content": "",
             "function_call": {"name": "get_price", "arguments": '{"inst": "BTC"}'}},
            {"role": "assistant", "content": "",
             "function_call": {"name": "get_candles", "arguments": '{"bar": "1H"}'}},
            {"role": "function", "name": "get_price", "content": "42"},
            {"role": "function", "name": "get_candles", "content": "[]"},
            {"role": "assistant", "content": "BTC is 42"},
        ],
        source="toucan",
    )  # fmt: skip
    assert len(s.calls) == 2
    first = s.calls[0]
    assert [t.name for t in s.call_tool_calls(first)] == ["get_price", "get_candles"]
    assert s.call_output(first)[0].reasoning == "need both"
    assert s.system == "# Tools ..." and s.messages[0].role == "user"


def test_toucan_sft_roles_with_python_repr() -> None:
    s = convert_messages(
        [
            {"role": "user", "content": "rhyme smile"},
            {"role": "assistant", "content": "Let me look."},
            {"role": "tool_call",
             "content": "{'name': 'find_rhymes', 'arguments': '{\"w\": \"smile\"}'}"},
            {"role": "tool_response", "content": "['while']"},
            {"role": "assistant", "content": "while"},
        ],
        source="toucan-sft",
    )  # fmt: skip
    (t,) = s.call_tool_calls(s.calls[0])
    assert (t.name, t.arguments) == ("find_rhymes", {"w": "smile"})
    assert s.call_text(s.calls[0]) == "Let me look."
    assert s.flags.static_context_missing


def test_template_tokens_counted_and_think_split() -> None:
    s = convert_messages(
        [
            {"role": "system", "content": "<|im_system|>tool_declare<|im_middle|>[...]"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>plan</think>answer"},
        ],
        source="kimi",
    )
    assert s.flags.template_tokens_stripped == 2 and "<|" not in (s.system or "")
    out = s.call_output(s.calls[0])[0]
    assert (out.content, out.reasoning) == ("answer", "plan")


def test_unknown_role_raises() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        convert_messages([{"role": "narrator", "content": "x"}], source="t")


def test_tau2_preset_usage_outcome_termination() -> None:
    sim = {
        "id": "sim1", "task_id": "7", "trial": 0, "termination_reason": "user_stop",
        "reward_info": {"reward": 1.0},
        "messages": [
            {"role": "assistant", "content": "Hi!", "tool_calls": None},
            {"role": "user", "content": "cancel order 9"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "x", "name": "cancel", "arguments": {"order": 9}}],
             "usage": {"prompt_tokens": 400, "completion_tokens": 20}},
            {"role": "tool", "content": "ok", "id": "x"},
            {"role": "assistant", "content": "Done."},
        ],
    }  # fmt: skip
    s = PRESETS["tau2"](sim, "tau2")
    assert s.outcome.success is True and s.outcome.termination == "finished"
    # the opening "Hi!" has no usage: tau2's scripted greeting, context but not a model call
    assert s.meta["scripted_greeting"] and [c.index for c in s.calls] == [0, 1]
    assert s.calls[0].usage is not None and s.calls[0].usage.input_tokens == 400
    assert s.messages[0].content == "Hi!"
    assert s.flags.static_context_missing


def test_tau2_preset_exported_context_db_check_and_infra_error() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "cancel",
            "description": "Cancel.",
            "parameters": {"type": "object", "properties": {"order": {"type": "integer"}}},
        },
    }
    sim = {
        "id": "s2", "termination_reason": "user_stop", "system": "<policy>p</policy>",
        "tools": [tool], "domain": "retail",
        "reward_info": {"reward": 0.0, "db_check": {"db_match": True, "db_reward": 1.0}},
        "messages": [
            {"role": "user", "content": "cancel order 9"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "x", "name": "cancel", "arguments": {"order": 9}}],
             "usage": {"prompt_tokens": 400, "completion_tokens": 20}},
            {"role": "tool", "content": "ok", "id": "x"},
        ],
    }  # fmt: skip
    s = PRESETS["tau2"](sim, "tau2")
    assert s.system == "<policy>p</policy>" and [t.name for t in s.tools] == ["cancel"]
    assert not s.flags.static_context_missing
    assert s.calls[0].usage is not None and s.calls[0].usage.input_tokens == 400
    # full reward failed (an LLM-judged assertion), the database end state is right
    assert s.outcome.success is False and s.meta["db_match"] is True
    assert s.calls[0].produced_by == "teacher"
    sim["messages"][1]["raw_data"] = {"graduate": {"by": "student"}}
    assert PRESETS["tau2"](sim, "tau2").calls[0].produced_by == "student"
    crashed = PRESETS["tau2"](
        {"id": "s3", "termination_reason": "infrastructure_error", "messages": []}, "tau2"
    )
    assert crashed.outcome.termination == "error" and crashed.outcome.success is None


def test_swe_rebench_censoring_at_step_limit() -> None:
    traj = [{"role": "user", "content": "fix"}]
    for _ in range(100):
        traj += [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "execute_bash", "arguments": '{"command": "ls"}'}}]},
            {"role": "tool", "content": "", "tool_call_id": "c"},
        ]  # fmt: skip
    s = PRESETS["swe-rebench"](
        {"trajectory": traj, "tools": [], "resolved": 0, "exit_status": "submit"}, "swe"
    )
    assert len(s.calls) == 100 and s.outcome.termination == "step_limit"
    assert s.outcome.success is False


# ---------- Hermes / ShareGPT ----------


def test_hermes_parallel_calls_and_tool_response() -> None:
    s = convert_sharegpt(
        [
            {"from": "system", "value": "You are a function calling AI model. <tools>...</tools>"},
            {"from": "human", "value": "view and record camera"},
            {"from": "gpt", "value": '<tool_call>\n{"name": "view", "arguments": {"id": 1}}\n'
                                     '</tool_call>\n<tool_call>\n{"name": "rec", "arguments": '
                                     '{"id": 1}}\n</tool_call>'},
            {"from": "tool", "value": "<tool_response>\n{\"ok\": true}\n</tool_response>"},
            {"from": "gpt", "value": "Done."},
        ],
        source="hermes",
        tools=json.dumps(TOOLS),
    )  # fmt: skip
    assert [t.name for t in s.call_tool_calls(s.calls[0])] == ["view", "rec"]
    assert s.call_text(s.calls[0]) == ""
    assert s.messages[2].content == '{"ok": true}'
    assert s.tools[0].name == "get_price"


# ---------- io + report ----------


def test_round_trip_and_report(tmp_path: Path) -> None:
    s = convert_messages(
        [{"role": "user", "content": "q"},
         {"role": "assistant", "function_call": {"name": "f", "arguments": "{'a': 1}"}}],
        source="t",
    )  # fmt: skip
    path = tmp_path / "s.jsonl"
    assert write_sessions([s, s.model_copy(update={"session_id": "b"})], path) == 2
    back = list(read_sessions(path))
    assert back[0] == s
    r = report(back)
    assert (r.sessions, r.calls, r.tool_calls, r.tool_calls_repaired) == (2, 2, 2, 2)
    assert r.repairs["python_literal"] == 2
    assert "repaired" in r.render()


# ---------- real samples (skipped when research/data/open is absent) ----------

REAL = [
    ("toucan", "toucan/train-00000-of-00047.parquet"),
    ("toucan", "toucan/train-00000-of-00040.parquet"),
    ("toucan", "toucan/train-00000-of-00003.parquet"),
    ("nemotron", "nemotron-agentic/interactive_agent.jsonl"),
    ("nemotron", "nemotron-agentic/tool_calling.jsonl"),
    ("nemotron-swe", "nemotron-swe/r2e_gym.jsonl"),
    ("swe-rebench", "swe-rebench-openhands/trajectories.parquet"),
    ("hermes", "hermes-fc/func-calling.json"),
    ("tau2", "tau2-bench/data/tau2/results/final/"
             "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json"),
]  # fmt: skip


@pytest.mark.parametrize(("preset", "rel"), REAL)
def test_real_samples_convert(preset: str, rel: str) -> None:
    path = DATA / rel
    if not path.exists():
        pytest.skip("open dataset samples not fetched (research/datasets/fetch.py)")
    r = report(ingest(path, preset, limit=None if preset == "hermes" else 300))
    assert r.sessions > 0 and r.calls > 0
    assert r.tool_calls_failed <= 0.02 * max(r.tool_calls, 1)
    if preset == "hermes":
        assert r.repairs["name_in_arguments"] > 0


def test_tau2_preset_keeps_only_the_agents_view() -> None:
    """Telecom: the user's own tool calls and results are not part of the agent's history."""
    sim = {
        "id": "t1", "termination_reason": "user_stop", "system": "policy", "tools": [],
        "messages": [
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": None,
             "tool_calls": [{"id": "u1", "name": "check_status_bar", "arguments": {},
                             "requestor": "user"}]},
            {"role": "tool", "content": "Airplane Mode: ON", "id": "u1", "requestor": "user"},
            {"role": "user", "content": "My data does not work."},
            {"role": "assistant", "content": "Please turn off airplane mode.",
             "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        ],
    }  # fmt: skip
    s = PRESETS["tau2"](sim, "tau2")
    assert [m.role for m in s.messages] == ["assistant", "user", "assistant"]
    assert all("Airplane" not in (m.content or "") for m in s.messages)
    assert len(s.calls) == 1 and s.calls[0].usage is not None
