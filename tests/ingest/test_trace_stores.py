"""The otel-genai and mlflow presets on public trace-store fixtures (block C of v0.2.3)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from graduate.core import Message, Session, ToolSchema
from graduate.core.dedup import near_duplicate_sessions
from graduate.ingest import ingest
from graduate.ingest.presets import mlflow, otel_genai
from graduate.ingest.validate import completeness, completeness_lines

HERE = Path(__file__).parent / "traces"


def _load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def _summary(s: Session) -> tuple:
    return (len(s.calls), sum(len(s.call_tool_calls(c)) for c in s.calls),
            [t.name for t in s.tools], bool(s.system))  # fmt: skip


def test_adk_genai_semconv_spans() -> None:
    (s,) = list(ingest(HERE / "adk-genai-semconv-schema-v2.json", "otel-genai"))
    assert s.agent_node == "some_root_agent" and _summary(s) == (2, 1, ["some_tool"], True)
    tc = s.call_tool_calls(s.calls[0])[0]
    assert (tc.name, tc.arguments) == ("some_tool", {"arg1": "val1"})
    assert s.messages[2].role == "tool" and "processed val1" in s.messages[2].content
    u = s.calls[-1].usage
    assert (u.input_tokens, u.output_tokens, u.cached_tokens, u.reasoning_tokens) == (
        150,
        50,
        60,
        15,
    )
    (g,) = otel_genai(_load("adk-grounding.json"), "t")
    assert _summary(g) == (2, 1, ["some_tool"], True)


def test_adk_legacy_redacted_request_is_named() -> None:
    with pytest.raises(ValueError, match="redacted"):
        otel_genai(_load("adk-legacy-call_llm-schema-v1.json"), "t")


def test_adk_legacy_gemini_request() -> None:
    r = _load("adk-legacy-call_llm-schema-v1.json")
    call_llm = r["root_span"]["children"][0]["children"][-1]["attributes"]
    call_llm["gcp.vertex.agent.llm_request"] = json.dumps({
        "system_instruction": {"parts": [{"text": "you are helpful"}]},
        "tools": [{"function_declarations": [{"name": "some_tool", "parameters": {}}]}],
        "contents": [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"functionCall": {"id": "f1", "name": "some_tool",
                                                          "args": {"arg1": "val1"}}}]},
            {"role": "user", "parts": [{"functionResponse": {"id": "f1", "name": "some_tool",
                                                              "response": {"result": "ok"}}}]},
        ],
    })  # fmt: skip
    call_llm["gcp.vertex.agent.llm_response"] = json.dumps(
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "done"}]}}]}
    )
    (s,) = otel_genai(r, "t")
    assert _summary(s) == (2, 1, ["some_tool"], True) and s.agent_node == "some_root_agent"
    assert s.call_tool_calls(s.calls[0])[0].id == "f1" and s.messages[2].tool_call_id == "f1"


def test_one_session_per_agent() -> None:
    r = _load("adk-genai-semconv-schema-v2.json")
    agent = r["root_span"]["children"][0]
    other = copy.deepcopy(agent["children"][2])
    other["attributes"]["gen_ai.agent.name"] = "billing_agent"
    agent["children"].append(other)
    nodes = sorted(s.agent_node for s in otel_genai(r, "t"))
    assert nodes == ["billing_agent", "some_root_agent"]


def test_mlflow_trace() -> None:
    (s,) = list(ingest(HERE / "mlflow-v3-trace-chat-model-tools.json", "mlflow"))
    assert s.session_id == "sess-0001" and _summary(s) == (2, 1, ["tell_joke"], True)
    assert s.call_tool_calls(s.calls[0])[0].id == "1" and s.messages[2].tool_call_id == "1"
    u = s.calls[-1].usage
    assert (u.input_tokens, u.output_tokens) == (15, 68)


def _last_chat(r: dict) -> dict:
    return [x for x in r["data"]["spans"] if "CHAT_MODEL" in str(x["attributes"].get(
        "mlflow.spanType"))][-1]["attributes"]  # fmt: skip


@pytest.mark.parametrize("shape", ["gemini", "parts", "old"])
def test_mlflow_inputs_are_sniffed(shape: str) -> None:
    r = _load("mlflow-v3-trace-chat-model-tools.json")
    a = _last_chat(r)
    if shape == "gemini":
        a["mlflow.spanInputs"] = json.dumps({"contents": [
            {"role": "user", "parts": [{"text": "tell me a joke"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "tell_joke",
                                                          "args": {"joke_length": 50}}}]},
            {"role": "user", "parts": [{"functionResponse": {
                "name": "tell_joke", "response": {"joke": "..."}}}]}]})  # fmt: skip
    elif shape == "parts":
        a["mlflow.spanInputs"] = json.dumps([
            {"role": "user", "parts": [{"type": "text", "content": "tell me a joke"}]},
            {"role": "assistant", "parts": [{"type": "tool_call", "id": "1", "name": "tell_joke",
                                             "arguments": {"joke_length": 50}}]},
            {"role": "user", "parts": [{"type": "tool_call_response", "id": "1",
                                        "response": "..."}]}])  # fmt: skip
    else:  # before MLflow 3: the history in mlflow.chat.messages, inputs a placeholder
        old = json.loads(a["mlflow.spanInputs"])["messages"]
        a["mlflow.spanInputs"] = '""'
        a["mlflow.chat.messages"] = json.dumps(old)
    s = mlflow(r, "t")
    assert len(s.calls) == 2 and s.call_tool_calls(s.calls[0])[0].name == "tell_joke"
    assert [t.name for t in s.tools] == ["tell_joke"]  # from mlflow.chat.tools when not inline


def test_completeness_block_and_duplicates() -> None:
    (s,) = otel_genai(_load("adk-genai-semconv-schema-v2.json"), "t")
    bare = Session(session_id="x", source="t", messages=[Message(role="user", content="[REDACTED]"),
                   Message(role="assistant", content="hi")],
                   tools=[ToolSchema(name=f"t{i}") for i in range(4)])  # fmt: skip
    c = completeness([s, bare])
    assert c["no_system_prompt"] == 1 and c["redacted_or_empty_input"] == 1
    assert c["no_tool_list"] == 0 and c["tool_calls_without_result"] == 0
    assert near_duplicate_sessions([s, s.model_copy(), bare]) == (1, 0)
    lines = completeness_lines(c, 1, 0)
    assert any("no system prompt" in x for x in lines) and any("repeat" in x for x in lines)
