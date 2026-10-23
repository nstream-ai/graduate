"""`graduate certify` end to end against a local fake OpenAI-compatible candidate server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from graduate.cli import main, openai_messages

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
        },
    }
]


def _trace(i: int) -> dict:
    return {
        "id": f"s{i}",
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": f"Where is order {i}?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": json.dumps({"id": str(i)})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "shipped"},
            {"role": "assistant", "content": "It has shipped."},
        ],
    }


class Candidate(BaseHTTPRequestHandler):
    """Calls get_order with the id the user asked about; replies 'Done.' after a tool result."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        last = body["messages"][-1]
        if last["role"] == "user":
            oid = last["content"].split()[-1].rstrip("?")
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "x",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": json.dumps({"id": oid})},
                    }
                ],
            }
        else:
            msg = {"role": "assistant", "content": "Done."}
        out = json.dumps({"choices": [{"message": msg}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args) -> None:
        pass


def test_openai_messages_strict_format() -> None:
    chat = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
        },
        {"role": "tool", "content": "r"},
    ]
    got = openai_messages(chat)
    assert got[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert got[0]["content"] is None
    assert got[1]["tool_call_id"] == got[0]["tool_calls"][0]["id"]


def test_certify_end_to_end(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(200)))
    server = HTTPServer(("127.0.0.1", 0), Candidate)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    out = tmp_path / "report"
    try:
        main(
            [
                "certify",
                str(traces),
                "--preset",
                "openai",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "fake",
                "--out",
                str(out),
                "--concurrency",
                "4",
            ]
        )
    finally:
        server.shutdown()
    card = json.loads((out / "card.json").read_text())
    assert card["reference_calls"] == 400 and card["failed_requests"] == 0
    # the tool call always matches (200 sessions, zero disagreements): certified
    assert any("get_order" in s for s in card["certified_slices"])
    # replies agree by decision only (reply vs act); their wording is not compared, so text slices
    # are "decision only" and stay out of the certified share unless --certify-text-slices
    assert not any("text" in s for s in card["certified_slices"])
    assert card["decision_split"]["replied_instead_of_acting"] == 0.0
    assert 0.0 < card["certified_call_share"] < 1.0
    assert "not compared" in (out / "card.md").read_text()
    verdicts = {r["slice"]: r["verdict"] for r in card["slices"]}
    assert all(v == ("decision only (wording unchecked)" if "text" in k else "certified")
               for k, v in verdicts.items())  # fmt: skip
    assert all(abs(r["detectable_margin"] - 3 / r["sessions"]) < 1e-12 for r in card["slices"])
    assert (out / "card.md").exists() and (out / "proposals.jsonl").exists()


def _run(traces: Path, out: Path, url: str, model: str = "fake", *extra: str) -> None:
    main(["certify", str(traces), "--base-url", url, "--model", model, "--out", str(out), *extra])


def _serve():
    server = HTTPServer(("127.0.0.1", 0), Candidate)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def test_cache_is_tied_to_the_candidate(tmp_path: Path) -> None:
    traces = tmp_path / "t.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(5)))
    server, url = _serve()
    try:
        _run(traces, tmp_path / "r", url)
        with pytest.raises(SystemExit) as e:  # another model, same --out: refused, not reused
            _run(traces, tmp_path / "r", url, "other")
        assert e.value.code == 2
        _run(traces, tmp_path / "r", url, "other", "--fresh")
    finally:
        server.shutdown()
    assert json.loads((tmp_path / "r" / "card.json").read_text())["candidate"] == "other"


def test_failed_requests_are_not_cached_and_all_failing_exits(tmp_path: Path) -> None:
    traces = tmp_path / "t.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(5)))
    with pytest.raises(SystemExit) as e:  # nothing listening there
        _run(traces, tmp_path / "r", "http://127.0.0.1:9/v1")
    assert e.value.code == 2
    lines = (tmp_path / "r" / "proposals.jsonl").read_text().splitlines()
    assert len(lines) == 1 and "_cache" in lines[0]  # only the header: failures are not kept
    server, url = _serve()
    try:
        with pytest.raises(SystemExit):  # the cache was made for another --base-url
            _run(traces, tmp_path / "r", url)
        _run(traces, tmp_path / "r", url, "fake", "--fresh")
    finally:
        server.shutdown()
    assert json.loads((tmp_path / "r" / "card.json").read_text())["failed_requests"] == 0


def test_bad_inputs_exit_with_a_message(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"messages": []}\nnot json\n')
    for args, needle in [
        ([str(tmp_path / "missing.jsonl")], "no such file"),
        ([str(bad)], "not valid JSON"),
    ]:
        with pytest.raises(SystemExit) as e:
            main(["certify", *args, "--base-url", "http://127.0.0.1:9/v1", "--model", "m"])
        assert e.value.code == 2 and needle in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["certify", str(bad), "--base-url", "u", "--model", "m", "--alpha", "5"])
    assert "fraction" in capsys.readouterr().err


class HashlessCandidate(Candidate):
    """Calls the right tool but drops the '#' from the order id, like a real small model."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        last = body["messages"][-1]
        if last["role"] == "user":
            oid = last["content"].split()[-1].rstrip("?").lstrip("#")
            call = {"name": "get_order", "arguments": json.dumps({"id": oid})}
            msg = {"role": "assistant", "content": None,
                   "tool_calls": [{"id": "x", "type": "function", "function": call}]}  # fmt: skip
        else:
            msg = {"role": "assistant", "content": "Done."}
        out = json.dumps({"choices": [{"message": msg}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


def test_card_says_why_a_slice_disagrees_and_normalise_args_is_opt_in(tmp_path: Path) -> None:
    traces = tmp_path / "t.jsonl"
    rows = []
    for i in range(120):
        t = _trace(i)
        t["messages"][1]["content"] = f"Where is order #W{1000 + i}?"
        t["messages"][2]["tool_calls"][0]["function"]["arguments"] = json.dumps(
            {"id": f"#W{1000 + i}"}
        )
        rows.append(json.dumps(t))
    traces.write_text("\n".join(rows))
    server = HTTPServer(("127.0.0.1", 0), HashlessCandidate)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        _run(traces, tmp_path / "strict", url)
        _run(traces, tmp_path / "lenient", url, "fake", "--normalise-args")
    finally:
        server.shutdown()
    strict = json.loads((tmp_path / "strict" / "card.json").read_text())
    tool = next(r for r in strict["slices"] if "get_order" in r["slice"])
    assert tool["verdict"] != "certified" and tool["rate"] == 1.0
    assert tool["disagreement_causes"] == {"format-only argument": 1.0}
    assert tool["example"]["reference_value"] == "#W1000"  # first in session order, raw in json
    md = (tmp_path / "strict" / "card.md").read_text()
    assert "Why slices disagree" in md and '"#W0000"' in md and '"W0000"' in md  # redacted
    assert "#W1000" not in md
    # the verdict follows the observed rate: 100% disagreement is not "collect more data"
    assert tool["verdict"].startswith("not certified: disagreement 100% is significantly above")
    assert "needs ~" not in md and "more conversations will not help" in md
    assert "reported below, not forgiven" in md  # the normaliser was not applied
    assert "| slice | sessions |" not in md  # one per-slice table in card.md
    assert strict["normaliser"]["version"] == "1" and not strict["format_only_counted_as_agreement"]
    lenient = json.loads((tmp_path / "lenient" / "card.json").read_text())
    assert any("get_order" in s for s in lenient["certified_slices"])
    assert lenient["format_only_counted_as_agreement"]
    lenient_md = (tmp_path / "lenient" / "card.md").read_text()
    assert "counted as agreement" in lenient_md
    assert "format-only argument 100% (counted as agreement: --normalise-args)" in lenient_md
    assert "not forgiven" not in lenient_md


def test_planner_history_is_monotone_and_records_calibration(tmp_path: Path) -> None:
    from graduate.certify.card import report_card
    from graduate.cli import plan_history

    def card(n: int):
        keys, sess, wrong = [], [], []
        for i in range(n):
            keys += ["tool", "text"]
            sess += [f"s{i:04d}"] * 2
            wrong += [True, False]
        return report_card(keys, sess, wrong, cap=1)

    path = tmp_path / "planner.json"
    first, _ = plan_history(card(40), path)
    target = 40 + next(r for r in first.slices if r.slice == "text").plan["low"]
    for n in (45, 50, 59):  # same rate (0): the promised total may not come closer
        c, _ = plan_history(card(n), path)
        assert n + next(r for r in c.slices if r.slice == "text").plan["low"] >= target
    done, calibration = plan_history(card(300), path)
    assert next(r for r in done.slices if r.slice == "text").certified
    assert calibration["text"]["certified_at_sessions"] == 300
    assert calibration["text"]["previous_prediction"]["sessions"] == 59
    plan_history(card(40), path, fresh=True)  # --fresh: history starts again
    assert json.loads(path.read_text())["text"][0]["sessions"] == 40


def test_causes_table_prints_tool_names_not_python_lists() -> None:
    from graduate.cli import why_table

    rows = [{"slice": "agent/tool/cancel_order/1", "verdict": "not certified",
             "disagreement_causes": {"wrong tool": 1.0},
             "example": {"cause": "wrong tool", "reference": ["get_order"],
                         "candidate": ["cancel_order"]}}]  # fmt: skip
    md = why_table(rows)
    assert "reference get_order, candidate cancel_order" in md and "['" not in md


def test_reference_that_breaks_its_tool_contract_is_not_admissible(tmp_path: Path) -> None:
    # the reference calls a tool that is not in the conversation's tool list in 1 of 3 sessions
    rows = []
    for i in range(150):
        t = _trace(i)
        if i % 3 == 0:
            t["messages"][2]["tool_calls"][0]["function"]["name"] = "run_code"
        rows.append(json.dumps(t))
    traces = tmp_path / "t.jsonl"
    traces.write_text("\n".join(rows))
    server, url = _serve()
    try:
        _run(traces, tmp_path / "r", url)
    finally:
        server.shutdown()
    card = json.loads((tmp_path / "r" / "card.json").read_text())
    c = card["reference_contract"]
    assert c["conversations_checked"] == 150 and c["violating_conversations"] == 50
    assert not c["admissible"] and c["top_violations"][0][0] == "run_code"
    assert card["certified_slices"] == [] and card["certified_call_share"] == 0.0
    assert "Reference not admissible" in (tmp_path / "r" / "card.md").read_text()


class StubTokenizer:
    """Enough of a Hugging Face tokenizer for --prompt-mode completions, without a download."""

    chat_template = "stub"
    pad_token = "<pad>"

    def apply_chat_template(self, messages, tools=None, **_kw):
        return "".join(f"<{m['role']}>{m.get('content') or ''}" for m in messages) + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


class CompletionsCandidate(BaseHTTPRequestHandler):
    """A /completions server that answers like Qwen3.5: an XML tool call for the order asked."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path.endswith("/completions") and isinstance(body["prompt"], list)
        assert body["max_tokens"] == 256 and body["temperature"] == 0.0
        text = ("<tool_call>\n<function=get_order>\n<parameter=id>\n7\n</parameter>\n"
                "</function>\n</tool_call>")  # fmt: skip
        out = json.dumps({"choices": [{"text": text}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args) -> None:
        pass


def test_prompt_mode_completions_takes_the_research_path(tmp_path: Path, monkeypatch) -> None:
    import graduate.cli as cli

    monkeypatch.setattr(cli, "load_renderer", lambda tok, pt, ml: {
        "tok": StubTokenizer(), "stop": [0], "prompt_tokens": pt, "max_len": ml})  # fmt: skip
    traces = tmp_path / "t.jsonl"
    traces.write_text(json.dumps(_trace(7)))
    server = HTTPServer(("127.0.0.1", 0), CompletionsCandidate)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        _run(traces, tmp_path / "r", url, "qwen", "--prompt-mode", "completions")
    finally:
        server.shutdown()
    card = json.loads((tmp_path / "r" / "card.json").read_text())
    assert card["candidate_path"].startswith("completions")
    assert "the paper's path" in (tmp_path / "r" / "card.md").read_text()
    head = json.loads((tmp_path / "r" / "proposals.jsonl").read_text().splitlines()[0])
    assert head["prompt_mode"] == "completions"  # its own cache: never mixed with chat answers
    assert card["failed_requests"] == 0 and card["decision_split"]["agreement"] > 0


def test_chat_and_completions_paths_parse_the_same_action() -> None:
    # E4 measures divergence on real traffic; this pins that the parsers agree on one action
    from graduate.core import ToolCall, action_of, agree
    from graduate.prompting import parse_completion

    chat_calls = [{"name": "get_order", "arguments": {"id": "7"}}]  # a chat server's tool_calls
    text = (
        "<tool_call>\n<function=get_order>\n<parameter=id>\n7\n</parameter>\n</function>\n"
        "</tool_call>"
    )
    _, parsed = parse_completion(text, TOOLS)  # Qwen3.5 on /completions, read by Graduate
    a = action_of([ToolCall(name=c["name"], arguments=c["arguments"]) for c in chat_calls])
    b = action_of([ToolCall(name=c.name, arguments=c.arguments) for c in parsed])
    assert agree(a, b, "args")


def test_contract_line_says_not_checked_without_a_tool_list():
    from graduate.cli import contract_line

    line = contract_line({"conversations_checked": 0, "admissible": True, "checked": False})
    assert "not checked" in line


def test_text_slices_certify_only_on_request(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(200)))
    server, url = _serve()
    try:
        _run(traces, tmp_path / "r", url, "fake", "--certify-text-slices")
    finally:
        server.shutdown()
    card = json.loads((tmp_path / "r" / "card.json").read_text())
    assert any("text" in s for s in card["certified_slices"])
    assert card["certified_call_share"] == 1.0


def test_noisy_reference_slices_are_held_and_stay_visible() -> None:
    from types import SimpleNamespace

    from graduate.cli import slice_verdicts

    r = SimpleNamespace(slice="agent0/tool/get_order/1", sessions=50, calls=50, rate=0.0,
                        upper=0.06, binding="disagree", certified=False, above_alpha=False,
                        plan=None)  # fmt: skip
    card = SimpleNamespace(slices=[r], alphas={"disagree": 0.05})
    out = slice_verdicts(card, 0.05, noisy={r.slice})
    assert out[0]["verdict"].startswith("held: reference noise")
