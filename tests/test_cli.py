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
    # replies agree by decision only (reply vs act); their wording is not compared
    assert any("text" in s for s in card["certified_slices"])
    assert card["decision_split"]["replied_instead_of_acting"] == 0.0
    assert card["certified_call_share"] == 1.0
    assert "not compared" in (out / "card.md").read_text()
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
