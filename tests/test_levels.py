"""Block B of v0.2.3: agreement level, auxiliary and primary tools, per-turn values, the reference
noise measured once, and the table by the reference's action."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from graduate.cli import compared, main
from graduate.ingest.presets import openai

TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
         for n in ("lookup", "cancel", "think")]  # fmt: skip


def _call(name: str, **args) -> dict:
    return {"id": f"c-{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}  # fmt: skip


def _trace(i: int, calls: list[dict], **extra) -> dict:
    return {"id": f"s{i}", "tools": TOOLS, "messages": [
        {"role": "user", "content": f"cancel order {i}"},
        {"role": "assistant", "content": "", "tool_calls": calls, **extra},
        *({"role": "tool", "tool_call_id": c["id"], "content": "ok"} for c in calls),
    ]}  # fmt: skip


def _serve(reply_calls: list[dict]):
    hits = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            hits.append(1)
            msg = {"role": "assistant", "content": None, "tool_calls": reply_calls}
            out = json.dumps({"choices": [{"message": msg}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1", hits


def _certify(tmp: Path, traces: list[dict], reply: list[dict], *extra: str) -> tuple[dict, str]:
    path = tmp / "t.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in traces))
    server, url, _ = _serve(reply)
    out = tmp / "r"
    try:
        main(["certify", str(path), "--base-url", url, "--model", "m", "--out", str(out),
              "--fresh", *extra])  # fmt: skip
    finally:
        server.shutdown()
    return json.loads((out / "card.json").read_text()), (out / "card.md").read_text()


def test_level_tools_ignores_arguments_and_the_card_prints_all_three(tmp_path: Path) -> None:
    traces = [_trace(i, [_call("cancel", id=i)]) for i in range(5)]
    reply = [_call("cancel", id=999)]  # the right tool, the wrong argument
    card, md = _certify(tmp_path, traces, reply)
    rates = next(iter(card["disagreement_by_level"].values()))
    assert rates == {"args": 1.0, "tools": 0.0, "decision": 0.0}
    assert card["level"] == "args" and "Agreement level: **args**" in md
    card, md = _certify(tmp_path, traces, reply, "--level", "tools")
    assert card["decision_split"]["agreement"] == 1.0 and "| args | tools | decision |" in md


def test_auxiliary_calls_do_not_mask_the_primary_one(tmp_path: Path) -> None:
    traces = [_trace(i, [_call("think"), _call("cancel", id=1)]) for i in range(5)]
    card, md = _certify(tmp_path, traces, [_call("cancel", id=1)], "--ignore-tools", "think")
    assert card["decision_split"]["agreement"] == 1.0 and card["ignored_tools"] == ["think"]
    assert "Left out of the comparison (auxiliary): think" in md
    ref = [{"name": "lookup", "arguments": {}}, {"name": "cancel", "arguments": {"id": 1}}]
    cand = [{"name": "cancel", "arguments": {"id": 1}}, {"name": "think", "arguments": {}}]
    assert compared(cand, ref, primary="cancel") == compared(ref, ref, primary="cancel")


def test_table_by_the_reference_action(tmp_path: Path) -> None:
    traces = [_trace(i, [_call("cancel", id=i)]) for i in range(4)]
    card, md = _certify(tmp_path, traces, [_call("lookup")])
    ((key, row),) = card["by_reference_action"].items()
    assert "/cancel/" in key and row["calls"] == 4 and row["disagreement"] == 1.0
    assert "By the reference's action (diagnostic, not certified)" in md


def test_openai_preset_reads_session_id_and_per_turn_values() -> None:
    r = _trace(0, [_call("cancel", id=1)], agent_node="billing", system="You bill.",
               tools=[TOOLS[1]])  # fmt: skip
    r["session_id"] = "conv-7"
    s = openai(r, "t")
    assert s.session_id == "conv-7"
    c = s.calls[0]
    assert (c.agent_node, c.system, [t.name for t in c.tools]) == (
        "billing",
        "You bill.",
        ["cancel"],
    )


def test_reference_noise_is_measured_once_and_reused(tmp_path: Path) -> None:
    traces = [_trace(i, [_call("cancel", id=i)]) for i in range(3)]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in traces))
    cand, curl, _ = _serve([_call("cancel", id=0)])
    ref, rurl, ref_hits = _serve([_call("cancel", id=0)])
    cache = tmp_path / "noise"
    try:
        for model in ("a", "b"):  # two candidates, one reference
            main(["certify", str(path), "--base-url", curl, "--model", model,
                  "--out", str(tmp_path / model), "--reference-base-url", rurl,
                  "--reference-model", "ref", "--reference-noise-cache", str(cache)])  # fmt: skip
    finally:
        cand.shutdown()
        ref.shutdown()
    assert len(ref_hits) == 3  # asked for the first candidate only
    card = json.loads((tmp_path / "b" / "card.json").read_text())
    assert card["reference_noise_cache"].startswith(str(cache))
    assert "measured once, cached in" in (tmp_path / "b" / "card.md").read_text()
