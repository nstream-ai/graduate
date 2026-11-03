"""Block D of v0.2.3: latency and usage, the reasoning policy, and the all-held hint."""

from __future__ import annotations

import json
from pathlib import Path

from graduate.cli import main, states
from graduate.ingest.converters.openai_chat import convert_messages
from test_levels import _call, _serve, _trace


def test_latency_usage_and_the_reasoning_policy(tmp_path: Path) -> None:
    traces = [_trace(i, [_call("cancel", id=i)]) for i in range(3)]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in traces))
    server, url, _ = _serve([_call("cancel", id=0)])
    try:
        main(
            ["certify", str(path), "--base-url", url, "--model", "m", "--out", str(tmp_path / "r")]
        )
    finally:
        server.shutdown()
    cached = [json.loads(x) for x in (tmp_path / "r" / "proposals.jsonl").read_text().splitlines()]
    assert all("latency_ms" in p for p in cached[1:])
    card = json.loads((tmp_path / "r" / "card.json").read_text())
    assert card["replay"]["latency_ms_p50"] is not None and card["carry_reasoning"] is False
    md = (tmp_path / "r" / "card.md").read_text()
    assert "replay endpoint, not production" in md and "Reasoning: never scored" in md


def test_reasoning_is_replayed_only_on_request() -> None:
    s = convert_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "reasoning_content": "think A",
         "tool_calls": [_call("lookup")]},
        {"role": "tool", "tool_call_id": "c-lookup", "content": "ok"},
        {"role": "assistant", "content": "done", "reasoning_content": "think B"},
    ], source="t")  # fmt: skip
    plain, _ = states([s])
    carried, _ = states([s], carry_reasoning=True)
    assert not any("reasoning_content" in m for m in plain[1]["messages"])
    assert any(m.get("reasoning_content") == "think A" for m in carried[1]["messages"])
    assert not any("think B" in json.dumps(m) for m in carried[1]["messages"])  # never its own


def test_every_slice_held_suggests_level_tools(tmp_path: Path) -> None:
    traces = [_trace(i, [_call("cancel", id=i)]) for i in range(3)]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in traces))
    cand, curl, _ = _serve([_call("cancel", id=0)])
    ref, rurl, _ = _serve([_call("cancel", id=999)])  # the reference never repeats itself
    try:
        main(["certify", str(path), "--base-url", curl, "--model", "m",
              "--out", str(tmp_path / "r"), "--reference-base-url", rurl,
              "--reference-model", "ref",
              "--reference-noise-cache", str(tmp_path / "noise")])  # fmt: skip
    finally:
        cand.shutdown()
        ref.shutdown()
    assert "Every slice is held by reference noise" in (tmp_path / "r" / "card.md").read_text()


def test_exact_duplicate_conversations_are_dropped(tmp_path: Path) -> None:
    t = _trace(0, [_call("cancel", id=0)])
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps({**t, "id": f"s{i}"}) for i in range(3)))
    server, url, _ = _serve([_call("cancel", id=0)])
    try:
        for out, extra in (("d", []), ("k", ["--keep-duplicates"])):
            main(["certify", str(path), "--base-url", url, "--model", "m",
                  "--out", str(tmp_path / out), *extra])  # fmt: skip
    finally:
        server.shutdown()
    dropped = json.loads((tmp_path / "d" / "card.json").read_text())
    kept = json.loads((tmp_path / "k" / "card.json").read_text())
    assert (dropped["reference_calls"], kept["reference_calls"]) == (1, 3)
    assert dropped["ingest_completeness"]["exact_duplicate_conversations"] == 2
    assert "were dropped" in (tmp_path / "d" / "card.md").read_text()
