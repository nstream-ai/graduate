"""Block A of v0.2.3: what real servers need from the replayed conversation and the client."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from graduate import cli
from graduate.ingest.converters.openai_chat import convert_messages
from graduate.prompting.chat import to_chat

THOUGHT = {"google": {"thought_signature": "c2ln"}}


def _session():
    return convert_messages(
        [
            {"role": "user", "content": "cancel my order"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "toolu_01", "type": "function", "extra_content": THOUGHT,
                 "function": {"name": "find_order", "arguments": "{\"q\": 1}"}},
                {"type": "function", "function": {"name": "find_user", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "toolu_01", "content": "order 7"},
            {"role": "tool", "tool_call_id": "lost_id", "content": "user 3"},
            {"role": "assistant", "content": "Done."},
        ],
        source="test",
    )  # fmt: skip


def test_logged_call_ids_and_provider_fields_survive_the_replay():
    msgs = cli.openai_messages(to_chat(_session().messages))
    calls = msgs[1]["tool_calls"]
    assert calls[0]["id"] == "toolu_01" and calls[0]["extra_content"] == THOUGHT
    assert msgs[2]["tool_call_id"] == "toolu_01"  # the original id, matched on both sides
    # a call logged without an id gets one, and the result that answers it gets the same one
    assert calls[1]["id"].startswith("call_") and msgs[3]["tool_call_id"] == calls[1]["id"]
    assert all(isinstance(c["function"]["arguments"], str) for c in calls)


def test_list_content_is_flattened_and_reasoning_dropped():
    blocks = [{"type": "reasoning", "text": "hmm"}, {"type": "text", "text": "Hel"},
              {"type": "output_text", "text": "lo"}]  # fmt: skip
    assert cli.text_of(blocks) == "Hello" and cli.text_of(None) == "" and cli.text_of("a") == "a"


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b""))  # type: ignore[arg-type]


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_rate_limits_and_server_errors_are_retried(monkeypatch: pytest.MonkeyPatch):
    calls, slept = [], []
    reply = {"choices": [{"message": {"content": [{"type": "text", "text": "ok"}]}}]}

    def urlopen(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(429 if len(calls) == 1 else 503)
        return _Reply(json.dumps(reply).encode())

    monkeypatch.setattr(cli.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(cli.time, "sleep", slept.append)
    assert (
        cli.ask("http://x/v1", "m", None, [{"role": "user", "content": "hi"}], [])["text"] == "ok"
    )
    assert len(calls) == 3 and len(slept) == 2 and slept[1] > slept[0] * 0.3


def test_other_errors_and_the_sixth_failure_are_raised(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    monkeypatch.setattr(cli.urllib.request, "urlopen",
                        lambda req, timeout: (_ for _ in ()).throw(_http_error(400)))  # fmt: skip
    with pytest.raises(urllib.error.HTTPError):
        cli.ask("http://x/v1", "m", None, [], [])
    n = []

    def always_503(req, timeout):
        n.append(1)
        raise _http_error(503)

    monkeypatch.setattr(cli.urllib.request, "urlopen", always_503)
    with pytest.raises(urllib.error.HTTPError):
        cli.ask("http://x/v1", "m", None, [], [])
    assert len(n) == cli.ATTEMPTS
