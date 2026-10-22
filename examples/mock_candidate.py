#!/usr/bin/env python3
"""A stand-in candidate for trying Graduate without a model: an OpenAI-compatible
/v1/chat/completions server on localhost, standard library only.

It answers a user's message by calling the first tool it is offered (with placeholder
arguments), and answers anything else with a short reply. Its report card shows what the
output looks like; it says nothing about any real model.

    python examples/mock_candidate.py            # serves http://localhost:8000/v1
    graduate certify examples/traces.jsonl --base-url http://localhost:8000/v1 --model mock
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def answer(body: dict) -> dict:
    tools = body.get("tools") or []
    last = (body.get("messages") or [{}])[-1]
    if tools and last.get("role") == "user":
        f = tools[0].get("function", tools[0])
        args = {k: "x" for k in (f.get("parameters") or {}).get("required", [])}
        call = {"id": "call_0", "type": "function",
                "function": {"name": f["name"], "arguments": json.dumps(args)}}  # fmt: skip
        return {"role": "assistant", "content": None, "tool_calls": [call]}
    return {"role": "assistant", "content": "Done."}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = json.dumps({"choices": [{"message": answer(body)}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"mock candidate on http://localhost:{port}/v1 (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
