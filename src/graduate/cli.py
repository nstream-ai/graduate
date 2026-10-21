"""`graduate certify`: which of your agent's calls can a candidate model take over?

Give it your logged agent traces (the reference model's conversations, with its tool calls) and an
OpenAI-compatible endpoint serving the candidate (vLLM, Ollama, any provider). For every logged
call the candidate answers the same state; a call is a disagreement when its action (reply, or tool
name and arguments) differs from the reference's. A reply agrees with a reply by decision only: its
wording is not compared (judge it separately, or use outcome forks). Calls are grouped into slices
by what the candidate proposes (what a router can see: agent node, call type, tool, position), and
a slice is certified when its disagreement rate is provably below alpha (exact one-sided bounds
over sessions, Benjamini-Hochberg across slices, a per-session cap so one long conversation cannot
dominate). Optionally the reference is re-asked on a sample of states: a slice where the reference
disagrees with itself at or above alpha cannot be certified by agreement by anyone.

    graduate certify traces.jsonl --preset openai \\
        --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-4B --out report/

Writes report/card.md, report/card.json and report/proposals.jsonl (resumable cache).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from graduate.certify.card import report_card
from graduate.core import ToolCall, action_of, agree, is_valid, routing_slice
from graduate.core.slicing import action_type_and_tool
from graduate.ingest import PRESETS, ingest
from graduate.train.data import to_chat, tools_spec
from graduate.train.decode import parse_completion


def openai_messages(chat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat turns in the strict OpenAI wire format: string arguments, call ids, matched tool
    results."""
    out, pending, n = [], [], 0
    for m in chat:
        m = dict(m)
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                f = tc.get("function", tc)
                cid = tc.get("id") or f"call_{n}"
                n += 1
                args = f.get("arguments")
                calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": f["name"],
                            "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                        },
                    }
                )
                pending.append(cid)
            m["tool_calls"] = calls
            m["content"] = m.get("content") or None
        elif m["role"] == "tool":
            m.setdefault("tool_call_id", pending.pop(0) if pending else f"call_{n}")
            if m["tool_call_id"] in pending:
                pending.remove(m["tool_call_id"])
        out.append(m)
    return out


def ask(
    base_url: str,
    model: str,
    api_key: str | None,
    messages: list,
    tools: list,
    temperature: float = 0.0,
    timeout: float = 300,
) -> dict[str, Any]:
    """One chat completion; returns {"text", "calls"} (calls as {"name", "arguments"})."""
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": 1024}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        msg = json.load(r)["choices"][0]["message"]
    calls = []
    for tc in msg.get("tool_calls") or []:
        f = tc.get("function") or {}
        try:
            args = json.loads(f.get("arguments") or "{}")
        except ValueError:
            args = None
        calls.append({"name": f.get("name", "?"), "arguments": args})
    text = msg.get("content") or ""
    if not calls and text:  # a server without tool parsing: the call is in the text
        text, parsed = parse_completion(text, tools)
        calls = [{"name": c.name, "arguments": c.arguments} for c in parsed]
    return {"text": text, "calls": calls}


def states(sessions) -> tuple[list[dict[str, Any]], Counter]:
    """Every reference call with its input state; calls naming a tool the agent does not have, or
    with unparsable arguments, are reference mistakes and left out (counted)."""
    out, skipped = [], Counter()
    for s in sessions:
        system = [{"role": "system", "content": s.system}] if s.system else []
        tools, names = tools_spec(s), {t.name for t in s.tools}
        for c in s.calls:
            if c.produced_by != "teacher":
                skipped["not_reference"] += 1
                continue
            ref = s.call_tool_calls(c)
            if any(t.parse == "failed" for t in ref) or (
                names and any(t.name not in names for t in ref)
            ):
                skipped["reference_invalid"] += 1
                continue
            out.append(
                {
                    "key": f"{s.session_id}:{c.index}",
                    "session": s.session_id,
                    "index": c.index,
                    "node": s.agent_node or "agent",
                    "tools": tools,
                    "schemas": s.tools,
                    "messages": openai_messages(system + to_chat(s.call_input(c))),
                    "reference": [{"name": t.name, "arguments": t.arguments} for t in ref],
                }
            )
    return out, skipped


def _action(calls: list[dict[str, Any]]):
    return action_of([ToolCall(name=c["name"], arguments=c.get("arguments")) for c in calls])


def certify(
    rows: list[dict[str, Any]],
    proposals: dict[str, dict],
    alpha: float,
    delta: float,
    cap: int,
    noisy: set[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    keys, sess, gates = [], [], {"disagree": [], "invalid": []}
    split = Counter()
    for r in rows:
        p = proposals[r["key"]]
        cand, ref = _action(p["calls"]), _action(r["reference"])
        keys.append(str(routing_slice(r["node"], cand, r["index"])))
        sess.append(r["session"])
        gates["disagree"].append(not agree(cand, ref, "args"))
        calls = [ToolCall(name=c["name"], arguments=c.get("arguments")) for c in p["calls"]]
        gates["invalid"].append(bool(calls) and not is_valid(calls, r["schemas"]))
        ck, rk = action_type_and_tool(cand)[0], action_type_and_tool(ref)[0]
        split["reference_acts"] += rk != "text"
        split["reference_replies"] += rk == "text"
        split["replied_instead_of_acting"] += rk != "text" and ck == "text"
        split["acted_instead_of_replying"] += rk == "text" and ck != "text"
    accepted = [k not in (noisy or set()) for k in keys]
    card = report_card(
        keys, sess, gates=gates, alpha=alpha, delta=delta, cap=cap, accepted=accepted
    )
    decision = {
        "replied_instead_of_acting": split["replied_instead_of_acting"]
        / max(1, split["reference_acts"]),
        "acted_instead_of_replying": split["acted_instead_of_replying"]
        / max(1, split["reference_replies"]),
        "agreement": 1 - sum(gates["disagree"]) / max(1, len(rows)),
    }
    return card, decision


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="graduate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("certify", help="which calls can a candidate take over?")
    c.add_argument("traces", help="JSON / JSONL / Parquet file of logged conversations")
    c.add_argument("--preset", default="openai", choices=sorted(PRESETS))
    c.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint of the candidate")
    c.add_argument("--model", required=True, help="candidate model name at that endpoint")
    c.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the key")
    c.add_argument("--alpha", type=float, default=0.05, help="allowed disagreement rate")
    c.add_argument("--delta", type=float, default=0.05, help="1 - confidence")
    c.add_argument(
        "--cap",
        type=int,
        default=1,
        help="calls counted per conversation and slice (the betting test's bet size "
        "scales with 1/cap: keep it near the calls a conversation really has)",
    )
    c.add_argument("--limit", type=int, default=None, help="first N conversations only")
    c.add_argument("--concurrency", type=int, default=8)
    c.add_argument(
        "--reference-base-url",
        default=None,
        help="re-ask the reference here to measure its own noise (optional)",
    )
    c.add_argument("--reference-model", default=None)
    c.add_argument("--reference-samples", type=int, default=200, help="states re-asked")
    c.add_argument("--out", default="graduate-report")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = os.environ.get(args.api_key_env)

    rows, skipped = states(ingest(args.traces, args.preset, limit=args.limit))
    print(f"{len(rows):,} reference calls ({dict(skipped)} left out)", file=sys.stderr)
    cache = out / "proposals.jsonl"
    proposals: dict[str, dict] = {}
    if cache.exists():
        proposals = {p["key"]: p for p in map(json.loads, cache.read_text().splitlines())}
    todo = [r for r in rows if r["key"] not in proposals]

    def one(r: dict[str, Any]) -> dict[str, Any]:
        try:
            return {
                "key": r["key"],
                **ask(args.base_url, args.model, key, r["messages"], r["tools"]),
            }
        except Exception as e:  # a failed request is a failed proposal, never a pass
            return {"key": r["key"], "text": "", "calls": [], "error": str(e)[:200]}

    with ThreadPoolExecutor(args.concurrency) as pool, cache.open("a") as f:
        for i, p in enumerate(pool.map(one, todo)):
            proposals[p["key"]] = p
            f.write(json.dumps(p) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1:,}/{len(todo):,} proposals", file=sys.stderr)
    errors = sum("error" in proposals[r["key"]] for r in rows)
    rows = [r for r in rows if "error" not in proposals[r["key"]]]

    noisy: set[str] = set()
    noise: dict[str, Any] = {}
    if args.reference_base_url and args.reference_model:
        sample = rows[:: max(1, len(rows) // args.reference_samples)][: args.reference_samples]
        again = {}
        with ThreadPoolExecutor(args.concurrency) as pool:
            for r, p in zip(
                sample,
                pool.map(
                    lambda r: ask(
                        args.reference_base_url,
                        args.reference_model,
                        key,
                        r["messages"],
                        r["tools"],
                    ),
                    sample,
                ),
                strict=True,
            ):
                again[r["key"]] = not agree(_action(p["calls"]), _action(r["reference"]), "args")
        by: dict[str, list[bool]] = {}
        for r in sample:
            k = str(routing_slice(r["node"], _action(r["reference"]), r["index"]))
            by.setdefault(k, []).append(again[r["key"]])
        noise = {k: sum(v) / len(v) for k, v in by.items()}
        noisy = {k for k, v in noise.items() if v >= args.alpha}

    card, decision = certify(rows, proposals, args.alpha, args.delta, args.cap, noisy)
    report = {
        "candidate": args.model,
        "reference_calls": len(rows),
        "failed_requests": errors,
        "left_out": dict(skipped),
        "alpha": args.alpha,
        "delta": args.delta,
        "cap": args.cap,
        "certified_call_share": card.certified_call_share,
        "certified_slices": [r.slice for r in card.slices if r.certified],
        "decision_split": decision,
        "reference_self_disagreement": noise,
        "too_noisy_to_certify_by_agreement": sorted(noisy),
    }
    (out / "card.json").write_text(json.dumps(report, indent=2, default=str))
    md = (
        f"# Graduate report card: {args.model}\n\n"
        f"{len(rows):,} reference calls; certified share **{card.certified_call_share:.1%}** "
        f"at alpha {args.alpha:g}, confidence {1 - args.delta:.0%}.\n\n"
        f"Replied instead of acting: {decision['replied_instead_of_acting']:.1%}; acted "
        f"instead of replying: {decision['acted_instead_of_replying']:.1%}; overall "
        f"agreement {decision['agreement']:.1%}. Text replies agree by decision only (reply "
        f"vs act); their wording is not compared.\n\n```\n{card.render(top=40)}\n```\n"
    )
    if noise:
        md += (
            "\nReference self-disagreement per slice (re-asked): "
            + ", ".join(f"{k} {v:.0%}" for k, v in sorted(noise.items()))
            + "\n"
        )
    (out / "card.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
