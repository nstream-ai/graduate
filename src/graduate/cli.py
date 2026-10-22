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
        --base-url http://localhost:8000/v1 --model <candidate> --out report/

Writes report/card.md, report/card.json and report/proposals.jsonl: a resumable cache of the
candidate's answers, tied to the traces, --model and --base-url it was made with (a different
candidate needs another --out, or --fresh). Failed requests are not cached: rerun to retry them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
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


CARD_LEGEND = """How to read the table: a slice is one kind of decision the candidate proposes
(agent node / call type / tool / position bucket in the conversation). sess = conversations
counted; calls = calls in the slice; rate = observed disagreement with the reference;
upper = its 1 - delta upper bound; p = p-value after the multiplicity correction; share =
fraction of all calls. GRADUATED = certified at alpha: a router may send this slice to the
candidate. "-" = not certified; "not enough evidence" says how many more conversations a
slice would likely need at its observed rate.
"""


def _version() -> str:
    try:
        return version("graduate")
    except PackageNotFoundError:
        return "unknown"


def _fraction(s: str) -> float:
    x = float(s)
    if not 0 < x < 1:
        raise argparse.ArgumentTypeError(f"must be a fraction in (0, 1), e.g. 0.05 (got {s})")
    return x


def _fail(msg: str) -> None:
    print(f"graduate: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _key(env: str | None, flag: str) -> str | None:
    if env is None:
        return None
    if env not in os.environ:
        print(f"graduate: warning: {flag} {env} is not set; no API key is sent", file=sys.stderr)
    return os.environ.get(env)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="graduate")
    ap.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser(
        "certify",
        help="which calls can a candidate take over?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    c.add_argument("traces", help="JSON / JSONL / Parquet file of logged conversations")
    c.add_argument(
        "--preset", default="openai", choices=sorted(PRESETS), help="format of the traces file"
    )
    c.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible endpoint of the candidate, ending in /v1 "
        "(e.g. http://localhost:11434/v1 for Ollama)",
    )
    c.add_argument("--model", required=True, help="candidate model name at that endpoint")
    c.add_argument(
        "--api-key-env",
        default=None,
        help="env var holding the candidate endpoint's API key; unset: no key is sent "
        "(a local server needs none)",
    )
    c.add_argument(
        "--alpha", type=_fraction, default=0.05, help="allowed disagreement rate, e.g. 0.05"
    )
    c.add_argument("--delta", type=_fraction, default=0.05, help="1 - confidence, e.g. 0.05")
    c.add_argument(
        "--cap",
        type=int,
        default=1,
        help="calls counted per conversation and slice (the betting test's bet size "
        "scales with 1/cap: keep it near the calls a conversation really has)",
    )
    c.add_argument("--limit", type=int, default=None, help="first N conversations only")
    c.add_argument("--concurrency", type=int, default=8, help="parallel requests")
    c.add_argument(
        "--reference-base-url",
        default=None,
        help="re-ask the reference here to measure its own noise (optional)",
    )
    c.add_argument("--reference-model", default=None, help="the reference model's name there")
    c.add_argument(
        "--reference-api-key-env",
        default=None,
        help="env var holding the reference endpoint's API key; unset: no key is sent",
    )
    c.add_argument("--reference-samples", type=int, default=200, help="states re-asked")
    c.add_argument("--out", default="graduate-report", help="output folder")
    c.add_argument(
        "--fresh", action="store_true", help="ignore cached answers in --out and ask again"
    )
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = _key(args.api_key_env, "--api-key-env")
    ref_key = _key(args.reference_api_key_env, "--reference-api-key-env")

    try:
        rows, skipped = states(ingest(args.traces, args.preset, limit=args.limit))
    except FileNotFoundError:
        _fail(f"no such file: {args.traces}")
    except json.JSONDecodeError as e:
        _fail(f"{args.traces} is not valid JSON ({e.msg})")
    except (KeyError, TypeError, ValueError) as e:
        _fail(
            f"{args.traces} does not look like --preset {args.preset} ({type(e).__name__}: {e}); "
            "check --preset (see README: trace formats)"
        )
    left = ", ".join(f"{n} {k}" for k, n in sorted(skipped.items())) or "none"
    print(f"{len(rows):,} reference calls (left out: {left})", file=sys.stderr)
    if not rows:
        _fail(f"no reference calls found in {args.traces} with --preset {args.preset}")
    cache = out / "proposals.jsonl"
    stamp = {"_cache": 1, "model": args.model, "base_url": args.base_url, "traces": args.traces}
    proposals: dict[str, dict] = {}
    if cache.exists() and not args.fresh:
        lines = [json.loads(x) for x in cache.read_text().splitlines() if x.strip()]
        head = lines[0] if lines and "_cache" in lines[0] else None
        if head is None or {k: head.get(k) for k in stamp} != stamp:
            was = f"{head.get('model')} at {head.get('base_url')}" if head else "an older run"
            _fail(
                f"{cache} holds answers from {was}, not {args.model} at {args.base_url}; "
                "use another --out, or --fresh to ask again"
            )
        proposals = {p["key"]: p for p in lines[1:] if "error" not in p}
    todo = [r for r in rows if r["key"] not in proposals]
    if args.fresh or not cache.exists():
        cache.write_text(json.dumps(stamp) + "\n")

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
            if "error" not in p:  # failures are not cached, so a rerun retries them
                f.write(json.dumps(p) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1:,}/{len(todo):,} proposals", file=sys.stderr)
    failed = [proposals[r["key"]] for r in rows if "error" in proposals[r["key"]]]
    errors = len(failed)
    if errors:
        print(
            f"{errors:,}/{len(rows):,} requests to {args.base_url} failed; first error: "
            f"{failed[0]['error']}",
            file=sys.stderr,
        )
        if errors == len(rows):
            _fail(
                f"every request to {args.base_url} failed: check that the server is running, "
                "that --base-url ends in /v1 and that --model is a name it serves"
            )
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
                        ref_key,
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
        + (
            f"**{errors:,} requests failed** and are left out; rerun to retry.\n\n"
            if errors
            else ""
        )
        + f"{len(rows):,} reference calls; certified share **{card.certified_call_share:.1%}** "
        f"at alpha {args.alpha:g}, confidence {1 - args.delta:.0%}.\n\n"
        f"Replied instead of acting: {decision['replied_instead_of_acting']:.1%}; acted "
        f"instead of replying: {decision['acted_instead_of_replying']:.1%}; overall "
        f"agreement {decision['agreement']:.1%}. A call is every model turn: a tool call or "
        f"a text reply. Text replies agree by decision only (reply vs act); their wording is not "
        f"compared.\n\n```\n{card.render(top=40)}\n```\n\n{CARD_LEGEND}"
    )
    if noise:
        md += (
            "\nReference self-disagreement per slice (re-asked): "
            + ", ".join(f"{k} {v:.0%}" for k, v in sorted(noise.items()))
            + "\n"
        )
    (out / "card.md").write_text(md)
    print(md)
    print(f"wrote {out / 'card.md'} and {out / 'card.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
