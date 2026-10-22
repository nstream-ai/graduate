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
import contextlib
import dataclasses
import json
import math
import os
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from graduate.certify.card import plan_text, report_card
from graduate.certify.causes import (
    CAUSES,
    NORMALISER,
    NORMALISER_VERSION,
    cause,
    first_difference,
    redact,
)
from graduate.core import ToolCall, action_of, agree, is_valid, routing_slice
from graduate.core.slicing import action_type_and_tool
from graduate.ingest import PRESETS, ingest
from graduate.prompting.chat import to_chat, tools_spec
from graduate.prompting.decode import parse_completion


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
    normalise_args: bool = False,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    keys, sess, gates = [], [], {"disagree": [], "invalid": []}
    split = Counter()
    why: dict[str, Any] = {}  # slice -> {"causes": Counter, "example": first disagreeing call}
    forgiven: set[str] = set()
    for r in rows:
        p = proposals[r["key"]]
        cand, ref = _action(p["calls"]), _action(r["reference"])
        key = str(routing_slice(r["node"], cand, r["index"]))
        keys.append(key)
        sess.append(r["session"])
        wrong = not agree(cand, ref, "args")
        if wrong:
            c = cause(p["calls"], r["reference"])
            w = why.setdefault(key, {"causes": Counter(), "example": None})
            w["causes"][c] += 1
            if w["example"] is None:  # first disagreeing call in session order: deterministic
                w["example"] = {"cause": c, **first_difference(p["calls"], r["reference"])}
            if normalise_args and c == "format-only argument":
                wrong = False
                forgiven.add(key)
        gates["disagree"].append(wrong)
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
    decision["forgiven_format_only"] = sorted(forgiven)
    return card, decision, why


def slice_verdicts(
    card: Any, alpha: float, why: dict[str, Any] | None = None, calibration: dict | None = None
) -> list[dict]:
    """Per slice: the verdict and the smallest disagreement margin this evidence can resolve
    (zero-event bound, about 3 / n sessions). A slice that cannot resolve alpha is "cannot be
    decided with this evidence", which is not a failure."""
    out = []
    for r in card.slices:
        detectable = 3 / r.sessions if r.sessions else 1.0
        note = f" ({plan_text(r.plan)})" if r.plan else ""
        if r.certified:
            verdict = "certified"
        elif detectable > alpha:
            verdict = "cannot be decided with this evidence" + (
                note or f" (needs ~{math.ceil(3 / alpha)}+ sessions)"
            )
        else:
            verdict = "not certified" + note
        row = {"slice": r.slice, "sessions": r.sessions, "calls": r.calls, "rate": r.rate,
               "upper": r.upper, "detectable_margin": detectable, "verdict": verdict}  # fmt: skip
        if r.plan:
            row["more_conversations"] = r.plan
        if r.slice in (calibration or {}):
            row.update(calibration[r.slice])
        w = (why or {}).get(r.slice)
        if w:
            n = sum(w["causes"].values())
            row["disagreement_causes"] = {c: w["causes"][c] / n for c in CAUSES if w["causes"][c]}
            row["example"] = w["example"]  # raw values here; card.md redacts them
        out.append(row)
    return out


def plan_history(card: Any, path: Path, fresh: bool = False) -> tuple[Any, dict[str, Any]]:
    """Keep the planner's "at least" from going down between runs into the same --out unless the
    rate improved (adding conversations must not move the target further away), and record, per
    certified slice, where it certified and what the planner predicted at the checkpoint before.
    The history lives in planner.json next to the cache (same candidate; --fresh resets it)."""
    hist: dict[str, list] = {}
    if path.exists() and not fresh:
        with contextlib.suppress(ValueError):
            hist = json.loads(path.read_text())
    slices, calibration = [], {}
    for r in card.slices:
        h = hist.setdefault(r.slice, [])
        last = next((x for x in reversed(h) if "low_total" in x), None)
        if r.plan:
            plan = dict(r.plan)
            if last and plan["rate"] >= last["rate"] - 1e-12:  # not better: never move it closer
                for side in ("low", "high"):
                    prev = last.get(f"{side}_total")
                    if plan[side] is not None and prev is not None:
                        plan[side] = max(plan[side], prev - r.sessions)
            totals = {f"{k}_total": (None if plan[k] is None else r.sessions + plan[k])
                      for k in ("low", "high")}  # fmt: skip
            if not h or h[-1].get("sessions") != r.sessions:
                h.append({"sessions": r.sessions, "rate": r.rate, **totals})
            r = dataclasses.replace(r, plan=plan)
        elif r.certified:
            done = next((x for x in h if x.get("certified")), None)
            if done is None:
                done = {"sessions": r.sessions, "certified": True}
                h.append(done)
            calibration[r.slice] = {
                "certified_at_sessions": done["sessions"],
                "previous_prediction": last,  # None: certified at the first checkpoint
            }
        slices.append(r)
    path.write_text(json.dumps(hist, indent=1))
    return dataclasses.replace(card, slices=slices), calibration


def why_table(rows: list[dict[str, Any]]) -> str:
    """card.md: one line per slice that is not certified and has disagreements."""
    failing = [r for r in rows if r["verdict"] != "certified" and r.get("disagreement_causes")]
    if not failing:
        return ""
    head = "\n### Why slices disagree (one cause per call; example values redacted: digits as 0)\n"
    lines = [head, "| slice | causes | example |", "|---|---|---|"]
    for r in failing:
        causes = ", ".join(f"{c} {v:.0%}" for c, v in r["disagreement_causes"].items())
        e = r["example"]
        if "argument" in e:
            ref, cand = (json.dumps(redact(e[k])) for k in ("reference_value", "candidate_value"))
            ex = f"`{e['tool']}.{e['argument']}`: reference {ref}, candidate {cand}"
        else:
            ex = f"reference {e['reference']}, candidate {e['candidate']}"
        lines.append(f"| {r['slice']} | {causes} | {ex} |")
    return "\n".join(lines) + "\n"


def verdict_table(rows: list[dict[str, Any]], alpha: float) -> str:
    lines = [f"\n| slice | sessions | disagreement (upper) | detectable margin | verdict at alpha "
             f"{alpha:g} |", "|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        lines.append(f"| {r['slice']} | {r['sessions']} | {r['rate']:.1%} ({r['upper']:.1%}) | "
                     f"{r['detectable_margin']:.1%} | {r['verdict']} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def claim_b(args: Any, out: Path) -> None:
    """`graduate claim-b`: per-slice and per-session certificate by effect from fork rows."""
    from graduate.certify.fork import card_pairs, fork_card, routing_table

    rows = []
    for path in args.forks.split(","):
        with Path(path).open() as f:
            rows += [json.loads(line) for line in f if line.strip()]
    ns = argparse.Namespace(pool_positions=not args.by_position)
    card = fork_card(card_pairs(rows, ns), session_margin=args.session_margin, cap=args.cap,
                     serving=json.loads(args.serving))  # fmt: skip
    card["routing"] = routing_table(card)
    (out / "claim_b.json").write_text(json.dumps(card, indent=2, default=str))
    d, f = card["delta_turn"], card.get("reference_floor") or {}
    md = [f"# Claim B by effect: {card['n_states']:,} forked states",
          f"\nPer routed turn: signed net harm <= {d:.1%} (session margin "
          f"{args.session_margin:.0%} / cap {args.cap}); per session <= {args.session_margin:.0%}."
          f" Reference self-noise floor: h_T {f.get('h_T', float('nan')):+.1%}, flips "
          f"{f.get('change_T', float('nan')):.1%}.",
          "\n| candidate proposes | states (tasks) | reference success | net harm (upper) | "
          "session harm (upper) | detectable harm | verdict |",
          "|---|---|---|---|---|---|---|"]  # fmt: skip
    for k, t in card["by_proposed_slice"].items():
        md.append(f"| {k} | {t['n_states']} ({t['n_tasks']}) | {t['p_T']:.0%} | "
                  f"{t['harm']:+.1%} ({t['harm_upper']:.1%}) | {t['session_harm_upper']:.1%} | "
                  f"{t['detectable_harm']:.1%} | {t['verdict']} |")  # fmt: skip
    md.append(f"\nComposition: {card['composition']}. Requires {card['requires']}.")
    md.append(f"\nCertified: {card['routing']['certified'] or 'none'}. "
              f"Never route: {card['routing']['never'] or 'none'}.")  # fmt: skip
    text = "\n".join(md) + "\n"
    (out / "claim_b.md").write_text(text)
    print(text)


def outcomes(path: str, outcome: str) -> list[tuple[str, bool]]:
    """{task_id, success} JSONL rows, or a tau2 results.json (infrastructure errors left out)."""
    text = Path(path).read_text()
    if text.lstrip().startswith("{") and '"simulations"' in text[:2000]:
        out = []
        for s in json.loads(text)["simulations"]:
            if s.get("termination_reason") == "infrastructure_error":
                continue
            info = s.get("reward_info") or {}
            ok = (bool((info.get("db_check") or {}).get("db_match")) if outcome == "db"
                  else (info.get("reward") or 0) >= 1.0)  # fmt: skip
            out.append((str(s["task_id"]), ok))
        return out
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [(str(r["task_id"]), bool(r["success"])) for r in rows if r.get("success") is not None]


def compare_cmd(args: Any, out: Path) -> None:
    """`graduate compare`: paired, task-clustered regret with its minimum detectable regret."""
    from graduate.certify.session import compare, per_task

    r = compare(per_task(outcomes(args.reference, args.outcome)),
                per_task(outcomes(args.candidate, args.outcome)), margin=args.margin)  # fmt: skip
    (out / "compare.json").write_text(json.dumps(r, indent=2))
    if not r["tasks"]:
        print("no paired tasks")
        return
    text = (f"# Session-level comparison: {r['tasks']} paired tasks\n\n"
            f"Reference success {r['reference_success']:.1%}, candidate "
            f"{r['candidate_success']:.1%}. Regret {r['regret']:+.1%} (one-sided 95% upper bound "
            f"{r['regret_upper']:+.1%}; candidate worse on {r['tasks_worse']} tasks, better on "
            f"{r['tasks_better']}). Smallest regret this evidence could detect: "
            f"{r['detectable_regret']:.1%}.\n\n**{r['verdict']}** at a margin of "
            f"{r['margin']:.0%}.\n")  # fmt: skip
    (out / "compare.md").write_text(text)
    print(text)


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
    c.add_argument(
        "--normalise-args",
        action="store_true",
        help="count format-only argument differences (e.g. 'W1000' for '#W1000') as agreement; "
        "default: reported, not forgiven",
    )
    b = sub.add_parser(
        "claim-b",
        help="certify by effect from fork outcomes (needs an environment that can be reset to a "
        "logged state)",
    )
    b.add_argument("forks", help="comma-separated JSONL files of fork rows (T1, T2, S branches)")
    b.add_argument("--session-margin", type=float, default=0.10, help="harm allowed per session")
    b.add_argument(
        "--cap", type=int, default=1, help="routed turns per session (delta = margin/cap)"
    )
    b.add_argument("--by-position", action="store_true", help="keep position buckets in slices")
    b.add_argument("--serving", default="{}", help="JSON: the reference's serving configuration")
    b.add_argument("--out", default="graduate-claim-b")
    m = sub.add_parser("compare", help="session level: candidate vs reference on the same tasks")
    m.add_argument("reference", help="JSONL of {task_id, success} rows (or a tau2 results.json)")
    m.add_argument("candidate", help="the same for the candidate or the hybrid")
    m.add_argument("--margin", type=float, default=0.10, help="allowed regret per session")
    m.add_argument("--outcome", choices=["reward", "db"], default="reward",
                   help="tau2 results: full reward or database end state")  # fmt: skip
    m.add_argument("--out", default="graduate-compare")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "claim-b":
        claim_b(args, out)
        return
    if args.cmd == "compare":
        compare_cmd(args, out)
        return
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

    card, decision, why = certify(
        rows, proposals, args.alpha, args.delta, args.cap, noisy, args.normalise_args
    )
    card, calibration = plan_history(card, out / "planner.json", fresh=args.fresh)
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
        "slices": slice_verdicts(card, args.alpha, why, calibration),
        "normaliser": {"version": NORMALISER_VERSION, "rule": NORMALISER},
        "format_only_counted_as_agreement": decision["forgiven_format_only"],
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
        f"agreement {decision['agreement']:.1%}. Argument normaliser v{NORMALISER_VERSION}: "
        f"{NORMALISER}. "
        + (
            "**Format-only differences counted as agreement** (--normalise-args) for slices: "
            + ", ".join(decision["forgiven_format_only"])
            + ". "
            if decision["forgiven_format_only"]
            else ""
        )
        + f"A call is every model turn: a tool call or "
        f"a text reply. Text replies agree by decision only (reply vs act); their wording is not "
        f"compared.\n\n```\n{card.render(top=40)}\n```\n\n{CARD_LEGEND}"
        + verdict_table(report["slices"], args.alpha)
        + why_table(report["slices"])
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
