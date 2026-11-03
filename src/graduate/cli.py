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
import os
import random
import sys
import time
import urllib.error
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
from graduate.core.canonical import stable_hash
from graduate.core.dedup import dedupe_sessions, near_duplicate_sessions
from graduate.core.slicing import action_type_and_tool
from graduate.ingest import PRESETS, ingest
from graduate.ingest.validate import completeness, completeness_lines
from graduate.prompting.chat import to_chat, tools_spec
from graduate.prompting.decode import parse_completion


def openai_messages(chat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat turns in the strict OpenAI wire format: string arguments, call ids, matched tool
    results. The logged call ids are kept; an id is made up only for a call that has none, and
    then the tool result that answers it gets the same id. Provider fields on a call (`extra`)
    pass through unchanged."""
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
                extra = {k: v for k, v in tc.items() if k not in ("id", "type", "function")}
                if "function" not in tc:
                    extra = {k: v for k, v in extra.items() if k not in ("name", "arguments")}
                calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": f["name"],
                            "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                        },
                        **extra,
                    }
                )
                pending.append(cid)
            m["tool_calls"] = calls
            m["content"] = m.get("content") or None
        elif m["role"] == "tool":
            tid = m.get("tool_call_id")
            if tid not in pending:  # missing, or the call's id was made up: answer the oldest
                tid = pending[0] if pending else tid or f"call_{n}"
            m["tool_call_id"] = tid
            if tid in pending:
                pending.remove(tid)
        out.append(m)
    return out


RETRY_STATUS = (429, 500, 502, 503, 504)
ATTEMPTS = 6


def post_json(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
    """POST with exponential backoff (1, 2, 4, 8, 16 s, with jitter) on rate limits and server
    errors; any other error, or the sixth failure, is raised."""
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == ATTEMPTS - 1:
                raise
            wait = float(e.headers.get("Retry-After") or 0) if e.headers else 0.0
            time.sleep(max(wait, 2**attempt * (0.5 + random.random())))
    raise AssertionError("unreachable")


def text_of(content: Any) -> str:
    """A reply's text: a string, or a list of content blocks (text / output_text kept, reasoning
    and other blocks dropped)."""
    if isinstance(content, list):
        return "".join(
            str(b.get("text") or "")
            for b in content
            if isinstance(b, dict) and b.get("type") in ("text", "output_text")
        )
    return content or ""


def ask(
    base_url: str,
    model: str,
    api_key: str | None,
    messages: list,
    tools: list,
    temperature: float = 0.0,
    timeout: float = 300,
) -> dict[str, Any]:
    """One chat completion; returns {"text", "calls", "latency_ms", "usage"} (calls as
    {"name", "arguments"}; usage as the server reports it)."""
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
    t0 = time.monotonic()
    resp = post_json(req, timeout)
    latency = (time.monotonic() - t0) * 1000
    msg = resp["choices"][0]["message"]
    calls = []
    for tc in msg.get("tool_calls") or []:
        f = tc.get("function") or {}
        try:
            args = json.loads(f.get("arguments") or "{}")
        except ValueError:
            args = None
        calls.append({"name": f.get("name", "?"), "arguments": args})
    text = text_of(msg.get("content"))
    if not calls and text:  # a server without tool parsing: the call is in the text
        text, parsed = parse_completion(text, tools)
        calls = [{"name": c.name, "arguments": c.arguments} for c in parsed]
    return {"text": text, "calls": calls, "latency_ms": latency, "usage": usage_of(resp)}


def usage_of(resp: dict[str, Any]) -> dict[str, int]:
    """Input, output and reasoning tokens as an OpenAI-compatible server reports them."""
    u = resp.get("usage") or {}
    details = u.get("completion_tokens_details") or {}
    out = {
        "input_tokens": u.get("prompt_tokens"),
        "output_tokens": u.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
    }
    return {k: int(v) for k, v in out.items() if isinstance(v, (int, float))}


COMPLETION_TOKENS = 256  # new tokens per call on the completions path, as in the research path


def load_renderer(tokenizer_id: str, prompt_tokens: int, max_len: int) -> dict[str, Any]:
    """The tokenizer and limits for --prompt-mode completions (needs the `completions` extra)."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        _fail('--prompt-mode completions needs transformers: pip install "graduate[completions]"')
    from graduate.prompting.render import ensure_template, stop_token_ids

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    ensure_template(tok)
    return {"tok": tok, "stop": sorted(stop_token_ids(tokenizer_id)),
            "prompt_tokens": prompt_tokens, "max_len": max_len}  # fmt: skip


def ask_completions(
    base_url: str, model: str, api_key: str | None, chat: list, tools: list, renderer: dict,
    timeout: float = 300,
) -> dict[str, Any]:  # fmt: skip
    """One call on the research path: the prompt rendered with the model's own chat template
    (thinking off), fitted to the token budget by dropping the oldest turns, sent to
    `/completions`, and the reply read by Graduate's parsers."""
    from graduate.prompting.render import fit_messages, render_prompt

    tok = renderer["tok"]
    msgs = fit_messages(chat, tools, tok, renderer["prompt_tokens"])
    ids = tok(render_prompt(msgs, tools, tok), add_special_tokens=False)["input_ids"]
    ids = ids[-(renderer["max_len"] - COMPLETION_TOKENS) :]
    body = {"model": model, "prompt": ids, "max_tokens": COMPLETION_TOKENS, "temperature": 0.0,
            "stop_token_ids": renderer["stop"], "skip_special_tokens": False}  # fmt: skip
    req = urllib.request.Request(
        base_url.rstrip("/") + "/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    t0 = time.monotonic()
    resp = post_json(req, timeout)
    latency = (time.monotonic() - t0) * 1000
    content, parsed = parse_completion(resp["choices"][0]["text"], tools)
    return {"text": content, "calls": [{"name": c.name, "arguments": c.arguments} for c in parsed],
            "latency_ms": latency, "usage": usage_of(resp)}  # fmt: skip


def states(sessions, carry_reasoning: bool = False) -> tuple[list[dict[str, Any]], Counter]:
    """Every reference call with its input state; calls naming a tool the agent does not have, or
    with unparsable arguments, are reference mistakes and left out (counted)."""
    out, skipped = [], Counter()
    violated: dict[
        str, bool
    ] = {}  # the reference's tool contract, per conversation with a tool list
    violation_names: Counter = Counter()
    for s in sessions:
        for c in s.calls:
            # per-turn overrides (multi-agent traces): the call's own node, system and tools
            sys_text = c.system if c.system is not None else s.system
            system = [{"role": "system", "content": sys_text}] if sys_text else []
            schemas = c.tools if c.tools is not None else s.tools
            tools, names = tools_spec(s, schemas), {t.name for t in schemas}
            if c.produced_by != "teacher":
                skipped["not_reference"] += 1
                continue
            ref = s.call_tool_calls(c)
            if names:
                bad = [t.name for t in ref if t.name not in names]
                violated[s.session_id] = violated.get(s.session_id, False) or bool(bad)
                violation_names.update(bad)
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
                    "node": c.agent_node or s.agent_node or "agent",
                    "tools": tools,
                    "schemas": schemas,
                    # prior turns' reasoning only with --carry-reasoning; the re-asked turn's
                    # own output (and its reasoning) is never in its input
                    "messages": openai_messages(
                        system + to_chat(s.call_input(c), reasoning=carry_reasoning)
                    ),
                    "chat": system + to_chat(s.call_input(c), reasoning=carry_reasoning),
                    "reference_usage": c.usage.model_dump(exclude_none=True) if c.usage else {},
                    "reference": [{"name": t.name, "arguments": t.arguments} for t in ref],
                }
            )
    checked = len(violated)
    share = sum(violated.values()) / checked if checked else 0.0
    skipped_contract = {
        "conversations_checked": checked,
        "violating_conversations": sum(violated.values()),
        "violating_share": share,
        "top_violations": violation_names.most_common(5),
        "admissible": share <= CONTRACT_LIMIT,
        # no tool list in the traces: the contract cannot be checked, and the card says so
        "checked": checked > 0,
    }
    states.contract = skipped_contract  # read by main(); keeps the (rows, skipped) signature
    return out, skipped


def _action(calls: list[dict[str, Any]]):
    return action_of([ToolCall(name=c["name"], arguments=c.get("arguments")) for c in calls])


LEVELS = ("args", "tools", "decision")


def compared(
    calls: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    ignore: frozenset[str] = frozenset(),
    primary: str | None = None,
) -> list[dict[str, Any]]:
    """The calls agreement is judged on: calls to ignored (auxiliary) tools removed and, when
    the reference calls the primary tool, only calls to the primary tool kept, so auxiliary calls
    do not mask agreement on the primary one."""
    kept = [c for c in calls if c["name"] not in ignore]
    if primary and any(c["name"] == primary for c in reference):
        kept = [c for c in kept if c["name"] == primary]
    return kept


def certify(
    rows: list[dict[str, Any]],
    proposals: dict[str, dict],
    alpha: float,
    delta: float,
    cap: int,
    noisy: set[str] | None = None,
    normalise_args: bool = False,
    contract: dict[str, Any] | None = None,
    certify_text: bool = False,
    level: str = "args",
    ignore: frozenset[str] = frozenset(),
    primary: str | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    keys, sess, gates = [], [], {"disagree": [], "invalid": []}
    by_level: dict[str, dict[str, list[bool]]] = {}  # slice -> level -> disagreements
    by_ref: dict[str, dict[str, Any]] = {}  # the reference's action -> calls, sessions, wrong
    split = Counter()
    why: dict[str, Any] = {}  # slice -> {"causes": Counter, "example": first disagreeing call}
    forgiven: set[str] = set()
    for r in rows:
        p = proposals[r["key"]]
        ref_calls = compared(r["reference"], r["reference"], ignore, primary)
        cand_calls = compared(p["calls"], r["reference"], ignore, primary)
        cand, ref = _action(cand_calls), _action(ref_calls)
        key = str(routing_slice(r["node"], cand, r["index"]))
        keys.append(key)
        sess.append(r["session"])
        lv = by_level.setdefault(key, {k: [] for k in LEVELS})
        for k in LEVELS:
            lv[k].append(not agree(cand, ref, k))
        wrong = not agree(cand, ref, level)
        d = by_ref.setdefault(str(routing_slice(r["node"], ref, r["index"])),
                              {"calls": 0, "sessions": set(), "disagree": 0})  # fmt: skip
        d["calls"] += 1
        d["sessions"].add(r["session"])
        d["disagree"] += wrong
        if wrong:
            c = cause(cand_calls, ref_calls)
            w = why.setdefault(key, {"causes": Counter(), "example": None})
            w["causes"][c] += 1
            if w["example"] is None:  # first disagreeing call in session order: deterministic
                w["example"] = {"cause": c, **first_difference(cand_calls, ref_calls)}
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
    accepted = [True] * len(keys)  # noisy slices stay visible; they are held, not dropped
    card = report_card(
        keys, sess, gates=gates, alpha=alpha, delta=delta, cap=cap, accepted=accepted
    )
    if not certify_text:  # a text reply agrees on the decision only: its wording is unchecked
        kept = [dataclasses.replace(r, certified=False) if is_text_slice(r.slice) else r
                for r in card.slices]  # fmt: skip
        card = dataclasses.replace(
            card,
            slices=kept,
            certified_call_share=sum(r.call_share for r in kept if r.certified),
            certified_token_share=sum(r.token_share for r in kept if r.certified),
        )
    decision = {
        "replied_instead_of_acting": split["replied_instead_of_acting"]
        / max(1, split["reference_acts"]),
        "acted_instead_of_replying": split["acted_instead_of_replying"]
        / max(1, split["reference_replies"]),
        "agreement": 1 - sum(gates["disagree"]) / max(1, len(rows)),
    }
    decision["forgiven_format_only"] = sorted(forgiven)
    decision["level"] = level
    decision["ignored_tools"] = sorted(ignore)
    decision["primary_tool"] = primary
    decision["by_level"] = {k: {lvl: sum(v) / len(v) for lvl, v in lv.items()}
                            for k, lv in by_level.items()}  # fmt: skip
    decision["by_reference_action"] = {
        k: {"calls": d["calls"], "sessions": len(d["sessions"]),
            "disagreement": d["disagree"] / d["calls"]}
        for k, d in sorted(by_ref.items(), key=lambda kv: -kv[1]["calls"])
    }  # fmt: skip
    decision["reference_contract"] = contract or {
        "conversations_checked": 0,
        "admissible": True,
        "checked": False,
    }
    if noisy:  # held: a slice where the reference disagrees with itself is never certified
        held = [dataclasses.replace(r, certified=False) if r.slice in noisy else r
                for r in card.slices]  # fmt: skip
        card = dataclasses.replace(
            card,
            slices=held,
            certified_call_share=sum(r.call_share for r in held if r.certified),
            certified_token_share=sum(r.token_share for r in held if r.certified),
        )
    if not decision["reference_contract"][
        "admissible"
    ]:  # no swap is certified against a reference that breaks its contract
        card = dataclasses.replace(
            card,
            slices=[dataclasses.replace(s, certified=False) for s in card.slices],
            certified_call_share=0.0,
            certified_token_share=0.0,
        )
    return card, decision, why


LEVEL_TEXT = {
    "args": "the same tools with the same arguments",
    "tools": "the same tool names (arguments not compared)",
    "decision": "the same decision, act or reply (tools and arguments not compared)",
}


def level_line(decision: dict[str, Any]) -> str:
    out = f"Agreement level: **{decision['level']}**, {LEVEL_TEXT[decision['level']]}."
    if decision["ignored_tools"]:
        out += " Left out of the comparison (auxiliary): " + ", ".join(
            decision["ignored_tools"]) + "."  # fmt: skip
    if decision["primary_tool"]:
        out += (f" Primary tool: {decision['primary_tool']} (when the reference calls it, only "
                "calls to it are compared).")  # fmt: skip
    return out + "\n\n"


def replay_stats(rows: list[dict[str, Any]], proposals: dict[str, dict]) -> dict[str, Any]:
    """Latency percentiles and mean tokens per call of the candidate as replayed, beside the
    reference's logged tokens on the same calls."""
    ps = [proposals[r["key"]] for r in rows]
    lat = sorted(p["latency_ms"] for p in ps if p.get("latency_ms") is not None)

    def pct(q: float) -> float | None:
        return lat[min(len(lat) - 1, int(q * len(lat)))] if lat else None

    def mean(xs: list[Any]) -> float | None:
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    tok = ("input_tokens", "output_tokens", "reasoning_tokens")
    return {
        "latency_ms_p50": pct(0.5),
        "latency_ms_p95": pct(0.95),
        "candidate": {k: mean([(p.get("usage") or {}).get(k) for p in ps]) for k in tok},
        "reference_logged": {
            k: mean([r.get("reference_usage", {}).get(k) for r in rows]) for k in tok
        },
    }


def replay_line(stats: dict[str, Any]) -> str:
    if stats["latency_ms_p50"] is None:
        return ""
    c, r = stats["candidate"], stats["reference_logged"]

    def tokens(d: dict[str, Any]) -> str:
        return ", ".join(f"{d[k]:,.0f} {k.split('_')[0]}" for k in d if d[k] is not None) or "-"

    return (f"Latency and tokens (replay endpoint, not production): P50 "
            f"{stats['latency_ms_p50']:,.0f} ms, P95 {stats['latency_ms_p95']:,.0f} ms per call; "
            f"candidate tokens per call {tokens(c)}; the reference's logged tokens on the same "
            f"calls {tokens(r)}.\n\n")  # fmt: skip


def reasoning_line(carry: bool, stats: dict[str, Any]) -> str:
    rt = stats["candidate"].get("reasoning_tokens")
    return (
        "Reasoning: never scored. The reference's reasoning from earlier turns is "
        + ("replayed in the history (--carry-reasoning)" if carry else "left out of the history")
        + "; at the re-asked turn it is never shown (it would give the decision away)."
        + (f" Candidate reasoning: {rt:,.0f} tokens per call." if rt else "")
        + "\n\n"
    )


ALL_HELD = (
    "**Every slice is held by reference noise.** Try `--level tools`: free-text arguments "
    "(messages, summaries) cannot certify by agreement when the reference words them differently "
    "each time. Certification by effect needs an environment that can replay a logged state "
    "(`graduate claim-b`).\n\n"
)


def level_tables(decision: dict[str, Any]) -> str:
    """Observed disagreement at all three levels per slice, and a diagnostic table sliced by the
    reference's action (not routable: a router cannot see the reference's answer)."""
    rows = ["\n## Disagreement at each level\n", "| slice | args | tools | decision |",
            "|---|---|---|---|"]  # fmt: skip
    rows += [f"| {k} | {v['args']:.1%} | {v['tools']:.1%} | {v['decision']:.1%} |"
             for k, v in sorted(decision["by_level"].items())]  # fmt: skip
    rows += ["\n## By the reference's action (diagnostic, not certified)\n",
             "Where the reference's own action is the slice: what traffic the disagreements fall "
             "on. A router cannot see the reference's answer, so certification uses the "
             "candidate's action (the table above the legend).\n",
             "| reference action | calls | conversations | disagreement |",
             "|---|---|---|---|"]  # fmt: skip
    rows += [f"| {k} | {d['calls']} | {d['sessions']} | {d['disagreement']:.1%} |"
             for k, d in decision["by_reference_action"].items()]  # fmt: skip
    return "\n".join(rows) + "\n"


def is_text_slice(key: str) -> bool:
    """A slice of text replies: agreement is on the decision (reply, not act), not the wording."""
    return "/text/" in key or key.endswith("/text")


def slice_verdicts(
    card: Any,
    alpha: float,
    why: dict[str, Any] | None = None,
    calibration: dict | None = None,
    noisy: set[str] | None = None,
) -> list[dict]:
    """Per slice: the verdict and the smallest disagreement margin this evidence can resolve
    (zero-event bound, about 3 / n sessions). A slice whose disagreement is significantly at or
    above alpha (the card's betting test mirrored, H0: rate <= alpha, rejects; sessions are the
    unit) is not certified and more evidence will not change
    that; otherwise a slice not yet certified "cannot be decided with this evidence" (not a
    failure), with the planner's range when its observed rate is below alpha."""
    out = []
    for r in card.slices:
        detectable = 3 / r.sessions if r.sessions else 1.0
        a = card.alphas[r.binding]
        if r.slice in (noisy or set()):  # the reference disagrees with itself at or above alpha
            verdict = "held: reference noise (cannot be certified by agreement)"
        elif r.certified:
            verdict = "certified"
        elif is_text_slice(r.slice):
            verdict = "decision only (wording unchecked)"
        elif r.above_alpha:
            verdict = (
                f"not certified: disagreement {r.rate:.0%} is significantly above alpha {a:g}; "
                "more conversations will not help"
            )
        elif r.rate >= a:
            verdict = (
                f"cannot be decided with this evidence: disagreement {r.rate:.0%} is not "
                f"significantly above alpha {a:g}"
            )
        elif r.plan:
            verdict = f"cannot be decided with this evidence: {plan_text(r.plan)}"
        else:
            verdict = "not certified"
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


def _names(x: Any) -> str:
    return ", ".join(x) if isinstance(x, list) else str(x)


def candidate_path(args: Any) -> str:
    """How the candidate was asked: cards are comparable only between the same paths."""
    if args.prompt_mode == "completions":
        tok = args.tokenizer or args.model
        return (
            f"completions (Graduate renders {tok}'s chat template with thinking off, prompt "
            f"fitted to {args.prompt_tokens} tokens, {COMPLETION_TOKENS} new tokens, replies read "
            "by graduate.prompting's parsers)"
        )
    return (
        "chat (the server's chat template and tool-call parser; not comparable to cards made "
        "with --prompt-mode completions)"
    )


def contract_line(c: dict[str, Any]) -> str:
    """card.md: the reference's tool contract. Above the limit nothing is certified."""
    if not c["conversations_checked"]:
        return (
            "Reference tool contract: **not checked**. The traces carry no tool list, so calls to "
            "tools outside the agent's list cannot be seen. Export the agent's tools with each "
            "session to check it; until then the reference's admissibility is unknown.\n"
        )
    top = ", ".join(f"`{n}` {k}" for n, k in c["top_violations"])
    if not c["admissible"]:
        return (
            f"**Reference not admissible:** in {c['violating_share']:.0%} of "
            f"{c['conversations_checked']:,} conversations it calls tools that are not in its tool "
            f"list ({top}). Above {CONTRACT_LIMIT:.0%}, no swap is certified against it; fix the "
            "reference (prompt or tool list) first.\n\n"
        )
    return (
        f"Reference tool contract: {c['violating_share']:.1%} of {c['conversations_checked']:,} "
        "conversations call tools outside the tool list" + (f" ({top})" if top else "") + ".\n\n"
    )


def why_table(rows: list[dict[str, Any]], forgiven: list[str] | None = None) -> str:
    """card.md: one line per slice with disagreements that is not certified, or whose format-only
    differences --normalise-args counted as agreement (marked as such)."""
    forgiven = set(forgiven or [])
    failing = [r for r in rows if r.get("disagreement_causes")
               and (r["verdict"] != "certified" or r["slice"] in forgiven)]  # fmt: skip
    if not failing:
        return ""
    head = "\n### Why slices disagree (one cause per call; example values redacted: digits as 0)\n"
    lines = [head, "| slice | causes | example |", "|---|---|---|"]
    for r in failing:
        causes = ", ".join(
            f"{c} {v:.0%}"
            + (
                " (counted as agreement: --normalise-args)"
                if c == "format-only argument" and r["slice"] in forgiven
                else ""
            )
            for c, v in r["disagreement_causes"].items()
        )
        e = r["example"]
        if "argument" in e:
            ref, cand = (json.dumps(redact(e[k])) for k in ("reference_value", "candidate_value"))
            ex = f"`{e['tool']}.{e['argument']}`: reference {ref}, candidate {cand}"
        else:
            ex = f"reference {_names(e['reference'])}, candidate {_names(e['candidate'])}"
        lines.append(f"| {r['slice']} | {causes} | {ex} |")
    return "\n".join(lines) + "\n"


def claim_b(args: Any, out: Path) -> None:
    """`graduate claim-b`: per-slice and per-session certificate by effect from fork rows."""
    from graduate.certify.fork import card_pairs, fork_card, routing_table

    need = ("state", "branch", "task_id", "slice", "success")
    rows = []
    for path in args.forks.split(","):
        if not Path(path).is_file():
            _fail(f"{path}: no such file")
        for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                _fail(f"{path}, line {n}: not JSON")
            miss = [k for k in need if k not in r] if isinstance(r, dict) else list(need)
            if miss:
                _fail(f"{path}, line {n}: missing {', '.join(miss)} (each row: {', '.join(need)}; "
                      "branch is T1, T2 or S)")  # fmt: skip
            rows.append(r)
    try:
        serving = json.loads(args.serving)
    except ValueError:
        _fail('--serving must be JSON, e.g. \'{"model": "ref"}\'')
    ns = argparse.Namespace(pool_positions=not args.by_position)
    pairs = card_pairs(rows, ns)
    if not pairs:
        _fail("no complete forks: a state needs all three branches (T1, T2 and S) with a success")
    card = fork_card(pairs, session_margin=args.session_margin, cap=args.cap, serving=serving)
    card["routing"] = routing_table(card)
    (out / "claim_b.json").write_text(json.dumps(card, indent=2, default=str))
    d, f = card["delta_turn"], card.get("reference_floor") or {}
    md = [f"# Claim B by effect: {card['n_states']:,} forked states",
          f"\nPer routed turn: signed net harm <= {d:.1%} (session margin "
          f"{args.session_margin:.0%} / cap {args.cap}); per session <= {args.session_margin:.0%}."
          " Reference self-noise floor (T1 vs T2, the reference against itself):"
          f" h_T {f.get('h_T', float('nan')):+.1%}, flips "
          f"{f.get('change_T', float('nan')):.1%}.",
          "\n| candidate proposes | states (tasks) | reference success | net harm (upper) | "
          "finite-sample / t / bootstrap | session harm (upper) | detectable harm | verdict |",
          "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for k, t in card["by_proposed_slice"].items():
        md.append(f"| {k} | {t['n_states']} ({t['n_tasks']}) | {t['p_T']:.0%} | "
                  f"{t['harm_task']:+.1%} ({t['harm_upper']:.1%}; states {t['harm']:+.1%}) | "
                  f"{t['harm_upper_finite']:.1%} / {t['harm_upper_t']:.1%} / "
                  f"{t['harm_upper_bootstrap']:.1%} | {t['session_harm_upper']:.1%} | "
                  f"{t['detectable_harm']:.1%} | {t['verdict']} |")  # fmt: skip
    if card["untested_proposed_slice"]:
        md.append(f"\nUntested (fewer than 30 states, kept out of the multiple-comparison family): "
                  f"{', '.join(card['untested_proposed_slice'])}.")  # fmt: skip
    md.append("\nSlices are domain/kind/tool"
              + (" with position buckets (--by-position)." if args.by_position
                 else "; positions are pooled (--by-position keeps them).")
              + " Never route: the harm is likely (above 0) and its upper bound exceeds half the "
              "reference's own success on the slice, so the router refuses it; not certified: "
              "the bound is above delta, without that evidence of harm.")  # fmt: skip
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
    rows = []
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append((n, json.loads(line)))
        except ValueError:
            _fail(f"{path}, line {n}: not JSON")
    out = []
    for n, r in rows:
        if not isinstance(r, dict) or "task_id" not in r or "success" not in r:
            _fail(f"{path}, line {n}: each row needs task_id and success (got "
                  f"{sorted(r) if isinstance(r, dict) else type(r).__name__})")  # fmt: skip
        v = r["success"]
        if v is None:
            continue
        if isinstance(v, bool) or (isinstance(v, int | float) and v in (0, 1)):
            out.append((str(r["task_id"]), bool(v)))
        else:  # "yes" / "false" would otherwise read as true
            _fail(f"{path}, line {n}: success must be true/false or 0/1, got {v!r}")
    return out


def compare_cmd(args: Any, out: Path) -> None:
    """`graduate compare`: paired, task-clustered regret with its minimum detectable regret."""
    from graduate.certify.session import compare, per_task

    r = compare(per_task(outcomes(args.reference, args.outcome)),
                per_task(outcomes(args.candidate, args.outcome)), margin=args.margin)  # fmt: skip
    (out / "compare.json").write_text(json.dumps(r, indent=2))
    if not r["tasks"]:
        _fail("no paired tasks: the two files share no task_id (rows need task_id and success)")
    unpaired = r["tasks_only_reference"] + r["tasks_only_candidate"]
    if unpaired:
        print(f"warning: {r['tasks_only_reference']} tasks only in the reference file and "
              f"{r['tasks_only_candidate']} only in the candidate file are left out (not paired)",
              file=sys.stderr)  # fmt: skip
    text = (f"# Session-level comparison: {r['tasks']} paired tasks\n\n"
            f"Reference success {r['reference_success']:.1%}, candidate "
            f"{r['candidate_success']:.1%}. Regret {r['regret']:+.1%} (one-sided 95% upper bound "
            f"{r['regret_upper']:+.1%}; candidate worse on {r['tasks_worse']} tasks, better on "
            f"{r['tasks_better']}). Smallest regret this evidence could detect: "
            f"{r['detectable_regret']:.1%}. Bounds: finite-sample upper "
            f"{r['regret_upper_finite']:+.1%} and lower {r['regret_lower_finite']:+.1%} "
            f"(decides), t {r['regret_upper_t']:+.1%} (asymptotic), bootstrap "
            f"{r['regret_upper_bootstrap']:+.1%}. Non-vacuity: the "
            f"reference succeeds {r['reference_success']:.0%} "
            f"({'at least' if r['non_vacuous'] else 'below'} 2 x margin).\n\n**{r['verdict']}** "
            f"at a margin of {r['margin']:.0%}.\n"
            + (f"\n{unpaired} tasks were in one file only and are left out.\n" if unpaired
               else ""))  # fmt: skip
    (out / "compare.md").write_text(text)
    print(text)


CONTRACT_LIMIT = 0.20  # share of conversations above which the reference is not admissible

CARD_LEGEND = """How to read the table: a slice is one kind of decision the candidate proposes
(agent node / call type / tool / position bucket in the conversation). sess = conversations
counted; calls = calls in the slice; counted = the calls the test uses (the first `cap` per
conversation); rate = observed disagreement on the counted calls, with the reference (so a slice
can read 100% when later, uncounted calls agreed; the causes below use the counted calls too);
upper = its 1 - delta upper bound; detect = the smallest disagreement rate this many conversations
can rule out (about 3 / sess; "-" under 3 conversations, when none can); p = p-value after the
multiplicity correction; share = fraction of all calls. GRADUATED = certified at alpha: a router
may send this slice to the candidate. "-" = not certified; "not enough evidence" says how many more
conversations a slice would likely need at its observed rate. "Will not help" means the same
betting test, mirrored (H0: disagreement <= alpha), rejects on the capped per-conversation counts;
it is anytime-valid, so rerunning as conversations arrive does not inflate it. Text-reply slices
read "decision only (wording unchecked)": agreement there is on the decision to reply, not on the
reply's wording, so they stay out of the certified share unless --certify-text-slices. "held:
reference noise" = the reference disagrees with itself on the slice at or above alpha
(--reference-base-url), so no candidate can certify it by agreement.
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
    c.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="parallel requests (8 suits a server; use 1 on a laptop)",
    )
    c.add_argument(
        "--certify-text-slices",
        action="store_true",
        help="count text-reply slices as certifiable (their wording is not checked)",
    )
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
    c.add_argument(
        "--prompt-mode",
        choices=["chat", "completions"],
        default="chat",
        help="chat: the server's chat template and tool parser (Ollama, vLLM, any "
        "OpenAI-compatible server); completions: Graduate renders the model's chat template "
        "(thinking off), calls /completions and parses the reply itself",
    )
    c.add_argument(
        "--tokenizer", default=None, help="completions mode: tokenizer id (default: --model)"
    )
    c.add_argument(
        "--prompt-tokens", type=int, default=12288, help="completions mode: prompt budget"
    )
    c.add_argument("--max-model-len", type=int, default=16384, help="completions mode: context")
    c.add_argument("--out", default="graduate-report", help="output folder")
    c.add_argument(
        "--fresh", action="store_true", help="ignore cached answers in --out and ask again"
    )
    c.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="keep conversations that repeat an earlier one exactly (default: dropped, since "
        "a repeat counts the same evidence twice)",
    )
    c.add_argument(
        "--carry-reasoning",
        action="store_true",
        help="replay the reference's reasoning from earlier turns in the history (default: "
        "left out); the re-asked turn's own reasoning is never shown",
    )
    c.add_argument(
        "--level",
        choices=LEVELS,
        default="args",
        help="what counts as agreement: args (same tools and arguments), tools (same tool "
        "names; arguments not compared), decision (act vs reply); the card prints all three",
    )
    c.add_argument(
        "--ignore-tools",
        default="",
        help="comma-separated auxiliary tools left out of the comparison on both sides",
    )
    c.add_argument(
        "--primary-tool",
        default=None,
        help="when the reference calls this tool, compare only the calls to it",
    )
    c.add_argument(
        "--reference-noise-cache",
        default=None,
        help="folder for the reference's measured noise, reused across candidates "
        "(default: ~/.cache/graduate/reference-noise)",
    )
    c.add_argument(
        "--normalise-args",
        action="store_true",
        help="count format-only argument differences (e.g. 'W1000' for '#W1000') as agreement; "
        "default: reported, not forgiven",
    )
    b = sub.add_parser(
        "claim-b",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="certify by effect from fork outcomes (needs an environment that can be reset to a "
        "logged state)",
    )
    b.add_argument("forks", help="comma-separated JSONL files of fork rows (T1, T2, S branches)")
    b.add_argument("--session-margin", type=_fraction, default=0.10,
                   help="harm allowed per session, a fraction (0.10 = 10 points)")  # fmt: skip
    b.add_argument(
        "--cap", type=int, default=1, help="routed turns per session (delta = margin/cap)"
    )
    b.add_argument("--by-position", action="store_true", help="keep position buckets in slices")
    b.add_argument("--serving", default="{}", help="JSON: the reference's serving configuration")
    b.add_argument("--out", default="graduate-claim-b")
    m = sub.add_parser("compare", help="session level: candidate vs reference on the same tasks",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # fmt: skip
    m.add_argument("reference", help="JSONL of {task_id, success} rows (or a tau2 results.json)")
    m.add_argument("candidate", help="the same for the candidate or the hybrid")
    m.add_argument("--margin", type=_fraction, default=0.10,
                   help="allowed regret per session, a fraction (0.10 = 10 points)")  # fmt: skip
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
        sessions = list(ingest(args.traces, args.preset, limit=args.limit))
        dropped = 0
        if not args.keep_duplicates:  # a repeated conversation would count its evidence twice
            sessions, dropped = dedupe_sessions(sessions)
        rows, skipped = states(sessions, args.carry_reasoning)
    except FileNotFoundError:
        _fail(f"no such file: {args.traces}")
    except json.JSONDecodeError as e:
        _fail(f"{args.traces} is not valid JSON ({e.msg})")
    except (KeyError, TypeError, ValueError) as e:
        _fail(
            f"{args.traces} does not look like --preset {args.preset} ({type(e).__name__}: {e}); "
            "check --preset (see README: trace formats)"
        )
    complete = completeness(sessions)
    dups = near_duplicate_sessions(sessions)
    if dropped:
        dups = (dropped, dups[1])
    complete_md = "\n".join(completeness_lines(complete, *dups, dropped=bool(dropped)))
    print(f"ingest completeness:\n{complete_md}", file=sys.stderr)
    left = ", ".join(f"{n} {k}" for k, n in sorted(skipped.items())) or "none"
    print(f"{len(rows):,} reference calls (left out: {left})", file=sys.stderr)
    if not rows:
        _fail(f"no reference calls found in {args.traces} with --preset {args.preset}")
    cache = out / "proposals.jsonl"
    stamp = {"_cache": 1, "model": args.model, "base_url": args.base_url, "traces": args.traces}
    renderer = None
    if args.prompt_mode == "completions":  # a different path gives different answers: own cache
        stamp["prompt_mode"] = "completions"
        stamp["tokenizer"] = args.tokenizer or args.model
        renderer = load_renderer(stamp["tokenizer"], args.prompt_tokens, args.max_model_len)
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
    print(f"{len(rows) - len(todo):,} cached answers reused, {len(todo):,} to ask", file=sys.stderr)
    # a run stopped from here on must not leave the previous card looking current
    for name in ("card.md", "card.json"):
        if (out / name).exists():
            (out / name).replace(out / f"previous-{name}")
    (out / "card.md").write_text(
        f"# Run in progress or stopped\n\nThis run started on {len(rows):,} calls "
        f"({len(todo):,} still to ask) and has not finished. Rerun the same command to finish it: "
        "cached answers are reused. The previous card, if any, is previous-card.md.\n"
    )

    def one(r: dict[str, Any]) -> dict[str, Any]:
        try:
            return {
                "key": r["key"],
                **(
                    ask_completions(args.base_url, args.model, key, r["chat"], r["tools"], renderer)
                    if renderer
                    else ask(args.base_url, args.model, key, r["messages"], r["tools"])
                ),
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
    noise_cache: Path | None = None
    ignore = frozenset(t.strip() for t in args.ignore_tools.split(",") if t.strip())
    if args.reference_base_url and args.reference_model:
        sample = rows[:: max(1, len(rows) // args.reference_samples)][: args.reference_samples]
        # the reference's own answers depend on the reference and the states only, so they are
        # measured once and reused by every candidate run on the same traces
        noise_cache = Path(
            args.reference_noise_cache
            or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
            / "graduate" / "reference-noise"
        ) / (
            stable_hash([args.reference_model, args.reference_base_url,
                         [(r["key"], r["messages"], r["reference"]) for r in sample]])
            + ".json"
        )  # fmt: skip
        answers: dict[str, Any] = {}
        if noise_cache.exists() and not args.fresh:
            with contextlib.suppress(ValueError):
                answers = json.loads(noise_cache.read_text())
        todo_ref = [r for r in sample if r["key"] not in answers]
        with ThreadPoolExecutor(args.concurrency) as pool:
            for r, p in zip(
                todo_ref,
                pool.map(
                    lambda r: ask(
                        args.reference_base_url,
                        args.reference_model,
                        ref_key,
                        r["messages"],
                        r["tools"],
                    ),
                    todo_ref,
                ),
                strict=True,
            ):
                answers[r["key"]] = p["calls"]
        if todo_ref:
            noise_cache.parent.mkdir(parents=True, exist_ok=True)
            noise_cache.write_text(json.dumps(answers))
        again = {}
        for r in sample:
            ref_calls = compared(r["reference"], r["reference"], ignore, args.primary_tool)
            re_calls = compared(answers[r["key"]], r["reference"], ignore, args.primary_tool)
            again[r["key"]] = not agree(_action(re_calls), _action(ref_calls), args.level)
        by: dict[str, list[bool]] = {}
        for r in sample:
            k = str(routing_slice(r["node"], _action(r["reference"]), r["index"]))
            by.setdefault(k, []).append(again[r["key"]])
        noise = {k: sum(v) / len(v) for k, v in by.items()}
        noisy = {k for k, v in noise.items() if v >= args.alpha}

    card, decision, why = certify(
        rows,
        proposals,
        args.alpha,
        args.delta,
        args.cap,
        noisy,
        args.normalise_args,
        getattr(states, "contract", None),
        args.certify_text_slices,
        args.level,
        ignore,
        args.primary_tool,
    )
    card, calibration = plan_history(card, out / "planner.json", fresh=args.fresh)
    stats = replay_stats(rows, proposals)
    all_held = bool(noisy) and all(r.slice in noisy for r in card.slices)
    report = {
        "candidate": args.model,
        "candidate_path": candidate_path(args),
        "reference_calls": len(rows),
        "failed_requests": errors,
        "left_out": dict(skipped),
        "alpha": args.alpha,
        "delta": args.delta,
        "cap": args.cap,
        "certified_call_share": card.certified_call_share,
        "certified_slices": [r.slice for r in card.slices if r.certified],
        "slices": slice_verdicts(card, args.alpha, why, calibration, noisy),
        "normaliser": {"version": NORMALISER_VERSION, "rule": NORMALISER},
        "format_only_counted_as_agreement": decision["forgiven_format_only"],
        "reference_contract": decision["reference_contract"],
        "ingest_completeness": {
            **complete,
            "exact_duplicate_conversations": dups[0],
            "exact_duplicates_dropped": bool(dropped),
            "near_duplicate_conversations": dups[1],
        },
        "decision_split": decision,
        "reference_self_disagreement": noise,
        "reference_noise_cache": str(noise_cache) if noise_cache else None,
        "level": args.level,
        "ignored_tools": sorted(ignore),
        "primary_tool": args.primary_tool,
        "disagreement_by_level": decision["by_level"],
        "replay": stats,
        "carry_reasoning": args.carry_reasoning,
        "by_reference_action": decision["by_reference_action"],
        "too_noisy_to_certify_by_agreement": sorted(noisy),
    }
    (out / "card.json").write_text(json.dumps(report, indent=2, default=str))
    labels = {
        r["slice"]: r["verdict"].split(" (cannot")[0]
        for r in report["slices"]
        if r["verdict"].startswith(("decision only", "held"))
    }
    md = (
        f"# Graduate report card: {args.model}\n\n"
        + f"Candidate path: {candidate_path(args)}.\n\n"
        + f"**Ingest completeness** ({args.preset}):\n{complete_md}\n\n"
        + (
            f"**{errors:,} requests failed** and are left out; rerun to retry.\n\n"
            if errors
            else ""
        )
        + contract_line(decision["reference_contract"])
        + (ALL_HELD if all_held else "")
        + level_line(decision)
        + replay_line(stats)
        + reasoning_line(args.carry_reasoning, stats)
        + f"{len(rows):,} reference calls; certified share **{card.certified_call_share:.1%}** "
        f"at alpha {args.alpha:g}, confidence {1 - args.delta:.0%}.\n\n"
        f"Replied instead of acting: {decision['replied_instead_of_acting']:.1%}; acted "
        f"instead of replying: {decision['acted_instead_of_replying']:.1%}; overall "
        f"agreement {decision['agreement']:.1%}. "
        + (
            f"Format-only argument differences (normaliser v{NORMALISER_VERSION}) are reported "
            "below, not forgiven; --normalise-args forgives them. "
            if not decision["forgiven_format_only"]
            else ""
        )
        + (
            "**Format-only differences counted as agreement** (--normalise-args) for slices: "
            + ", ".join(decision["forgiven_format_only"])
            + ". "
            if decision["forgiven_format_only"]
            else ""
        )
        + f"A call is every model turn: a tool call or "
        f"a text reply. Text replies agree by decision only (reply vs act); their wording is not "
        f"compared.\n\n```\n{card.render(top=40, labels=labels)}\n```\n\n{CARD_LEGEND}"
        + f"Argument normaliser v{NORMALISER_VERSION}: {NORMALISER}.\n"
        + why_table(report["slices"], decision["forgiven_format_only"])
    )
    md += level_tables(decision)
    if noise:
        md += (
            "\nReference self-disagreement per slice (re-asked): "
            + ", ".join(f"{k} {v:.0%}" for k, v in sorted(noise.items()))
            + f" (measured once, cached in {noise_cache}; reused by any candidate on these "
            "traces, --fresh measures again)\n"
        )
    (out / "card.md").write_text(md)
    print(md)
    print(f"wrote {out / 'card.md'} and {out / 'card.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
