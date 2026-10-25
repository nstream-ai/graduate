# Graduate architecture

## Data flow of `graduate certify`

```
traces file (JSON / JSONL / Parquet)
  └─ ingest.read_records ─ PRESETS[--preset](record, source) ─► core.Session (schema v0.3)
       └─ cli.states(sessions)
            one row per reference call: input messages, tools, the reference's tool calls
            skips: calls not made by the teacher, unparsable reference calls, calls to tools
            outside the session's tool list (counted, and checked against CONTRACT_LIMIT)
       └─ cli.ask / cli.ask_completions  (ThreadPoolExecutor, --concurrency)
            candidate answers each state; cached in <out>/proposals.jsonl
       └─ optional: the reference re-asked on --reference-samples states ─► noisy slices
       └─ cli.certify(rows, proposals, ...)
            slice = routing_slice(node, candidate action, call index)
            gates: disagree (core.canonical.agree at level "args"), invalid (core.validity)
            causes.cause / first_difference ─► "Why slices disagree"
            certify.card.report_card(keys, sessions, gates, alpha, delta, cap)
            then: text slices un-certified, noisy slices held, inadmissible contract ─► nothing certified
       └─ cli.plan_history ─► <out>/planner.json
       └─ cli.slice_verdicts ─► card.json, card.md (with legend)
```

`claim-b` (`cli.claim_b` → `certify/fork.py`) and `compare` (`cli.compare_cmd` →
`certify/session.py`) read pre-computed outcome files and call no model.

## core/

- `schema.py`: Pydantic models with `extra="forbid"`. A **Call** is a maximal run of consecutive
  assistant messages (`calls_from_messages`). The input of call k is `messages[:call.start]`.
  Static context (`system`, `tools`), dynamic context (`messages`) and hidden context
  (`Message.reasoning`, stored but never scored) are kept apart. `Call.produced_by` is
  `"teacher"` or `"student"` (hybrid runs). `Flags` records data-quality facts from ingest.
- `canonical.py`: `Action` is a sorted multiset of (tool name, canonical JSON args), or text.
  Normalisation is deliberately conservative: sorted keys, stripped strings, integral floats
  become ints, and nothing semantic. Agreement levels: `decision` < `tools` < `args`. Unparsed
  args never agree at `args`.
- `slicing.py`: `Slice(node, call_type, tool, position)`. The call type is `tool`,
  `parallel_tool` or `text`. Position buckets are `1`, `2-3`, `4-10`, `11-30` and `31+`.
- `split.py`: splits by session. They are time-ordered when every session has a timestamp, and
  otherwise use a stable hash order flagged `time_ordered=False`. `group=` keeps repeated trials
  of one task together. `rolling_rounds` gives rotating held-out windows.
- `diff.py`: failure types (`wrong_decision`, `wrong_tool`, `missing_field`, `extra_field`,
  `wrong_value`, `unparsed`), the verbosity and repetition gates, free-text field detection, and
  look-alike Unicode.
- `validity.py`: whether a proposed call satisfies the tool schema.
- `dedup.py`: exact-state dedup and train/test leakage reports.

## ingest/

- `io.py`: reads JSON, JSONL and Parquet records (Parquet needs the `[parquet]` extra).
- `converters/openai_chat.py`: `convert_messages` handles OpenAI messages (`tool_calls` or the
  legacy `function_call`). `converters/hermes.py`: `convert_sharegpt` handles `<tool_call>`
  blocks. `converters/common.py` holds the shared helpers.
- `repair.py`: argument parsing and template-token stripping. Every repair is recorded on the
  `ToolCall` (`parse`, `repairs`) and in `Flags`.
- `presets.py`: `openai` (default), `hermes`, `toucan`, `nemotron`, `nemotron-swe`,
  `swe-rebench` and `tau2`. The tau2 preset drops user-side device tool calls (dual control) and
  the scripted greeting, keeps the DB check apart from the full reward, and never counts an
  infrastructure error as a failure.

## prompting/

- `chat.py`: `to_chat(messages)` and `tools_spec(session)` build what the candidate sees.
- `render.py`: renders a prompt from the candidate's own chat template (transformers), fits it to
  `--max-model-len`, and supplies stop tokens.
- `decode.py`: `parse_completion(text, tools)` returns `(reply text, [ToolCall])`. It handles
  OpenAI JSON, Qwen XML calls typed by the schema, and Gemma syntax. `gemma.py` and `harmony.py`
  are format-specific. In harmony, analysis is dropped, a commentary message to `functions.X` is
  a tool call, and final is the reply.
- `--prompt-mode chat` (the default) relies on the server's template and tool parser.
  `completions` uses this package.

## certify/

- `card.py`: `report_card` takes per-call gate arrays and returns a card with per-slice rows
  (`sessions`, `calls`, `rate`, `upper`, `certified`, `above_alpha`, `binding` gate, `plan`) and
  the certified call and token shares. `more_sessions_needed` / `plan_text` simulate futures by
  resampling whole conversations.
- `causes.py`: exactly one cause per disagreeing call, first match in a fixed order, so the shares
  sum to 100%. `--normalise-args` forgives "format-only argument" differences, and the card says
  so. `redact` is used for `card.md`.
- `fork.py`: claim B by effect, covering pairing T1/T2/S rows per state, harm per slice, e-BH,
  never-route, vacuity, minimum detectable harm and `cap_max_informative`.
- `session.py`: `compare`, with paired per-task regret and the verdict ladder.
- `stats/`: see `statistics.md`.

## Tests

- `tests/test_cli.py` runs the CLI end to end against an in-process fake OpenAI-compatible HTTP
  server. Use the same pattern for new CLI behaviour; never call a real model in tests.
- `tests/certify/` has a file per stats module plus card, causes, fork and session tests.
  `hypothesis` is available for property tests.
- `tests/ingest/test_ingest.py` has real-sample tests that need `GRADUATE_OPEN_DATA` pointing at
  local copies of open datasets. They skip otherwise.

## Outputs (`--out`, default `graduate-report/`)

`card.md` (human, redacted, ends with a legend), `card.json` (machine-readable, with raw
examples), `proposals.jsonl` (resumable cache), `planner.json` (planner memory). `--fresh`
ignores the cache and the planner history.
