---
name: graduate
description: Work on Graduate (github.com/nstream-ai/graduate), the open-source Python tool that certifies which slices of an agent's logged LLM calls a cheaper candidate model can take over, with finite-sample statistical bounds. Use when changing, testing, debugging, documenting or releasing code in this repo (src/graduate, the `graduate certify` / `claim-b` / `compare` CLI, trace presets, prompt parsers, the report card, the statistics) or when running Graduate on traces.
---

# Graduate

Graduate takes **logged production agent traces** (the reference model's conversations) and a
**candidate model** behind an OpenAI-compatible endpoint. It asks the candidate every logged
state and writes a **report card**: the slices of traffic where the candidate's disagreement
with the reference is provably below `alpha`. The intended deployment is a cascade: the candidate
answers every turn, and its answer is kept only on certified slices.

The product is the certificate, so **statistical validity and honest wording come before
features.** A wrong "certified" is the worst bug this repo can have.

## Repo map

| Path | What it is |
|---|---|
| `src/graduate/cli.py` | All three subcommands (`certify`, `claim-b`, `compare`), argument parsing, the candidate HTTP client, caching, and writing `card.md`/`card.json` |
| `src/graduate/core/` | Trajectory schema v0.3 (`Session`, `Call`, `ToolCall`), canonical actions and agreement, slices, session splits, failure types, validity, dedup |
| `src/graduate/ingest/` | Trace formats to `Session`: `presets.py` (one function per source), `converters/` (OpenAI chat, Hermes/ShareGPT), `repair.py` (audited argument repairs), `validate.py` / `catalog.py` (ingest reports) |
| `src/graduate/prompting/` | Rendering the candidate's prompt and parsing replies for `--prompt-mode completions` (Qwen XML, Gemma, harmony/gpt-oss) |
| `src/graduate/certify/` | `card.py` (per-slice report card and planner), `causes.py` (why a call disagrees), `fork.py` (claim-b), `session.py` (compare) |
| `src/graduate/certify/stats/` | Pure statistical primitives: bounds, betting tests, multiplicity, planning, sampling, task-level bounds |
| `src/graduate/train/` | Stub that raises: renamed to `prompting` in v0.2. Keep it raising |
| `tests/` | Mirrors `src/` (`core/`, `ingest/`, `prompting/`, `certify/`, `test_cli.py`) |
| `examples/` | `traces.jsonl` (3 made-up conversations) and `mock_candidate.py` (stdlib fake endpoint) |
| `index.html`, `grid/` | The website, served by GitHub Pages from `main` |

For the data flow, module responsibilities and extension points, read
[reference/architecture.md](reference/architecture.md). Before touching anything that changes
a verdict, a bound or a card's wording, read
[reference/statistics.md](reference/statistics.md).

## Commands

```bash
pip install -e ".[dev]"          # or: uv run --extra dev <cmd>
ruff check src tests             # must pass, CI runs it
pytest -q                        # about 90 s; 9 open-data tests skip unless GRADUATE_OPEN_DATA is set
graduate certify --help          # CI smoke check
```

To see the whole pipeline end to end without a model:

```bash
python examples/mock_candidate.py &
graduate certify examples/traces.jsonl --base-url http://localhost:8000/v1 --model mock --out report/
cat report/card.md
```

CI (`.github/workflows/ci.yml`) runs ruff, `graduate certify --help` and pytest on Python 3.12
and 3.13.

## Rules that are not obvious from the code

1. **Sessions are the unit, not calls.** Calls inside a conversation are dependent. Split, cap,
   resample and bound by session or task. Never treat calls as i.i.d.
2. **Slices are keyed by the candidate's proposed action** (`core.slicing.routing_slice`), not
   by the reference's. A router cannot see the reference's answer. `slice_of` is only for
   describing traffic.
3. **No CLT for decisions.** Certification uses finite-sample bounds (Clopper–Pearson, betting /
   e-values). t and bootstrap bounds may be printed "for information" but must never decide.
4. **Every slice is in the multiplicity family.** BH across slices in `certify`, e-BH in
   `claim-b`. Fix the family before looking at results, and enter ineligible slices with e = 1.
5. **Text replies agree on the decision only.** Wording is never compared, so text slices read
   "decision only (wording unchecked)" and stay out of the certified share unless
   `--certify-text-slices` is passed.
6. **The reference's flaws block certification and must not be hidden.** Noisy slices read
   "held: reference noise". A reference that calls tools outside its tool list in more than 20%
   of conversations (`CONTRACT_LIMIT`) certifies nothing. Traces with no tool list read
   "contract: not checked".
7. **Undecided is not failed.** Keep the distinction between "cannot be decided with this
   evidence", "not certified", "significantly above alpha: more conversations will not help",
   "worse than the margin", "vacuous" and "never route". Users act on these words.
8. **Privacy:** `card.md` redacts example values (`causes.redact`) and `card.json` keeps raw
   values. Anything meant to be shared (issues, docs) uses `card.md`.
9. **Caching:** `proposals.jsonl` is tied to traces, `--model` and `--base-url`. Failed requests
   are not cached. `planner.json` keeps the "more conversations" estimate from moving closer
   across reruns unless the rate actually improved.
10. **Core dependencies stay at NumPy, SciPy and Pydantic.** pyarrow (`[parquet]`) and
    transformers (`[completions]`) are optional extras, imported lazily with an install hint.

## Code style

- Python 3.12+, `from __future__ import annotations`, ruff with line length 100 (rules E, F, I,
  B, UP, N, SIM, RUF, NPY).
- Each module opens with a docstring that explains *why*: the claim it makes, what it
  deliberately does not do, and what it was changed from. Match that density. Short inline
  comments explain decisions, not mechanics.
- Dense dict or verdict literals are written compactly and marked `# fmt: skip`.
- `certify/stats/` is pure: no I/O, no global state, randomness only through an explicit `seed`.
- Errors a user can trigger (bad input file, a field that is missing, a margin out of range) stop
  with a plain message through `_fail` that names the file, line or flag. They never surface as a
  traceback or produce an empty card with exit 0.
- CLI fractions go through `_fraction`, so `10` is rejected instead of being read as 1000%.
  `--help` shows defaults.

## Common tasks

**Add a trace format.** Write a function in `ingest/presets.py` that maps one record to a
`Session` through `convert_messages` or `convert_sharegpt`, and register it in `PRESETS`. Record
data-quality problems in `Session.flags` and never fix them silently. Map harness step limits to
`termination="step_limit"` (censored, not finished). Do not count scripted or orchestrator turns as
model calls. Add tests in `tests/ingest/` and a row in the README's "Trace formats" table.

**Add a candidate reply format.** Add the parser in `prompting/`, wire it into
`decode.parse_completion`, and add tests in `tests/prompting/`. Reasoning and analysis channels
are dropped from the action.

**Change a statistic, gate or verdict.** Read [reference/statistics.md](reference/statistics.md)
first. Add a simulation at the one-sided null showing the false-certification rate stays at or
below nominal (the `sim` pytest marker is registered for slow ones). Update the README section
that explains it, the card legend in `cli.py`, and the CHANGELOG. If a released bound turns out to
be wrong, say so plainly in the CHANGELOG, as v0.2.0 did.

**Change user-facing wording.** README, `--help`, the card legend and the website say the same
thing in the same words. Prefer short, literal sentences. Never claim more than the method
proves: say "agreement with the reference", not "correct", and "a range, not a promise" for
estimates.

## Releases and git

- One version lives in `pyproject.toml` (`__version__` comes from package metadata). Each release
  adds a dated `## [x.y.z] - YYYY-MM-DD` entry at the top of `CHANGELOG.md` (Semantic
  Versioning), written for users.
- Work happens on a branch and merges into `main` through a pull request. Commit subjects are
  either `vX.Y.Z: <what changed, as a semicolon-separated list>` for a release or
  `<area>: <what changed>; tests` (for example `certify: ...`, `prompting: ...`, `Website: ...`).
- Commit or push only when asked.
- `CITATION.cff`, `REFERENCES.md` (cite any new method) and `NOTICE` (third-party code) are
  maintained by hand.
