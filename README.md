# Graduate

**Find out which of your agent's calls a cheaper model can take over — with proof.**

Your agent asks its model many small questions per conversation: which tool to call, with which
arguments, or what to reply. Many are routine. Graduate takes your **logged production traces**
and a **candidate model** (any model behind an OpenAI-compatible endpoint: a small open model on
vLLM or Ollama, a cheaper API, or a model you trained), asks the candidate every logged state,
and gives back a **report card**: the slices of traffic the candidate is certified to handle, with
a statistical bound that holds for real agent traffic (calls linked within a conversation, many
slices tested at once, small samples). Use them as a cascade: the candidate answers each turn, its answer is kept when it falls in a certified slice, and the reference answers otherwise. The cascade pays for a candidate call on every turn and adds latency on the turns it escalates.

Website: https://nstream-ai.github.io/graduate/ (experiments overview at `/grid/`)

## Quick start

Requires Python 3.12 or later.

```bash
pip install "git+https://github.com/nstream-ai/graduate.git"
```

See a report card in seconds, with a stand-in candidate and no model (from a clone of this repo):

```bash
python examples/mock_candidate.py &
graduate certify examples/traces.jsonl --base-url http://localhost:8000/v1 --model mock --out report/
```

Then point it at a real candidate behind any OpenAI-compatible endpoint. A small model on your
CPU works: with [Ollama](https://ollama.com), `ollama pull <model>` and use
`--base-url http://localhost:11434/v1 --model <model>`. `examples/README.md` also covers
llama.cpp. On your own traces:

```bash
graduate certify traces.jsonl --preset openai --base-url http://localhost:11434/v1 --model <model> --out report/
```

`traces.jsonl` holds one conversation per line in the OpenAI chat format
(`{"id": ..., "messages": [...], "tools": [...]}`, with the reference model's `tool_calls` and
the tool results). Other formats: see [Trace formats](#trace-formats). Parquet input needs
`pip install "graduate[parquet] @ git+https://github.com/nstream-ai/graduate.git"`.

**API keys.** No key is sent unless you pass `--api-key-env NAME` (the environment variable
holding the candidate endpoint's key; `--reference-api-key-env` for the reference). A local
server (Ollama, llama.cpp, vLLM) needs none. That is how we run candidates in our own τ²-bench
experiments: on a local server, with `--preset tau2` traces.

You get, in `report/`:

- **certified slices**: each slice is the candidate's proposed action (call type, tool, position
  in the session), which is known once the candidate has answered (the routing is a cascade); a slice is certified when its
  disagreement with your reference model is provably below `--alpha` (default 5%) at 95%
  confidence;
- **certified share**: the fraction of your calls inside certified slices;
- **decision split**: how often the candidate replied when the reference acted, and acted when it
  replied, the most common way a substitute fails;
- optionally, the **reference's own noise** (`--reference-base-url`, `--reference-model`): a
  slice where your reference disagrees with itself at or above alpha cannot be certified by
  agreement by anyone, and is reported as such;
- `proposals.jsonl`: a cache of the candidate's answers. A rerun into the same folder resumes and
  retries only failed requests. It is tied to the `--model` and `--base-url` it was made with, so
  use a new `--out` for each candidate (or `--fresh` to ask again).
- `planner.json`: the evidence estimate's memory across reruns into the same folder, so the
  "more conversations" target does not move closer unless the disagreement rate improved.

`card.md` ends with a legend for the table. A slice that is not certified for lack of evidence
gives a range of how many more conversations it would need; a slice whose disagreement is
significantly above alpha (its lower bound reaches alpha) says that more conversations will not
help. "Why slices disagree" gives one cause per
disagreeing call and an example with its values redacted.

## Terms

- **Call**: every model turn in a conversation, a tool call or a text reply.
- **Slice**: one kind of decision the candidate proposes, known once it has answered: agent node, call type (tool or text), tool name, and position in the conversation.
- **alpha**: the disagreement rate you allow on a slice (default 0.05). **delta**: 1 − confidence
  (default 0.05, so 95% confidence).
- **cap**: calls counted per conversation and slice (default 1), so one long conversation cannot
  certify a slice on its own.
- **Reference noise**: how often your reference model disagrees with itself on the same state.

## Trace formats

| `--preset` | Reads |
|---|---|
| `openai` (default) | one conversation per line: `messages` (OpenAI chat format, `tool_calls` or legacy `function_call`), optional `tools` |
| `hermes` | ShareGPT-style `conversations` with `<tool_call>` blocks (Hermes function calling) |
| `toucan` | rows of the Toucan tool-agent dataset |
| `nemotron` | rows of the Nemotron-Agentic dataset |
| `nemotron-swe` | Nemotron software-engineering trajectories (first 100 steps) |
| `swe-rebench` | SWE-rebench OpenHands trajectories |
| `tau2` | τ²-bench simulation results; add the agent's system prompt and tool definitions to each simulation (`system`, `tools`) when you export them |

## What it certifies

**Agreement with your reference model, per slice**: the candidate's action matches the
reference's on at least 1 − alpha of the slice's calls, with 1 − delta confidence. And,
optionally, **the reference's own noise** on the same states.

If your reference disagrees with itself on a slice more often than your gate (alpha), **no
candidate can certify that slice by agreement**. That is a property of your reference, not of the
candidate. Tool slices of real agents often look like this: we measured 5–17% self-disagreement
for one widely used API model on τ²-bench tool calls, against a 5% gate.

**Default protocol:** cap 1 (one call per conversation and slice), the anytime-valid betting test
of H0: disagreement ≥ alpha, Benjamini–Hochberg across slices, alpha = 5%, delta = 5%. As a rule
of thumb at these defaults, a single slice needs about 70 conversations if the candidate never
disagrees with the reference, about 120 at a true 1% disagreement rate and about 270 at 2%, and
more when many slices are tested at once.

## Certify by effect: `graduate claim-b`

Agreement with the reference is a proxy. When your environment can be **reset to a logged state**
(a simulator, a sandbox, or a replay of your tools), you can test the outcome instead. Fork each
sampled conversation at a turn, and run three branches from that exact state:

- **T1, T2:** the reference answers the turn and finishes the conversation (twice, to measure its
  own noise);
- **S:** the candidate answers that one turn, and the reference finishes.

Write one JSON line per branch:
`{"state", "task_id", "slice", "branch": "T1"|"T2"|"S", "success": true|false, "termination"}`
(plus `proposed_slice`, the candidate's own action, when it differs from the logged one).
Graduate scores forks; it does not run them. Producing them needs a runner that resets your
environment to the logged state. Then:

    graduate claim-b forks.jsonl --session-margin 0.10 --cap 1 --out claim-b/

For each slice of the candidate's proposals, the card reports:
- the **signed net harm**: sessions the candidate's turn made worse minus sessions it made
  better, per routed turn, with a one-sided finite-sample upper bound with tasks as the unit (a betting bound; a
  task-cluster t bound and a bootstrap bound are printed for information only);
- **never route**: the harm is likely (above 0) and its upper bound exceeds half the reference's
  own success on the slice, so the router always sends it to the reference. "Not certified"
  means the bound is above delta without that evidence of harm;
- **vacuous**: the reference itself rarely succeeds there, so "no harm" means nothing;
- the **smallest harm this evidence could detect**: "cannot be decided with this evidence" is
  not a failure;
- the **reference's noise floor**: how often T1 and T2 end differently.

Slices with fewer than 30 states are listed as untested and kept out of the multiple-comparison
family. Slices are named domain/kind/tool: positions in the conversation are pooled unless
`--by-position`. Each input row needs `state`, `branch` (`T1`, `T2` or `S`), `task_id`, `slice`
and `success`; a row missing a field, or input with no complete fork, stops with an error.

**Two labels on every card:**
- *Composition:* the per-session bound is the union bound over at most `cap` routed turns
  (session harm ≤ cap × per-turn bound). The card also reports `cap_max_informative`, the
  largest cap for which your per-turn bound still supports the session margin: for example, a
  per-turn bound of 4 points and a 10-point margin give 2. A dependence-aware composition is
  research, not used here.
- *Requires:* an environment that can be reset to a logged state. Without one, use
  `graduate certify` (agreement) and `graduate compare` (whole sessions).

**Text replies and noisy slices.** A text-reply slice agrees on the decision to reply, not on
the reply's wording, so it reads "decision only (wording unchecked)" and stays out of the
certified share; `--certify-text-slices` counts it. With `--reference-base-url` the reference is
re-asked on a sample of states, and a slice where it disagrees with itself at or above alpha reads
"held: reference noise". `--prompt-mode completions` renders the candidate's prompt from its own
chat template and parses the reply with `graduate.prompting` (Qwen, Gemma and the harmony format
of gpt-oss models); `chat`, the default, uses the server's template and tool parser.

## Compare whole sessions: `graduate compare`

Run the reference and the candidate (or a hybrid router) on the same tasks, then:

    graduate compare reference.jsonl candidate.jsonl --margin 0.10 --out compare/

Input: `{"task_id", "success"}` lines, or τ²-bench `results.json` files (`--outcome db` for the
database end state, `reward` for the full reward; infrastructure errors are left out). The report
gives the paired regret (reference success − candidate success), with a one-sided 95% upper bound
from a finite-sample betting bound with tasks as the unit (t and bootstrap bounds printed for
information), a lower bound from the same test, and the smallest regret the evidence could
detect. Verdicts: "non-inferior"; "worse than the margin" when even the lower bound is above it;
"non-inferiority not shown (not proven worse)"; "cannot be decided with this evidence" when the
task count is too small for the margin (fewer than 3 / margin tasks, and a few dozen tasks rarely
resolve 10 points); "vacuous" when the reference itself succeeds on less than twice the margin.
`success` must be `true`/`false` or `0`/`1`; other values stop with an error. Tasks in only one
file are left out and counted. `--margin` is a fraction (0.10 = 10 points).

## How the certificate works

- **Finite-sample bounds, not normal approximations.** Clopper–Pearson bounds per call, and
  betting bounds for fork and session claims;
  the central-limit approximations common in LLM evaluation under-cover at the sample sizes real
  slices have.
- **Many slices, one error budget.** Benjamini–Hochberg across slices (Holm, Bonferroni and
  Benjamini–Yekutieli are available), and an intersection–union test across gates (disagreement,
  invalid calls).
- **Conversations are not independent calls.** Calls are capped per conversation and slice, so one
  long session cannot certify a slice on its own.
- **Anytime-valid monitoring.** A betting-based sequential test lets you keep checking a certified
  slice on live traffic without inflating the error rate.

## What it does not check (yet)

- **Wording.** Two text replies agree when both are replies; their content is not compared. Judge
  replies separately, or measure outcomes.
- **Outcomes.** Agreement with the reference is a proxy. Where an environment can be replayed, the
  stronger claim is *by effect*: fork the conversation, let the candidate answer one turn, let the
  reference finish, and compare task outcomes. `graduate claim-b` scores such forks (below); running them needs an environment runner.

## Using a judge

If you grade replies with an LLM judge, use a judge from a **different model family** from both
the candidate and the reference. Models recognise and favour their own outputs, so grading with
the model you are replacing (or its sibling) biases the certificate towards it.

## Your traces and your provider's terms

Traces produced by a commercial API are subject to that provider's terms. Checking that you may
use them this way is your responsibility.

## Tell us what went wrong

The point of this release is to run it on **your** traces. If the card says something that does
not match what you see in production (a slice certified that should not be, a format we cannot
read, a candidate failure the card misses), please open an issue with the "Report from my traces"
template. Share counts and `card.md` (example values are redacted there), not `card.json`,
which keeps raw values from your traces, and not your customers' data.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
pytest
```

The core needs only NumPy, SciPy and Pydantic. Licence: Apache-2.0 (third-party notices in `NOTICE`). References: `REFERENCES.md`.

## This repo

`index.html` and `grid/` (the experiments overview) are the website, published with GitHub Pages from the `main` branch;
`src/graduate` is the package.
