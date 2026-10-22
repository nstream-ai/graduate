# Graduate

**Find out which of your agent's calls a cheaper model can take over — with proof.**

Your agent asks its model many small questions per conversation: which tool to call, with which
arguments, or what to reply. Many are routine. Graduate takes your **logged production traces**
and a **candidate model** (any model behind an OpenAI-compatible endpoint: a small open model on
vLLM or Ollama, a cheaper API, or a model you trained), asks the candidate every logged state,
and gives back a **report card**: the slices of traffic the candidate is certified to handle, with
a statistical bound that holds for real agent traffic (calls linked within a conversation, many
slices tested at once, small samples). Plug the certified slices into the router you already have.

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
  in the session), which is what a router can see before deciding; a slice is certified when its
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

`card.md` ends with a legend for the table. A slice that is not certified for lack of evidence
says roughly how many more conversations it would need.

## Terms

- **Call**: every model turn in a conversation, a tool call or a text reply.
- **Slice**: one kind of decision the candidate proposes, which a router can see before it
  routes: agent node, call type (tool or text), tool name, and position in the conversation.
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

## What v0.1 certifies

**Agreement with your reference model, per slice**: the candidate's action matches the
reference's on at least 1 − alpha of the slice's calls, with 1 − delta confidence. And,
optionally, **the reference's own noise** on the same states.

If your reference disagrees with itself on a slice more often than your gate (alpha), **no
candidate can certify that slice by agreement**. That is a property of your reference, not of the
candidate. Tool slices of real agents often look like this: we measured 5–17% self-disagreement
for one widely used API model on τ²-bench tool calls, against a 5% gate.

**Not in v0.1:**
- outcome-based (fork) certification, which compares task outcomes after the candidate answers
  one turn and the reference finishes;
- the never-route list.

**Default protocol:** cap 1 (one call per conversation and slice), the anytime-valid betting test
of H0: disagreement ≥ alpha, Benjamini–Hochberg across slices, alpha = 5%, delta = 5%. As a rule
of thumb at these defaults, a single slice needs about 70 conversations if the candidate never
disagrees with the reference, about 120 at a true 1% disagreement rate and about 270 at 2%, and
more when many slices are tested at once.

## How the certificate works

- **Exact bounds, not normal approximations.** Clopper–Pearson bounds and exact one-sided tests;
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
  reference finish, and compare task outcomes. Our τ²-bench experiments use this (see the
  website); it is not in the CLI yet.

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
