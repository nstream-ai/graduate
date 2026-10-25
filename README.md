<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/nstream-ai/graduate/main/docs/assets/graduate-dark.svg">
    <img alt="Graduate" src="https://raw.githubusercontent.com/nstream-ai/graduate/main/docs/assets/graduate-light.svg" width="256">
  </picture>
</p>

<p align="center"><b>Find out which of your agent's calls a cheaper model can take over, with proof.</b></p>

<p align="center">
  <a href="https://nstream-ai.github.io/graduate/">Website</a> ·
  <a href="https://graduate.readthedocs.io/">Documentation</a> ·
  <a href="https://graduate.readthedocs.io/en/latest/laptop.html">Run on a laptop</a> ·
  <a href="https://nstream-ai.github.io/graduate/grid/">Experiments</a>
</p>

Your agent asks its model many small questions per conversation: which tool to call, with which
arguments, or what to reply. Many are routine. Graduate takes your **logged traces** and a
**candidate model** (any model behind an OpenAI-compatible endpoint: a small open model on vLLM
or Ollama, a cheaper API, or a model you trained), asks the candidate every logged state, and
gives back a **report card**: the slices of traffic the candidate is certified to handle, with a
bound that holds for real agent traffic (calls linked within a conversation, many slices tested
at once, small samples).

A certified slice means the candidate **agrees with your reference model** there, within the
bound and for traffic like the traces you gave it. It does not mean the candidate is better, or
that its reply wording is right. Use certified slices as a cascade: the candidate answers each
turn, its answer is kept in a certified slice, and the reference answers otherwise.

## Install

Requires Python 3.12 or later (CI tests 3.12, 3.13 and 3.14). Install into a virtual
environment; on macOS use `python3`:

```bash
git clone https://github.com/nstream-ai/graduate && cd graduate
python3 -m venv .venv && source .venv/bin/activate
pip install .
```

`pip install graduate` installs the same package from PyPI once it is published there; the clone
also has the examples. Parquet input needs `pip install ".[parquet]"`.

## Quick start

See a report card in seconds, with a stand-in candidate and no model:

```bash
python examples/mock_candidate.py &
graduate certify examples/traces.jsonl --base-url http://localhost:8000/v1 --model mock --out report/
kill %1        # stop the stand-in
```

Open `report/card.md`. Then point `--base-url` and `--model` at a real candidate, and pass your
own traces:

```bash
graduate certify traces.jsonl --base-url http://localhost:11434/v1 --model qwen3:0.6b --out report/
```

`traces.jsonl` holds one conversation per line in the OpenAI chat format
(`{"id": ..., "messages": [...], "tools": [...]}`, with the reference model's `tool_calls` and the
tool results); `--preset` reads
[other formats](https://graduate.readthedocs.io/en/latest/guide.html#trace-formats). `--model`
must match the served name exactly. For a model on a laptop (memory, speed, Ollama), see
[Run on a laptop](https://graduate.readthedocs.io/en/latest/laptop.html). The example file has 3
conversations: enough to see the format, far too few to certify anything (a slice needs about 70
conversations).

Rerun the same command to resume: the candidate's answers are cached in `report/`, and a run
that stops leaves `card.md` saying so. Use a new `--out` for each candidate.

## Read the card

Each row is a **slice**: one kind of decision the candidate proposes (agent node, call type, tool,
position in the conversation).

| The card says | Meaning | What to do |
| --- | --- | --- |
| `GRADUATED` | Disagreement with the reference is below alpha (default 5%) at 95% confidence | The candidate can take this slice |
| `-` with "more conversations" | Not enough evidence yet; the card gives a range | Add that many conversations and rerun |
| `-` with "will not help" | Disagreement is significantly above alpha | Keep the slice on the reference |
| `-` with "too few to show it is above alpha" | A high rate on very few conversations | Add conversations to tell |
| `decision only (wording unchecked)` | A text reply: only the decision to reply is compared | Judge the wording separately, or pass `--certify-text-slices` |
| `held: reference noise` | Your reference disagrees with itself here at or above alpha | No candidate can certify it by agreement |

The card also gives the **certified share** of your calls, how often the candidate replied when
the reference acted (and the reverse), and one cause per disagreeing call. It ends with a legend
for every column.

## Commands

| Command | What it does |
| --- | --- |
| `graduate certify` | Certify a candidate against your reference, per slice, from logged traces |
| `graduate claim-b` | Certify by effect: score forked conversations where the candidate answered one turn |
| `graduate compare` | Compare whole sessions: paired task success of reference and candidate |

`graduate <command> --help` lists every option with its default. To compare two candidates, run
`certify` once per candidate on the same traces with the same options and compare the cards
slice by slice.

## What it does not check

- **Wording.** Two text replies agree when both are replies; their content is not compared.
- **Outcomes.** Agreement with the reference is a proxy. Where your environment can be reset to a
  logged state, `graduate claim-b` scores the stronger claim, by effect; running the forks needs
  your own environment runner.

If you grade replies with an LLM judge, use one from a different model family from both the
candidate and the reference. Traces produced by a commercial API are subject to that provider's
terms; checking that you may use them this way is your responsibility.

## Documentation

- [Guide](https://graduate.readthedocs.io/en/latest/guide.html): what the card reports, trace
  formats, `claim-b`, `compare`, and how the certificate works
- [Run on a laptop](https://graduate.readthedocs.io/en/latest/laptop.html): Ollama and llama.cpp,
  memory, speed, thinking models
- [Changelog](https://github.com/nstream-ai/graduate/blob/main/CHANGELOG.md) and [references](https://github.com/nstream-ai/graduate/blob/main/REFERENCES.md)

## Tell us what went wrong

If the card says something that does not match what you see in production (a slice certified that
should not be, a format we cannot read, a failure the card misses), open an issue with the
"Report from my traces" template. Share counts and `card.md` (example values are redacted there),
not `card.json`, which keeps raw values from your traces.

## Citation

```bibtex
@software{graduate2026,
  title  = {Graduate: statistical certificates for handing agent traffic to a cheaper model},
  author = {{The Graduate Authors}},
  year   = {2026},
  url    = {https://github.com/nstream-ai/graduate},
  note   = {Version 0.2.2}
}
```

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
pytest
```

The core needs only NumPy, SciPy and Pydantic. `src/graduate` is the package; `index.html` and
`grid/` are the website (GitHub Pages from `main`); `docs/` is the documentation site on Read the
Docs (`pip install -r docs/requirements.txt . && sphinx-build -W docs docs/_build`).

## License

Apache-2.0; see [LICENSE](https://github.com/nstream-ai/graduate/blob/main/LICENSE). Third-party notices are in [NOTICE](https://github.com/nstream-ai/graduate/blob/main/NOTICE).
