# Changelog

All notable changes to this project are listed here. Versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-10-22

Fixes from a new-user walkthrough.

- The answer cache in `--out` is tied to the candidate (`--model`, `--base-url`, traces): reusing
  a folder for another candidate is refused instead of silently scoring the old answers. `--fresh`
  asks again.
- Failed requests are no longer cached; a rerun retries them. Failures are printed with the first
  error, counted in `card.md`, and the run exits with an error when every request fails.
- No API key is sent unless `--api-key-env` is given (it used to default to `OPENAI_API_KEY`);
  the reference endpoint has its own `--reference-api-key-env`.
- Plain error messages for a missing file, invalid JSON (with the line number), a file that does
  not match `--preset`, and `--alpha`/`--delta` outside (0, 1). `--help` shows defaults;
  `graduate --version`.
- The report card has a legend, shows the binding gate only on slices that did not pass, and says
  roughly how many more conversations a slice needs when evidence is short.
- `examples/mock_candidate.py` and CPU recipes (Ollama, llama.cpp) to try Graduate without a GPU;
  README sections on trace formats, terms and API keys; website corrected to what v0.1 does.

## [0.1.0] - 2026-10-21

First public release.

- `graduate certify` CLI: ask a candidate model every logged state of your traces and get a report
  card of the traffic slices it is certified to handle.
- Certifier statistics: exact binomial bounds, multiplicity control (Bonferroni, Holm,
  Benjamini–Hochberg/Yekutieli), anytime-valid betting test, power planning.
- Trace ingest with presets for OpenAI chat, Hermes, Toucan, Nemotron, SWE-rebench and τ²-bench.
- Session-level time-ordered splits, deduplication and slicing.

[0.1.1]: https://github.com/nstream-ai/graduate/releases/tag/v0.1.1
[0.1.0]: https://github.com/nstream-ai/graduate/releases/tag/v0.1.0
