# Changelog

All notable changes to this project are listed here. Versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.2] - 2026-10-22

Two report-card fixes from a second new-user walkthrough.

- **Why a slice disagrees.** Each disagreeing call gets one cause, first match in this order:
  replied instead of acting, acted instead of replying, wrong tool (including a different number of
  calls), missing or extra argument, format-only argument, wrong argument value. `card.json` has
  the shares per slice and the first disagreeing call; `card.md` has a "Why slices disagree" table
  with example values redacted (digits as 0, emails as x@x.x), so "#W0000" vs "W0000" stays
  visible. Format-only means equal after the argument normaliser (version 1, printed on the card).
  Such differences are reported, not forgiven; `--normalise-args` counts them as agreement, and
  the card says so.
- **How many more conversations.** The estimate now uses the card's own test: simulated futures
  that append whole conversations resampled from yours, run through the same betting test. It is
  a range: at least N at the uncorrected level, up to N at the level corrected for the number of
  slices, each where 80% of simulated futures pass. Near the margin the card adds "estimates near
  the margin are unreliable; this is a range, not a promise". The old estimate ignored the
  correction and the cost of the always-valid test, and said "~1 more" when about 11 were needed.
  Reruns into the same `--out` never move the target closer unless the rate improved
  (`planner.json`). Certified slices record where they certified and the previous prediction.

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

[0.1.2]: https://github.com/nstream-ai/graduate/releases/tag/v0.1.2
[0.1.1]: https://github.com/nstream-ai/graduate/releases/tag/v0.1.1
[0.1.0]: https://github.com/nstream-ai/graduate/releases/tag/v0.1.0
