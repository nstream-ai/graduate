# Changelog

All notable changes to this project are listed here. Versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-23

Certification by effect and session-level comparison: the methods the paper uses.

**Corrected before release (2026-10-23).** The bound first written for `claim-b` and `compare`,
the stricter of a task-clustered bootstrap and an "exact" bound conditional on the discordant
states, was not valid: the conditional bound treats the discordance rate as known, returns the
point estimate when the candidate is never better, and returns 0 when no task differs. At a
one-sided null it falsely certified 14–20% of slices at 30–60 states. It is replaced:

- `graduate certify`: text-reply slices read "decision only (wording unchecked)" and stay out of
  the certified share (their agreement is on the decision, not the wording);
  `--certify-text-slices` counts them. Slices where the reference disagrees with itself at or
  above alpha (`--reference-base-url`) stay on the card as "held: reference noise" instead of
  disappearing.
- `claim-b`: `cap_max_informative` comes from certified slices only (an uncertified slice is not
  routed); with no certified slice it is "not defined".
- `claim-b` and `compare` certify with a **finite-sample, task-level betting bound** on the
  per-task mean difference (magnitudes, not signs; `graduate.certify.stats.taskbounds`). In a
  120-cell simulation at one-sided nulls, with within-task correlation from 0 to 1, its worst
  false-certification rate is 1.1% at a nominal 5%.
- A task-cluster t bound and the bootstrap are printed for information. On forks the t bound
  falsely certifies up to 46%, so fork cards never use it. `compare` labels a result
  "supported (asymptotic only)" only with at least 100 tasks or two trials per task.
- `claim-b` controls multiplicity with **e-BH** on the finite-sample e-values (valid under the
  dependence that shared tasks create); the family is fixed before the e-values are read, and
  ineligible slices enter with e = 1.
- `compare` reads "cannot be decided" first when there are fewer than 3 / margin tasks.
- `certify` says "reference tool contract: not checked" when the traces carry no tool list,
  instead of treating the reference as admissible without comment.
- New extra: `pip install "graduate[completions]"` (transformers), which the completions path
  asks for.
- README: routing is described as a cascade (the candidate answers every turn; its answer is
  kept on certified slices), and `claim-b` is described as scoring forks, which need an
  environment runner Graduate does not ship.

The items below were written before the correction; where they mention "the stricter of a
task-clustered bootstrap and an exact bound" or "the binding one", read the bounds above.

- `graduate claim-b`: certify slices by outcome from forked conversations (the candidate answers
  one turn, the reference finishes). Signed net harm with the stricter of a task-clustered
  bootstrap and an exact bound, multiplicity control across slices, the reference's noise floor,
  "never route" and "vacuous" verdicts, and the smallest harm the evidence could detect. Every card
  states its composition (union bound over the routed turns per session) and its requirement (an
  environment that can be reset to a logged state).
- `graduate compare`: paired session-level regret of a candidate or hybrid against the reference
  on the same tasks, with its bound and the smallest detectable regret; reads JSONL or τ²-bench
  results.
- `graduate certify`: a `detect` column, the smallest disagreement rate the slice's conversations
  can rule out. A slice whose disagreement is significantly above alpha (its one-sided lower
  bound reaches alpha) is "not certified" and says more conversations will not help; every other
  slice not yet certified "cannot be decided with this evidence", with the planner's range when
  its observed rate is below alpha. card.md has one per-slice table; the second stays in
  card.json. Format-only causes that `--normalise-args` forgave are marked as such, the card says
  whether the normaliser was applied (its rule is in the legend), and tool names print plainly.
- `claim-b` cards report `cap_max_informative`: the largest router cap for which the per-turn
  bound still supports the session margin under the union bound.
- `graduate certify --prompt-mode completions`: the candidate's prompt is rendered from its own
  chat template (thinking off), fitted to `--prompt-tokens`, sent to `/completions`, and parsed
  with `graduate.prompting`. This is the path the paper's exams use. `chat` (the server's
  template and tool parser) stays the default.
- `graduate compare`: session-level non-vacuity. If the reference succeeds on less than twice
  the margin, the verdict reads "vacuous". Both bounds are printed, and the binding one is
  named.
- The reference contract line: a reference that calls tools outside its own tool list in more
  than 20% of sessions on a domain is not an admissible reference there, and nothing is
  certified against it.
- `claim-b`: slices with fewer than 30 states stay out of the multiple-comparison family.
  Zero-discordance slices get the exact zero-event bound. `states_needed` is sized at 80% power.
- **Import path renamed:** `graduate.train` is now `graduate.prompting` (`to_chat`,
  `tools_spec`, `parse_completion`, the Gemma parser). Code that imported
  `graduate.train.decode` or `graduate.train.data` from v0.1 must change its imports. No trainer
  was ever included; the old name only held prompting and parsing.

## [0.1.3] - 2026-10-22

- README: share `card.md` (example values redacted) when reporting, not `card.json`, which keeps
  raw values from your traces.
- Website: two source comments no longer name internal files.

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

[0.2.0]: https://github.com/nstream-ai/graduate/releases/tag/v0.2.0
[0.1.3]: https://github.com/nstream-ai/graduate/releases/tag/v0.1.3
[0.1.2]: https://github.com/nstream-ai/graduate/releases/tag/v0.1.2
[0.1.1]: https://github.com/nstream-ai/graduate/releases/tag/v0.1.1
[0.1.0]: https://github.com/nstream-ai/graduate/releases/tag/v0.1.0
