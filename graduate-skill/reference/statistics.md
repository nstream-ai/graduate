# Graduate statistics: what each claim is and what must stay true

Read this before changing anything in `certify/` or a verdict string in `cli.py`.

## Claim A: agreement per slice (`graduate certify`)

- **Null hypothesis per slice and gate:** H0: rate ≥ alpha. Certified means H0 is rejected at
  delta after multiplicity correction.
- **Population:** the first `cap` calls per session in arrival order (default cap 1), so one long
  conversation cannot certify a slice.
- **Test:** the anytime-valid betting test (a mixture of bets, `MIXTURE` in `card.py`) from
  `stats.sequential`. It stays valid under continuous monitoring on live traffic.
- **Gates:** `disagree` (args-level agreement) and `invalid` (schema validity). A slice passes
  only if every gate passes. This is an intersection-union test (Berger 1982): the slice's p-value
  is the largest gate p-value, with no correction across gates.
- **Across slices:** Benjamini–Hochberg by default (Holm, Bonferroni and BY are available in
  `stats.multiplicity`).
- **After the test:** text slices are set to not certified unless `--certify-text-slices` is
  passed. Slices where the reference re-asked disagrees with itself at or above alpha are "held".
  If the reference breaks its tool contract in more than 20% of conversations, every slice is set
  to not certified.
- **Verdict ladder** (`cli.slice_verdicts`, in order): held, then certified, then decision only,
  then "significantly above alpha" (the mirrored test of H0: rate ≤ alpha rejects, so more data
  will not help), then rate ≥ alpha but not significant ("cannot be decided", including too few
  conversations), then the planner's range, then not certified.
- **Rule of thumb at the defaults:** about 70 conversations for one slice with zero
  disagreements, about 120 at a 1% true rate, about 270 at 2%, and more with many slices.

## Claim B by effect (`graduate claim-b`)

- Rows: `{"state", "task_id", "slice", "branch": "T1"|"T2"|"S", "success", "termination",
  optional "proposed_slice"}`. Rows with an excluded termination (`infrastructure_error`,
  `replay_error`, `error`, `budget_stop`) or `success: null` are left out, never scored.
- **Signed net harm**, task-weighted, is the per-task P(T1 pass, S fail) minus P(T1 fail, S pass).
- **Decision bound:** a finite-sample task-level betting bound (`stats.taskbounds.finite_upper`)
  on magnitudes, with tasks as the unit. In a 120-cell null simulation its worst
  false-certification rate was 1.1% at a nominal 5%.
- **Information only:** the task-cluster t bound (it falsely certified up to 46% on forks, so it
  is never used there) and the cluster bootstrap.
- **Multiplicity:** e-BH on finite-sample e-values, which is valid under the dependence that
  shared tasks create.
- **Certified iff** e-BH rejects h ≥ delta_turn, the finite bound ≤ delta_turn, n ≥ 3 /
  delta_turn, the slice is non-vacuous (P(T1 pass) ≥ 2·delta_turn), and the reference agrees with
  itself (h_T, T2 vs T1, within 1.645 clustered SE of 0).
- delta_turn = session margin / cap. The session bound is cap × the per-turn bound (union bound).
  `cap_max_informative` comes from certified slices only.
- **Never route:** harm is likely (above 0) and its bound exceeds half the reference's own success
  on the slice.
- Slices with fewer than 30 states are listed as untested and kept out of the family. Positions
  are pooled unless `--by-position` is passed.

## Session comparison (`graduate compare`)

- Paired per-task regret d = reference success rate − candidate success rate. Tasks are the unit
  and trials are averaged within a task. Tasks found in only one file are counted, not paired.
- **Verdict ladder** (`certify/session.py`, in order):
  1. "worse than the margin": the finite-sample lower bound is above the margin;
  2. "cannot be decided": fewer than 3 / margin tasks;
  3. "vacuous": the reference succeeds on less than 2 × margin;
  4. "non-inferior": the finite upper bound ≤ margin;
  5. "supported (asymptotic only)": only the t bound ≤ margin, and only with ≥ 100 tasks or
     ≥ 2 trials per task;
  6. "cannot be decided with this evidence": the detectable regret is above the margin;
  7. otherwise, "non-inferiority not shown (not proven worse)".
- Detectable regret = max(3 / n, (1.645 + 0.84) · SD / √n), which is one-sided 5% at 80% power.
- `success` must be `true`/`false` or `0`/`1`. Anything else stops with an error.

## Invariants to preserve

1. A valid bound must not return the point estimate, or 0, when there are no discordant
   outcomes. The v0.2.0 "conditional exact" bound did this and falsely certified 14–20% of slices
   at 30–60 states before it was replaced.
2. New decision bounds come with a null simulation across within-task correlation from 0 to 1 and
   sample sizes down to about 30. Report the worst false-certification rate against nominal.
3. Resampling (planner, bootstrap) resamples whole conversations or tasks, never single calls.
4. The multiplicity family is fixed before the e-values or p-values are read.
5. Anything printed "for information" is labelled as such and never feeds `certified`.
6. Planner estimates are "a range, not a promise". The planner history must not let the
   estimate improve without the rate improving.

## Where the methods come from

`REFERENCES.md`: Clopper–Pearson; Benjamini–Hochberg and Benjamini–Yekutieli; Holm; Berger
(1982) for IUT; Waudby-Smith & Ramdas (2024) for betting; Ramdas et al. (2023) for anytime-valid
inference; Bowyer et al. (2025) on why the CLT fails at small n; Miller (2024).
