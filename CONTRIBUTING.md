# Contributing

The most useful contribution right now is a **report from your own traces**: run
`graduate certify` on your agent's logs with a candidate you are considering, and open an issue
with the "Report from my traces" template when the card disagrees with what you see in
production. New trace formats (presets) and candidate failure modes the card misses are especially
welcome.

Code changes: fork, branch, `pip install -e ".[dev]"`, keep `pytest` and `ruff check` green, and
open a pull request describing the change and how you tested it. CI runs the same checks on every
pull request.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report security problems privately, as
described in [SECURITY.md](SECURITY.md), not in a public issue.
