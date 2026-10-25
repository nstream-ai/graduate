# Graduate skill for Claude Code

A [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills) that teaches Claude how
this repo works: its layout, its statistical invariants, its code and wording conventions, and
the release workflow.

- `SKILL.md` is loaded when the skill is used.
- `reference/architecture.md` and `reference/statistics.md` are read only when a task needs them.

## Install

**Inside a clone of this repo:** nothing to do. `.claude/skills/graduate` links here, so Claude
Code picks the skill up automatically.

**Anywhere else (for every project):**

```bash
cp -r graduate-skill ~/.claude/skills/graduate
```

**Sharing as a file:** `zip -r graduate-skill.zip graduate-skill`. The recipient unzips it into
`~/.claude/skills/graduate` (or `<their-repo>/.claude/skills/graduate`).

Check that it loaded by asking Claude Code "what skills are available?"; `graduate` should be
listed.
