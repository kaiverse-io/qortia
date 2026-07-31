---
kind: agent-context
status: active
last_reviewed: <!-- update when reviewing -->
---

# Qortia — Agent Context

> **Canonical agent context** (AGENTS.md open standard — Linux Foundation AAIF).
> `CLAUDE.md` is a symlink to this file. Update here; the symlink follows.

Portable memory layer for AI agents

## Project structure

```
src/qortia/   — application source (see import contracts in .importlinter)
tests/                           — pytest suite
prompts/                         — versioned prompt artifacts (never inline — see prompts/README.md)
evals/                           — eval harness slot (wired-but-waiting; see evals/README.md)
docs/tutorials/                  — learning-oriented guides
docs/how-to/                     — task-oriented recipes
docs/reference/                  — API / config reference
docs/explanation/                — concepts and theory
docs/decisions/adrs/             — MADR decision records
.opengrep/rules/                 — custom semantic lint rules (self-weakening + prompts-as-code + project)
.agents/skills/                  — reusable agent skills (SKILL.md open standard; .claude/skills/ symlinks here)
.agents/memory/                  — agent memory, git-tracked (~/.claude/…/memory symlinks here; post-create recreates it)
```

## Hard rules

<!-- ── PROJECT INVARIANTS SLOT ────────────────────────────────────────────── -->
<!-- Add project-specific invariants here. Examples:                           -->
<!--   "Nothing in core/ may import boto3 or any cloud SDK."                  -->
<!--   "All DB writes go through the repository port, never raw psycopg3."    -->
<!-- ─────────────────────────────────────────────────────────────────────── -->

## How to work here

- `just ci` before every push — runs lint, type-check, import boundaries, tests, docs.
- `just fmt` to auto-format (ruff format + ruff check --fix).
- `just sync` after changing `pyproject.toml`.
- Conventional commits enforced: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`.
- All significant decisions → MADR ADR in `docs/decisions/adrs/` first.
- Coverage ratchet: `fail_under` in `[tool.coverage.report]` — only ever increase it.

## Agent-memory convention

After a significant multi-step task, save reusable learnings to:
`~/.claude/projects/<project-slug>/memory/`

One file per insight; `MEMORY.md` index (one line per entry). This prevents knowledge
loss across sessions and feeds the coaching loop (Factor 2/3 of 12-factor-agents).

<!-- cockpit-section:start (kept in sync with chassis's own root AGENTS.md — see justfile ci-lint) -->
## AI-usage cockpit & coaching

- `just metrics` (codeburn — cost/burn by project, model, task; one-shot rate) and
  `just monitor` (abtop — live context %, tokens, rate limits) are installed by
  `.devcontainer/post-create.sh`. Both read local session data; no OTEL required.
- **AI Engineer Coach** (VS Code dashboard, [microsoft/ai-engineering-coach](https://github.com/microsoft/ai-engineering-coach))
  is **opt-in** (`install_ai_coach`, default off). When enabled, `.devcontainer/post-create.sh`
  builds it from source (pinned to a commit) — 45 anti-pattern rules, practice scores, skill
  mining; open via Cmd/Ctrl+Shift+P → "AI Engineer Coach: Open Dashboard". Off by default
  because it builds upstream source unattended and its **Claude Code session-log support is
  unconfirmed** (the project documents GitHub Copilot harnesses) — if you enable it, verify
  what it actually surfaces for this project.
- **graphify** (`/graphify`, [Graphify-Labs/graphify](https://github.com/Graphify-Labs/graphify))
  — turns the repo into a queryable knowledge graph (`graphify-out/graph.json`/`GRAPH_REPORT.md`),
  installed via `uv tool install graphifyy` by `.devcontainer/post-create.sh`. Check
  `graphify-out/GRAPH_REPORT.md`'s God Nodes/communities before exploring unfamiliar code —
  fewer tokens spent re-discovering structure the graph already answers.
- **ctx** ([ctxrs/ctx](https://github.com/ctxrs/ctx)) — indexes full local session transcripts
  (not just memory files) for `ctx search "…"`, so `/dev-coach` can find corrections the agent
  actually got, not just what a past session chose to write down. Installed by
  `.devcontainer/post-create.sh` via its own installer (prebuilt binary, no Rust toolchain).
- OTEL telemetry is on and exports to console locally (`.claude/settings.json`) — the
  spine for a future OTLP collector + team dashboard (bucket D, activates at users > 1).
- `/dev-coach` — anti-pattern detection + AGENTS.md auditor, specifically for Claude Code
  session/memory data (complements AI Engineer Coach, doesn't replace it). Also draws on `ctx`
  (full-transcript correction search) when present. Run
  periodically to close the loop between "the user corrected something" and "the rule is durable
  in AGENTS.md." See `.agents/skills/dev-coach/SKILL.md`.
- This cockpit is a chassis-level standard, not project-specific — every project stamped from
  this chassis gets it via `post-create.sh`. See
  [chassis's docs/explanation/ai-usage-cockpit.md](https://github.com/km2411/chassis/blob/main/docs/explanation/ai-usage-cockpit.md)
  for the full writeup and [ADR-002](https://github.com/km2411/chassis/blob/main/docs/decisions/adrs/adr-002-ai-usage-cockpit.md).

<!-- cockpit-section:end -->

## Forbidden patterns — never do these

- `git commit --no-verify` — fix the violation, never skip the gate.
- Inline prompt literals (200+ chars) — move to `prompts/` and load at runtime.
- Bare `# noqa` or `# type: ignore` without a specific code and reason.
- Hardcode secrets — use `.env` (gitignored) and `.env.example` as the template.
- Lower the `fail_under` coverage floor.
- `continue-on-error: true` in CI to silence a failing step.
- `pre-commit uninstall` — the hooks are the guardrails, not obstacles.
- Commit to `main` without CI green.
