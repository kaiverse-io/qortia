---
kind: agent-context
status: active
last_reviewed: <!-- update when reviewing -->
---

# Qortia — Agent Context

> **Canonical agent context** (AGENTS.md open standard — Linux Foundation AAIF).
> `CLAUDE.md` is a symlink to this file. Update here; the symlink follows.

Portable memory layer for AI agents (standalone FastAPI + Postgres/pgvector).

> **Doc authority:** [`ARCHITECTURE.md`](ARCHITECTURE.md) and
> [`docs/index.md`](docs/index.md) describe current behavior.
> Host-platform extraction dumps live under
> [`docs/archive/extraction/`](docs/archive/extraction/) — historical only.
> See [`docs/decisions/adrs/adr-001-standalone-extraction.md`](docs/decisions/adrs/adr-001-standalone-extraction.md).

## Project structure

```
ARCHITECTURE.md                  — component docs; just ci-arch enforces coverage (see below)
src/qortia/                      — application source (see import contracts in .importlinter)
tests/                           — pytest suite
prompts/                         — versioned prompt artifacts (never inline — see prompts/README.md)
evals/                           — memory-quality harnesses (see evals/README.md)
migrations/                      — V1 squashed schema (standalone identity + RLS)
docs/index.md                    — doc router (canonical vs archive)
docs/explanation/                — extraction notes, competitive landscape
docs/decisions/adrs/             — active MADR decision records
docs/archive/extraction/         — host-era design/ADRs (do not cite as current)
docs/enhancements/               — proposals (paths may still say platform/ — verify in code)
docs/tutorials|how-to|reference/ — Diátaxis slots (mostly empty; fill as needed)
.opengrep/rules/                 — custom semantic lint rules
.agents/skills/                  — reusable agent skills
.agents/memory/                  — agent memory, git-tracked
```

## Hard rules

<!-- ── PROJECT INVARIANTS SLOT ────────────────────────────────────────────── -->
- Tenant-scoped DB access goes through `qortia.db.tenant_transaction` (RLS GUCs).
- Auth is API-key + `X-Agent-Id` — no JWT, no Vault, no host IDP in this repo.
- Embeddings go through `qortia.embeddings` only; run `qortia-worker` / `just worker`
  alongside the API (`docs/how-to/embeddings.md`). Model/dim via
  `QORTIA_EMBEDDING_MODEL` / `QORTIA_EMBEDDING_DIMENSION` (defaults `bge-m3` / `1024`).
- `ARCHITECTURE.md` wins over archived extraction docs on conflict.
- New significant decisions → MADR under `docs/decisions/adrs/` (never append to the archive ADR dump).
<!-- ─────────────────────────────────────────────────────────────────────── -->

## How to work here

- `just ci` before every push — runs lint, type-check, import boundaries, tests, docs, architecture coverage.
- `just fmt` to auto-format (ruff format + ruff check --fix).
- `just sync` after changing `pyproject.toml`.
- Conventional commits enforced: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`.
- All significant decisions → MADR ADR in `docs/decisions/adrs/` first.
- Coverage ratchet: `fail_under` in `[tool.coverage.report]` (and matching
  `--cov-fail-under` in pytest `addopts`) — only ever increase it. Initial floor is
  stamped from the Copier answer `coverage_fail_under` (per package via
  `python_package_name` as the cov target); raise both values together after measuring.

## Architecture documentation

Every top-level module or package directly under `src/qortia/` needs a
matching `## <Name>` section in `ARCHITECTURE.md`: purpose, public interface, dependencies, and
— if its internal flow isn't obvious from the name — a small ASCII/Unicode box diagram in a
plain (untagged) fenced code block. Hand-drawn, not Mermaid: a diagram you redraw in the same PR
as the code change, not a separate rendering step.

- `just ci-arch` blocks CI if a component has no section — mechanical presence only.
- It does **not** verify the content still matches the code. Run `/arch-review`
  (`.agents/skills/arch-review/SKILL.md`) periodically, or after a PR that changes a documented
  component's behavior, to catch drift the coverage check can't see.
- Add the component's section in the **same PR** that adds the component — don't defer it.

## Agent-memory convention

After a significant multi-step task, save reusable learnings to `.agents/memory/` — the
canonical, git-tracked location (survives a devcontainer rebuild and travels with the repo).
On Claude Code, `~/.claude/projects/<project-slug>/memory/` is a symlink `post-create.sh`
maintains into the same files, not a separate location — write to `.agents/memory/` directly
so the memory exists even if that symlink hasn't been (re)created yet.

One file per insight; `MEMORY.md` index (one line per entry). This prevents knowledge
loss across sessions and feeds the coaching loop (Factor 2/3 of 12-factor-agents).

## Harness-specific configuration

This project's safety/quality floor comes from two different places: universally, in
CI and git (Guardrails, Ratchet — run regardless of which harness or human wrote the
commit), and per-harness, via that harness's own config — which requires a human to
explicitly trust this project on their machine before it does anything at all:

- **Claude Code**: `.claude/settings.json` (`permissions.allow`/`deny`) — active once
  you accept the workspace-trust dialog.
- **Codex CLI**: `.codex/config.toml` (`approval_policy`, `sandbox_mode`) and
  `.codex/rules/default.rules` (denies `git commit --no-verify` /
  `git push --no-verify`, best-effort — see that file's own comment for the exact
  matching limitation) — active only once you mark this project trusted in Codex;
  Codex skips `.codex/` entirely for untrusted projects.
- **goose**: no project-level config file exists for goose at all (confirmed against
  block/goose's own docs). `GOOSE_MODE=smart_approve` is instead set for every
  process in this devcontainer via `.devcontainer/devcontainer.json`'s
  `containerEnv` — a coarser, session-wide posture, not a per-command allow/deny
  list. A `goose` run anywhere outside this devcontainer does not inherit it.

None of the three is a substitute for the others, and none is a substitute for CI.
Stamping a config file is necessary, never sufficient. See
[docs/explanation/design.md](docs/explanation/design.md#harness-neutrality-whats-guaranteed-vs-whats-harness-enhanced)
for the full reasoning.

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
  [chassis's docs/explanation/ai-usage-cockpit.md](https://github.com/kaiverse-io/chassis/blob/main/docs/explanation/ai-usage-cockpit.md)
  for the full writeup and [ADR-002](https://github.com/kaiverse-io/chassis/blob/main/docs/decisions/adrs/adr-002-ai-usage-cockpit.md).

<!-- cockpit-section:end -->

## Forbidden patterns — never do these

- `git commit --no-verify` — fix the violation, never skip the gate.
- Inline prompt literals (200+ chars) — move to `prompts/` and load at runtime.
- Bare `# noqa` or `# type: ignore` without a specific code and reason.
- Hardcode secrets — use `.env` (gitignored) and `.env.example` as the template.
- Lower the coverage floor (`fail_under` or `--cov-fail-under`) — raise both together.
- `continue-on-error: true` in CI to silence a failing step.
- `pre-commit uninstall` — the hooks are the guardrails, not obstacles.
- Commit to `main` without CI green.
