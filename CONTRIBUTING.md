---
audience: contributor
last_reviewed: <!-- update when reviewing -->
---

# Contributing to Qortia

## Workflow

1. Branch off `main`.
2. Make changes; run `just fmt` then `just ci` locally — CI must be green before pushing.
3. Commit with [conventional commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `perf:`.
4. Open a PR; all CI checks must pass.

## Decisions

Significant decisions go in `docs/decisions/adrs/` as [MADR](https://adr.github.io/madr/) files.
Copy `docs/decisions/adrs/adr-000-madr-template.md`, increment the number, fill it in.

## Secrets

- Never commit secrets or credentials.
- Copy `.env.example` to `.env` (gitignored) and fill in values.
- `gitleaks` runs in pre-commit and will block accidental leaks.

## Guardrail rules

See the **Forbidden patterns** section of `AGENTS.md` — these apply to humans and agents equally.
