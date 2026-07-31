# Qortia — task runner
# https://just.systems  |  run `just` to see all targets

set dotenv-load := true

default:
    @just --list

# ── CI gates (called by .github/workflows/ci.yaml) ───────────────────────────

ci: ci-lint ci-test ci-docs

ci-lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src/
    uv run lint-imports

ci-test:
    uv run pytest

# Advisory in v1 — install linkchecker to activate
ci-docs:
    @echo "doc link-check: install linkchecker and replace this stub"

# ── Dev helpers ───────────────────────────────────────────────────────────────

fmt:
    uv run ruff format .
    uv run ruff check --fix .

sync:
    uv sync

lint:
    pre-commit run --all-files

# ── On-demand (bucket C) ─────────────────────────────────────────────────────

# AI-usage cost/burn report — installed by .devcontainer/post-create.sh
metrics:
    @command -v codeburn >/dev/null || (echo "install codeburn first: npm install -g codeburn" && exit 1)
    codeburn

# Live session monitor (context %, tokens, rate limits) — install: cargo install abtop
monitor:
    @command -v abtop >/dev/null || (echo "install abtop first: cargo install abtop" && exit 1)
    abtop

# Run the dev-coach skill's analysis (also invokable as /dev-coach in Claude Code)
coach:
    @echo "Run '/dev-coach' in Claude Code, or see .agents/skills/dev-coach/SKILL.md"
