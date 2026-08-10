# Qortia — task runner
# https://just.systems  |  run `just` to see all targets

set dotenv-load := true

default:
    @just --list

# ── CI gates (called by .github/workflows/ci.yaml) ───────────────────────────

ci: ci-lint ci-test ci-docs ci-arch

ci-lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src/
    uv run lint-imports
    gitleaks detect --source . --verbose
    opengrep scan --config .opengrep/rules/ . --error --severity ERROR

ci-test:
    uv run pytest

# Advisory in v1 — install linkchecker to activate
ci-docs:
    @echo "doc link-check: install linkchecker and replace this stub"

# Architecture coverage: every top-level src/qortia/ component must have a
# matching ARCHITECTURE.md section (see AGENTS.md "Architecture documentation"). Mechanical
# presence check only — semantic accuracy is the /arch-review skill's job, not this gate's.
ci-arch:
    #!/usr/bin/env python3
    import pathlib, re, sys

    pkg_dir = pathlib.Path("src/qortia")
    arch_doc = pathlib.Path("ARCHITECTURE.md")
    excluded = {"__init__.py", "__pycache__", "py.typed"}

    def normalize(name: str) -> str:
        return re.sub(r"[-_\s]+", " ", name).strip().lower()

    components = [
        item.stem if item.suffix == ".py" else item.name
        for item in sorted(pkg_dir.iterdir())
        if item.name not in excluded and not item.name.startswith(".")
        and (item.is_dir() or item.suffix == ".py")
    ]

    if not components:
        print("Architecture coverage: no top-level components under src/qortia/ yet — nothing to check.")
        sys.exit(0)

    if not arch_doc.exists():
        print(f"::error::ARCHITECTURE.md is missing but src/qortia/ has {len(components)} component(s): {', '.join(components)}", file=sys.stderr)
        sys.exit(1)

    headings = {
        normalize(m.group(1))
        for m in re.finditer(r"^#{2,3}\s+(.+)$", arch_doc.read_text(), re.MULTILINE)
    }

    missing = [c for c in components if normalize(c) not in headings]
    if missing:
        print("::error::ARCHITECTURE.md is missing a section for: " + ", ".join(missing), file=sys.stderr)
        print("::error::Add a '## <Name>' section (purpose, dependencies, ASCII diagram if non-trivial) — see AGENTS.md.", file=sys.stderr)
        sys.exit(1)

    print(f"Architecture coverage: all {len(components)} component(s) documented in ARCHITECTURE.md.")

# ── Dev helpers ───────────────────────────────────────────────────────────────

fmt:
    uv run ruff format .
    uv run ruff check --fix .

sync:
    uv sync

lint:
    pre-commit run --all-files

# Background workers (embeddings, archival, idle-reflect, weekly-summary).
# Run alongside `uvicorn qortia.app:app` — see docs/how-to/embeddings.md.
worker *args:
    uv run qortia-worker {{ args }}

# Full stack: Postgres+pgvector + Ollama + the Qortia API + worker.
# Embeddings hit Ollama directly — layer on docker-compose.gateway.yml for
# LiteLLM (virtual keys / budgets / multi-tenant OTel, ADR-003).
stack-up:
    docker compose up -d --build
    # A single $ here, not $$: just doesn't collapse $$ to a literal $ in plain
    # recipe text (that's a Makefile convention, not just's) — it was passed
    # straight through to sh, where $$ is the shell's own PID variable,
    # immediately followed by a bare "(seq 1 60)" it can't parse. The recipe
    # silently never ran its health-check wait — it always fell straight
    # through to a syntax error, on every invocation, not just sometimes.
    @echo "Waiting for API…"; \
      for i in $(seq 1 60); do \
        curl -sf http://127.0.0.1:${QORTIA_APP_PORT:-8081}/docs >/dev/null 2>&1 && break; \
        sleep 1; \
      done; \
      echo "stack up — API :${QORTIA_APP_PORT:-8081}  DB :5434  Ollama :11434"

stack-down:
    docker compose down

stack-pull-model:
    docker compose exec ollama ollama pull bge-m3

# Live E2E (mock | ollama | litellm). See docs/how-to/embeddings.md.
e2e-embeddings backend="mock":
    EMBED_BACKEND={{ backend }} bash scripts/e2e_embeddings_live.sh

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
