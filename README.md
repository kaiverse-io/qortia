# Qortia

Portable memory layer for AI agents — standalone FastAPI service over
Postgres+pgvector (remember / recall / reflect / knowledge).

## Setup

```bash
uv sync          # install deps
just ci          # run all checks
```

## Development

| Command | What it does |
|---|---|
| `just ci` | Full CI: lint + type-check + test + docs + arch coverage |
| `just fmt` | Auto-format (ruff) |
| `just sync` | Sync deps with uv |
| `just lint` | Run all pre-commit hooks |
| `just metrics` | AI-usage cost report (requires codeburn) |

## Docs

**Start with [`ARCHITECTURE.md`](ARCHITECTURE.md)** and [`docs/index.md`](docs/index.md).

| Path | Role |
|---|---|
| `ARCHITECTURE.md` | Current component contracts (`just ci-arch`) |
| `docs/explanation/` | Extraction notes, competitive landscape |
| `docs/decisions/adrs/` | Active MADRs (e.g. standalone extraction) |
| `docs/archive/extraction/` | Host-platform-era design/ADRs — historical only |
| `docs/enhancements/` | Feature proposals (verify paths against `src/qortia/`) |
| `evals/` | Live memory-quality harnesses |

Diátaxis slots (`docs/tutorials/`, `how-to/`, `reference/`) exist for new material.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
