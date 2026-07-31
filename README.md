# Qortia

Portable memory layer for AI agents

## Setup

```bash
uv sync          # install deps
just ci          # run all checks
```

## Development

| Command | What it does |
|---|---|
| `just ci` | Full CI: lint + type-check + test + docs |
| `just fmt` | Auto-format (ruff) |
| `just sync` | Sync deps with uv |
| `just lint` | Run all pre-commit hooks |
| `just metrics` | AI-usage cost report (requires codeburn) |

## Docs

Docs follow the [Diátaxis](https://diataxis.fr/) structure:

- `docs/tutorials/` — learning-oriented
- `docs/how-to/` — task-oriented
- `docs/reference/` — API / config reference
- `docs/explanation/` — concepts and theory
- `docs/decisions/adrs/` — architecture decision records (MADR)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
