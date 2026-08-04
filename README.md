# Qortia

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/kaiverse-io/qortia/actions/workflows/ci.yaml/badge.svg)](https://github.com/kaiverse-io/qortia/actions/workflows/ci.yaml)

**Portable memory for AI agents** — a standalone FastAPI service over Postgres + pgvector.

Remember, recall, reflect, and traverse an entity graph. Hybrid lexical + vector search with RRF fusion, background consolidation, and outcome-linked confidence decay. Drop it in beside any agent runtime; tenant isolation is enforced by Postgres RLS, not application hope.

## Features

- **Hybrid recall** — BM25 + embeddings fused with reciprocal rank fusion
- **Entity graph** — link people, projects, and concepts; traverse at query time
- **Consolidation** — background reflect workers keep memory coherent over time
- **Multi-tenant by default** — row-level security with session GUCs as the trust boundary
- **Gateway-friendly** — embeddings and LLMs via LiteLLM-compatible endpoints (local Ollama included)

## Quick start

```bash
cp .env.example .env
uv sync

# Local stack: Postgres + Ollama + LiteLLM
just stack-up
just stack-pull-model

# API + embedding worker
uv run uvicorn qortia.app:app --port 8080
just worker -- --only embed
```

Defaults: embedding model `bge-m3`, dimension `1024`. Per-tenant LiteLLM keys: `QORTIA_LITELLM_TENANT_KEYS` (see [ADR-003](docs/decisions/adrs/)).

```bash
just ci          # lint + type-check + tests + docs + architecture coverage
just e2e-embeddings mock   # remember → embed → recall smoke
just fmt         # ruff format + fix
```

## Documentation

| Doc | Purpose |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Component contracts (enforced by `just ci-arch`) |
| [`docs/index.md`](docs/index.md) | Doc map |
| [`docs/how-to/`](docs/how-to/) | Task recipes (embeddings, ops) |
| [`docs/decisions/adrs/`](docs/decisions/adrs/) | Design decisions |
| [`evals/`](evals/) | Live memory-quality harnesses |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md).

## License

[Apache License 2.0](LICENSE)
