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

Full stack, one command — Postgres, Ollama, the API, and the background worker:

```bash
just stack-up            # docker compose up --build; API at :8081/docs
just stack-pull-model    # ollama pull bge-m3 (once)
```

Embeddings talk to Ollama directly by default. Need per-tenant virtual keys,
budgets, or OTel attribution instead? Layer on the optional gateway compose
file, which adds LiteLLM in front of Ollama and repoints the API/worker at it:

```bash
docker compose -f docker-compose.yml -f docker-compose.gateway.yml up --build
```

(see [ADR-003](docs/decisions/adrs/)).

Iterating on the app itself without rebuilding images:

```bash
cp .env.example .env
uv sync
docker compose up -d db ollama
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
