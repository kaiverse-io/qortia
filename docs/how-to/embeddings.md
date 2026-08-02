---
kind: how-to
status: active
last_reviewed: 2026-08-02
---

# How to configure embeddings (OSS setup)

Qortia stores vectors in Postgres+pgvector and obtains them from any
LiteLLM-compatible `/embeddings` endpoint. This is the portable OSS shape used
by most memory layers: **API process + embedding worker + configurable model**.

## Defaults (shipped)

| Setting | Env | Default | Notes |
|---------|-----|---------|--------|
| Model | `QORTIA_EMBEDDING_MODEL` | `bge-m3` | Multilingual; register this alias in LiteLLM/Ollama |
| Dimension | `QORTIA_EMBEDDING_DIMENSION` | `1024` | **Must** match `vector(1024)` in `migrations/V1` |
| Gateway | `QORTIA_LITELLM_URL` | `http://localhost:4000` | OpenAI-compatible **gateway** base |
| API key | `QORTIA_LITELLM_API_KEY` | _(empty)_ | Shared/master key; required for live embed/validate |
| Tenant keys | `QORTIA_LITELLM_TENANT_KEYS` | `{}` | JSON `{tenant_id: virtual_key}` (ADR-003) |

NER (spaCy `en_core_web_sm` / `xx_ent_wiki_sm`) is **independent** of the embedder.
Language routing for Indic NER does not swap the embedding model — BGE-M3 covers
multilingual semantic search in one vector space (same decision as the host-era
ADR that retired dual-model IndicSBERT routing).

## Minimal local stack (recommended)

```bash
# Infra: Postgres+pgvector + Ollama (engine) + LiteLLM (gateway)
just stack-up
just stack-pull-model   # once — pulls bge-m3 into the ollama container
# Or reuse a host cache: OLLAMA_MODELS_DIR=$HOME/.ollama just stack-up

cp .env.example .env    # QORTIA_LITELLM_URL=http://127.0.0.1:4000

# API + worker on the host
uv run uvicorn qortia.app:app --host 127.0.0.1 --port 8080
just worker -- --only embed
```

Without the worker, remember/reflect still write rows, but vector recall stays
cold until embeddings are filled.

### Multi-tenant keys + tracing

1. Mint a LiteLLM **virtual key** per tenant (Admin UI / `/key/generate`).
2. Map them: `QORTIA_LITELLM_TENANT_KEYS='{"<tenant-uuid>":"sk-…"}'`.
3. Enable LiteLLM OpenTelemetry v2 on the proxy (`LITELLM_OTEL_V2=true` + OTLP
   exporter). Qortia already sends `user=<tenant_id>` and
   `metadata.qortia.tenant_id` on every embed call.

See [ADR-003](../decisions/adrs/adr-003-litellm-gateway-tenant-tracing.md).

## Live smoke test

Requires Docker. Spins up pgvector (ephemeral), an `/embeddings` backend, the
API, and `qortia-worker --only embed`, then runs remember → embed → recall:

```bash
just e2e-embeddings                  # mock (CI)
just e2e-embeddings ollama           # host ollama
just stack-up && just stack-pull-model
just e2e-embeddings litellm          # gateway → ollama (real BGE-M3)
```

`QORTIA_LITELLM_URL` is any OpenAI-compatible base. Prefer the LiteLLM gateway
for multi-tenant keys/tracing; Ollama/`…/v1` is fine for single-tenant dogfood.
See ADR-002 / ADR-003.

## Changing the model (same dimension)

1. Point LiteLLM at the new alias; set `QORTIA_EMBEDDING_MODEL`.
2. Keep `QORTIA_EMBEDDING_DIMENSION=1024`.
3. NULL existing vectors and re-run the worker (different embedding spaces must
   not be mixed in one index):

```sql
UPDATE hindsight_memories SET embedding = NULL, embedding_attempts = 0;
UPDATE org_memory SET embedding = NULL, embedding_attempts = 0;
UPDATE org_knowledge SET embedding = NULL, embedding_attempts = 0;
UPDATE qortia_entities SET embedding = NULL, embedding_attempts = 0;
```

4. Re-tune `QORTIA_DEDUP_SIMILARITY_THRESHOLD` if the new model’s cosine scale differs.

## Changing the dimension

pgvector column width is DDL-fixed. You must:

1. Ship a new migration that drops HNSW indexes, alters all four `embedding`
   columns to `vector(N)`, recreates indexes, and NULLs existing vectors.
2. Set `QORTIA_EMBEDDING_DIMENSION=N` (must match the migration).
3. Bump `SCHEMA_EMBEDDING_DIMENSION` in `src/qortia/embeddings.py` in the same PR.
4. Re-run `qortia-worker --only embed`.

Startup refuses to boot if `QORTIA_EMBEDDING_DIMENSION` ≠ schema constant.

## Code ownership

| Call | Module |
|------|--------|
| Write / worker embed | `qortia.embeddings.embed_text` |
| Recall query embed (+ cache) | `qortia.embeddings.embed_query` |
| Startup / worker validate | `qortia.embeddings.validate_embedding_config` |

Do not `POST /embeddings` from other modules.

## Related

- [`docs/decisions/adrs/adr-002-configurable-embeddings.md`](../decisions/adrs/adr-002-configurable-embeddings.md)
- [`ARCHITECTURE.md`](../../ARCHITECTURE.md) — `embeddings`, `workers`, `embedding_cache`
