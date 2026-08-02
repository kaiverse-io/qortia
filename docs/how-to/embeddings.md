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
| Gateway | `QORTIA_LITELLM_URL` | `http://localhost:4000` | OpenAI-compatible base URL |
| API key | `QORTIA_LITELLM_API_KEY` | _(empty)_ | Required for live embed/validate |

NER (spaCy `en_core_web_sm` / `xx_ent_wiki_sm`) is **independent** of the embedder.
Language routing for Indic NER does not swap the embedding model — BGE-M3 covers
multilingual semantic search in one vector space (same decision as the host-era
ADR that retired dual-model IndicSBERT routing).

## Minimal local stack

```bash
# 1. Apply migrations (vector(1024) columns)
# 2. Point LiteLLM/Ollama at a BGE-M3 (or compatible 1024-dim) model
# 3. Export env (see .env.example)
export QORTIA_DATABASE_URL=postgresql://...
export QORTIA_LITELLM_URL=http://localhost:4000
export QORTIA_LITELLM_API_KEY=sk-...

# API (validates model/dim at startup when API key is set)
uvicorn qortia.app:app --host 0.0.0.0 --port 8080

# Worker (fills NULL embeddings, archival, idle-reflect, weekly summary)
just worker
# or: qortia-worker --only embed
```

Without the worker, remember/reflect still write rows, but vector recall stays
cold until embeddings are filled.

## Live smoke test

Requires Docker. Spins up pgvector, an `/embeddings` backend, the API, and
`qortia-worker --only embed`, then runs remember → embed → recall:

```bash
# Deterministic mock vectors (CI / no model download)
bash scripts/e2e_embeddings_live.sh

# Real BGE-M3 via Ollama (install ollama, then: ollama serve && ollama pull bge-m3)
EMBED_BACKEND=ollama bash scripts/e2e_embeddings_live.sh
```

`QORTIA_LITELLM_URL` is any OpenAI-compatible base (LiteLLM proxy, Ollama
`…/v1`, TEI, vLLM). LiteLLM is the recommended *gateway* for multi-tenant keys
and tracing — not the inference engine. See ADR-002.

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
