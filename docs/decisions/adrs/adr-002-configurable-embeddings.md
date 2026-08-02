---
kind: adr
status: accepted
owner: platform
last_reviewed: 2026-08-02
---

# ADR-002 — Configurable embeddings (model + dimension) with a worker process

- **Status:** Accepted (2026-08-02)
- **Deciders:** founder
- **Related:** ADR-001 (standalone extraction), host-era ADR-076/079/081 in
  [`docs/archive/extraction/legacy-adrs.md`](../../archive/extraction/legacy-adrs.md)

## Context

Standalone Qortia inherited a hardcoded `EMBEDDING_MODEL = "bge-m3"` and
`vector(1024)` schema, with duplicate LiteLLM call sites in `recall` and
`reflect`, and no worker process started by the API. OSS consumers need to:

1. Point at their own LiteLLM/Ollama/OpenAI-compatible gateway
2. Optionally swap models (same dimension) without a code fork
3. Understand that dimension is schema-bound (pgvector)
4. Run embedding fill as a separate process (API stays latency-sensitive)

Industry practice for portable memory layers (mem0-class services): config-driven
embedder, single client module, API + async worker, fail-fast dimension checks.

Multilingual NER (spaCy) stays separate from the embedder. Dual-model Indic
routing was already superseded by a single multilingual BGE-M3 space — keep that.

## Decision

1. **Config surface** (env, defaults = current behavior):
   - `QORTIA_EMBEDDING_MODEL` (default `bge-m3`)
   - `QORTIA_EMBEDDING_DIMENSION` (default `1024`)
   - existing `QORTIA_LITELLM_URL` / `QORTIA_LITELLM_API_KEY`
2. **One module** `qortia.embeddings` owns all `/embeddings` HTTP calls
   (`embed_text`, `embed_query` with cache, `validate_embedding_config`).
3. **Schema constant** `SCHEMA_EMBEDDING_DIMENSION = 1024` must match migration V1;
   startup errors if settings.dimension differs (dimension change = migration +
   re-embed, documented in `docs/how-to/embeddings.md`).
4. **Worker entrypoint** `qortia-worker` / `just worker` runs embedding (+ optional
   archival / idle-reflect / weekly-summary) outside the API process.
5. **Cache keys** include the model name so model swaps cannot serve stale vectors.
6. **Validate at API + worker startup** when an API key is configured; skip the
   live probe when the key is empty (local tests / docs builds).

## Gateway vs engine (and multi-tenant tracing)

`QORTIA_LITELLM_URL` names an **OpenAI-compatible base URL**, not a commitment
to run inference inside LiteLLM:

| Layer | Role | Typical choice |
|-------|------|----------------|
| Gateway | Virtual keys, budgets, routing, OTel | **LiteLLM Proxy** (prod multi-tenant) |
| Engine | Actually run BGE-M3 | Ollama (dev), TEI / vLLM (prod embeddings) |

For multi-tenant tracing: keep one LiteLLM Proxy in front; issue a virtual key
per tenant; enable LiteLLM OpenTelemetry (v2) and put `aither.tenant_id` /
`qortia.tenant_id` on every embed request via LiteLLM `metadata` / `user` (or
pass-through headers). Qortia v1 uses a single configured key — per-tenant
virtual keys are the next step when more than one tenant shares a gateway.
Do **not** replace the OpenAI-compatible client with an in-process
sentence-transformers call: that breaks swapability and loses the gateway’s
auth/budget/trace seam.

## Consequences

**Good:** OSS operators configure gateway + model without forking; call sites stop
diverging; worker is an explicit part of the setup story.

**Trade-off:** True dimension changes remain a deliberate migration (pgvector),
not a hot config flip — that is correct for index safety.
