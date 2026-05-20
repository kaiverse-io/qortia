---
kind: index
status: active
owner: platform
last_reviewed: 2026-05-20
---

# Qortia — Component Index

Memory service: episodic recall, reflection, org knowledge, RBAC, embedding, benchmarking.

## Design Docs

| File | Purpose |
|------|---------|
| [01-design.md](01-design.md) | Consolidated design: data model, recall pipeline, reflection, RBAC, API surface |
| [02-benchmarking.md](02-benchmarking.md) | Memory layer benchmarking guide — latency, quality, throughput targets |
| [03-eval-strategy.md](03-eval-strategy.md) | Evaluation strategy — why and how we measure memory quality |
| [04-api-contracts.md](04-api-contracts.md) | Authoritative request/response schemas — recall, remember, reflect, forget, knowledge, HNSW params, NER, dynamic importance, RBAC |

## Key Decisions (ADRs)

| ADR | Decision |
|-----|---------|
| [ADR-041](../decisions/qortia.md) | Embedding worker per-row error isolation |
| [ADR-054](../decisions/qortia.md) | NER entity extraction at write time |
| [ADR-055](../decisions/qortia.md) | Dynamic importance: recall_count + last_recalled_at |
| [ADR-056](../decisions/qortia.md) | Thought trace preservation (cognitive persistence) |
| [ADR-071](../decisions/platform-api.md) | HNSW index parameters |
| [ADR-074](../decisions/qortia.md) | Knowledge candidate keyword boost before MMR |
| [ADR-080](../decisions/qortia.md) | Org memory RBAC: two-axis access control (clearance + division) |
| [ADR-105](../decisions/qortia.md) | Memory quality: MRL + dedup strategy |

## Open Enhancements

| File | Title | Status |
|------|-------|--------|
| [enhancements/qortia-background-reflection-trigger.md](enhancements/qortia-background-reflection-trigger.md) | Background reflection trigger | post-mvp |
| [enhancements/qortia-extraction-prompt-improvements.md](enhancements/qortia-extraction-prompt-improvements.md) | Extraction prompt improvements | post-mvp |
| [enhancements/qortia-semantic-embedding-cache.md](enhancements/qortia-semantic-embedding-cache.md) | Semantic embedding cache (LRU) | post-mvp |
| [enhancements/qortia-temporal-fact-bounds.md](enhancements/qortia-temporal-fact-bounds.md) | Temporal fact bounds (valid_from/valid_until) | post-mvp |
| [enhancements/qortia-user-profile-synthesis.md](enhancements/qortia-user-profile-synthesis.md) | Stakeholder profile synthesis | post-mvp |
| [enhancements/multilingual-memory.md](enhancements/multilingual-memory.md) | Multilingual memory | post-mvp |
| [enhancements/recall-cross-encoder-profiles.md](enhancements/recall-cross-encoder-profiles.md) | Cross-encoder reranking profiles | post-mvp |
| [enhancements/recall-importance-decay-type-routed.md](enhancements/recall-importance-decay-type-routed.md) | Importance decay type-routed | post-mvp |
| [enhancements/recall-rerank-model-decoupling.md](enhancements/recall-rerank-model-decoupling.md) | Rerank model decoupling | post-mvp |

## Related Components

- [Platform](../platform/index.md) — API surface, tenant isolation, work orders
- [Agent](../agent/index.md) — MCP bridge calls remember/recall, reflection trigger
- [Infrastructure](../infrastructure/index.md) — pgvector, Postgres, embedding worker
