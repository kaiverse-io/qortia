---
kind: index
status: active
owner: platform
last_reviewed: 2026-05-30
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

| ADR | Decision | Status |
|-----|---------|--------|
| [ADR-041](../decisions/qortia.md) | Embedding worker per-row error isolation | implemented |
| [ADR-054](../decisions/qortia.md) | NER entity extraction at write time | implemented |
| [ADR-055](../decisions/qortia.md) | Dynamic importance: recall_count + last_recalled_at | implemented |
| [ADR-056](../decisions/qortia.md) | Thought trace preservation (cognitive persistence) | implemented |
| [ADR-071](../decisions/platform-api.md) | HNSW index parameters | implemented |
| [ADR-074](../decisions/qortia.md) | Knowledge candidate keyword boost before MMR | implemented |
| [ADR-078](../decisions/qortia.md) | Bi-temporal fact bounds: valid_from / valid_until on all recall paths | implemented |
| [ADR-080](../decisions/qortia.md) | Org memory RBAC: two-axis access control (clearance + division) | implemented |
| [ADR-105](../decisions/qortia.md) | Memory quality: MRL + dedup strategy | implemented |
| [ADR-125](../decisions/qortia.md) | Causal tracking + outcome-driven confidence decay (dark-launch) | implemented |

## Open Enhancements

| File | Title | Status |
|------|-------|--------|
| [enhancements/qortia-background-reflection-trigger.md](enhancements/qortia-background-reflection-trigger.md) | Background reflection trigger | implemented |
| [enhancements/qortia-extraction-prompt-improvements.md](enhancements/qortia-extraction-prompt-improvements.md) | Extraction prompt improvements | implemented |
| [enhancements/qortia-semantic-embedding-cache.md](enhancements/qortia-semantic-embedding-cache.md) | Semantic embedding cache (LRU) | implemented |
| [enhancements/qortia-temporal-fact-bounds.md](enhancements/qortia-temporal-fact-bounds.md) | Temporal fact bounds (valid_from/valid_until) | implemented |
| [enhancements/qortia-user-profile-synthesis.md](enhancements/qortia-user-profile-synthesis.md) | Stakeholder profile synthesis | post-mvp |
| [enhancements/multilingual-memory.md](enhancements/multilingual-memory.md) | Multilingual memory | implemented |
| [enhancements/recall-cross-encoder-profiles.md](enhancements/recall-cross-encoder-profiles.md) | Cross-encoder reranking profiles | post-mvp |
| [enhancements/recall-importance-decay-type-routed.md](enhancements/recall-importance-decay-type-routed.md) | Importance decay type-routed | implemented |
| [enhancements/recall-rerank-model-decoupling.md](enhancements/recall-rerank-model-decoupling.md) | Rerank model decoupling | implemented |

## Key Modules

| Module | Purpose |
|--------|---------|
| `platform/app/qortia/recall.py` | Main recall pipeline — 10 SQL paths, RRF fusion, `valid_until` filter |
| `platform/app/qortia/recall_helpers.py` | `dynamic_importance()`, `_cosine()`, `_mmr()`, RRF utils |
| `platform/app/qortia/recall_rerank.py` | Cross-encoder reranking (post-MVP, gated behind `#74`) |
| `platform/app/qortia/links.py` | Write-time and read-time entity link expansion |
| `platform/app/qortia/reflect.py` | Reflection / consolidation LLM pipeline |
| `platform/app/qortia/remember.py` | Memory ingestion |
| `platform/app/qortia/knowledge.py` | Knowledge corpus ingestion |
| `platform/app/qortia/entity_graph.py` | BFS entity graph traversal |
| `platform/app/qortia/models.py` | Pydantic models: `RecallResult`, `RecallRequest`, etc. |
| `platform/app/qortia/eval_router.py` | Internal eval endpoints (EVAL_MODE guard — never in prod) |

## Related Components

- [Platform](../platform/index.md) — API surface, tenant isolation, work orders
- [Agent](../agent/index.md) — MCP bridge calls remember/recall, reflection trigger
- [Infrastructure](../infrastructure/index.md) — pgvector, Postgres, embedding worker
