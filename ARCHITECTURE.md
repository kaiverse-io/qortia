# Qortia Architecture

> Hand-maintained, not generated. `just ci-arch` fails CI if a top-level module/package under
> `src/qortia/` has no matching section here — see `AGENTS.md` "Architecture
> documentation". Diagrams are plain ASCII/Unicode box-drawing text in an untagged fenced code
> block, not Mermaid — a diagram you can update in the same PR as the code, not a separate
> rendering step.
>
> This file is the **canonical** description of standalone Qortia. Host-platform extraction
> dumps (old design docs / ADR dump) live under
> [`docs/archive/extraction/`](docs/archive/extraction/) and must not override this document.
> See also [`docs/index.md`](docs/index.md) and
> [`docs/decisions/adrs/adr-001-standalone-extraction.md`](docs/decisions/adrs/adr-001-standalone-extraction.md).

## Executive Summary

Qortia is a standalone Postgres+pgvector memory layer for AI agents — remember/recall/reflect
endpoints backed by hybrid (BM25 + vector) search, an entity graph, and LLM-driven consolidation,
extracted this week from a larger internal monorepo so it can run independently (like mem0 or
Hindsight) against any Postgres+pgvector instance and any LiteLLM-compatible model gateway. Two
decisions shape everything else a contributor will read in the code. First, **tenant isolation is
enforced by Postgres row-level security, not application logic**: every DB operation runs inside
`qortia.db.tenant_transaction`, which sets `app.tenant_id`/`app.agent_id`/
`app.memory_clearance_order`/`app.agent_division` as session GUCs that RLS policies on every
tenant-scoped table check — the application code is not the trust boundary, the database is.
Second, **auth is deliberately simple**: a single SHA-256-hashed API key per tenant (no JWT, no
external IDP, no Vault), because qortia is meant to be dropped in as a portable component rather
than assume a host platform's identity infrastructure is present.

## System Overview

```
                                  HTTP clients (agent runtime)
                                            │
                                            ▼
                              ┌──────────────────────────┐
                              │   qortia.app (FastAPI)    │
                              │   lifespan: init/close     │
                              │   LiteLLM client + PG pool │
                              └─────────────┬─────────────┘
                                            │ mounts
                        ┌───────────────────┼─────────────────────────┐
                        ▼                                             ▼
              qortia.router (/v1/*)                    qortia.eval_router (/v1/internal/eval/*)
      remember + reflect + recall + knowledge                 only mounted if eval_mode=true
                        │                                             │
                        ▼                                             │
              qortia.auth.require_agent                               │
        API key (Bearer) → SHA-256 → tenant_id                        │
        X-Agent-Id → verified belongs to tenant                       │
        → AgentIdentity(agent_id, tenant_id,                          │
                         clearance_order, division)                   │
                        │                                             │
                        ▼                                             ▼
        ┌──────────────────────────────────────────────────────────────┐
        │              core memory pipeline                             │
        │  remember / recall / reflect / knowledge / entity_graph/links │
        │  (recall_helpers, recall_rerank = pure/algorithmic support)   │
        └───────────────────────────┬─────────────────┬─────────────────┘
                                     │                 │
                                     ▼                 ▼
                  ┌───────────────────────────┐   ┌───────────────────────────┐
                  │  Postgres + pgvector        │   │  LiteLLM gateway            │
                  │  RLS-scoped via              │   │  (embeddings, chat/rerank,  │
                  │  tenant_transaction()        │   │   configurable model)       │
                  └───────────────────────────┘   └───────────────────────────┘
```

## Components

Each top-level module or package directly under `src/qortia/` gets a `##`
section below: what it's for, its public interface, what it depends on, and — if its internal
flow isn't obvious from the name — a small ASCII diagram. `just ci-arch` only checks that the
section exists (mechanical coverage); it does not verify the content still matches the code. Run
the `/arch-review` skill (`.agents/skills/arch-review/SKILL.md`) periodically, or after a PR that
changes a documented component's behavior, to catch drift a coverage check can't.

### app

Standalone FastAPI entrypoint (`uvicorn qortia.app:app`) — the first time qortia boots on its own
rather than mounted inside a host platform's app. Its `lifespan` context manager initialises the
LiteLLM httpx client and the Postgres pool at startup and closes both at shutdown, and
best-effort loads the spaCy NER model (a load failure is logged and swallowed, since every NER
call site already degrades to an empty entity list). It always mounts `qortia.router.router`
and additionally mounts `qortia.eval_router.router` when `config.settings.eval_mode` is true.

Depends on: `qortia.config`, `qortia.common` (LiteLLM client init/close), `qortia.db` (pool
init/close), `qortia.router`, `qortia.eval_router`, `qortia.knowledge.load_spacy_model`.
External: FastAPI, uvicorn.

### auth

Standalone request authentication — API keys only, no JWT/Vault/external IDP. `require_agent` is
the FastAPI dependency used on almost every endpoint: it hashes the `Authorization: Bearer <key>`
header with SHA-256 and looks it up in `qortia_api_keys` to resolve a tenant, then checks the
`X-Agent-Id` header names an agent that actually belongs to that tenant before trusting it — this
is what stops tenant A's key from addressing tenant B's agent. Keys are stored only as SHA-256
hashes (deliberate: high-entropy random tokens, not user passwords, so a slow password hash would
only add latency on every request for no benefit).

Public interface: `AgentIdentity` (agent_id, tenant_id, clearance_order, division),
`hash_api_key`, `require_agent`, `get_litellm_key` (per-tenant virtual key from
`QORTIA_LITELLM_TENANT_KEYS`, else shared `QORTIA_LITELLM_API_KEY` — ADR-003),
`get_platform_embed_key` (shared/master key for probes), `provision_eval_litellm_key` (no-op).

Depends on: `qortia.config`, `qortia.db`. Used via `Depends(require_agent)` by `remember.py`,
`recall.py`, `reflect.py`, `knowledge.py`, and `eval_router.py`.

```
Authorization: Bearer <api_key>        X-Agent-Id: <uuid>
         │                                     │
         ▼                                     │
  SHA-256(key) ──▶ qortia_api_keys             │
         │        (key_hash, revoked_at)       │
         ▼                                     │
     tenant_id                                 │
         │                                     ▼
         └──────────────▶ qortia_agents JOIN qortia_clearance_levels
                             (does agent belong to tenant_id?)
                                      │
                                      ▼
                        AgentIdentity(agent_id, tenant_id,
                                       clearance_order, division)
```

### common

Small shared runtime pieces used across the request path: the module-level `httpx.AsyncClient`
to LiteLLM (`init_litellm_client`/`close_litellm_client`/`get_litellm_client`), a back-compat
`EMBEDDING_MODEL` alias (prefer `qortia.embeddings.embedding_model()`), and
`assert_agent_active`, which every write/read endpoint calls first to 403 on a non-active agent.

Depends on: `qortia.config` (litellm_url / embedding_model), `httpx`, `asyncpg`, `fastapi`.

### config

Env-driven runtime settings: the `Settings` dataclass, `load_settings()`, and the module-level
`settings` singleton. Includes database URL, LiteLLM URL/API key, **`embedding_model` /
`embedding_dimension`** (defaults `bge-m3` / `1024`), dedup similarity threshold, embedding
cache size/TTL, eval_mode, rerank model, reflection threshold, idle-reflection interval/window —
plain environment variables with sane local defaults (see `.env.example`).

Depends on: nothing internal — stdlib `os`/`dataclasses` only.

### db

Postgres connection pool lifecycle (`init_main_pool`/`close_main_pool`/`get_main_pool`) plus
`tenant_transaction`, the RLS session-variable helper that every tenant-scoped DB operation goes
through. It sets `app.tenant_id`/`app.agent_id`/`app.memory_clearance_order`/`app.agent_division`
as Postgres session GUCs via parameterised `set_config(name, value, is_local=true)` calls — never
f-string SQL, even for tenant-controlled values like `agent_division` — which the RLS policies on
every tenant-scoped table read via `current_setting()`.

Depends on: `qortia.config` (database_url), `asyncpg`. This module is what makes RLS the actual
tenant-isolation boundary described in the Executive Summary; `remember`, `recall`, `reflect`,
`knowledge`, `links`, and `recall_rerank` all call `tenant_transaction` rather than touching the
pool directly.

### embeddings

Single owner of LiteLLM `/embeddings` calls for the whole package (ADR-002).

```
embed_text(text, key, tenant_id?)  ──► LiteLLM /embeddings (+ user/metadata)
embed_query(q, tid, lang) ──► cache? ──► embed_text(tenant_id=tid)
validate_embedding_config() ──► settings.dim == SCHEMA(1024); live probe if API key set
```

Defaults: model `bge-m3`, dimension `1024` (matches `migrations/V1` `vector(1024)`). Changing
dimension requires a migration + re-embed — see `docs/how-to/embeddings.md`. Multilingual NER
(spaCy) is intentionally separate; one multilingual embedder covers semantic search.

Depends on: `qortia.config`, `qortia.common` (httpx client), `qortia.auth` (key),
`qortia.embedding_cache`. Used by `recall`, `reflect`, `app` lifespan, `workers`.

### embedding_cache

In-process per-tenant TTL+LRU cache (`cachetools.TTLCache`) of query embeddings. Cache key is
SHA-256 of `"{tenant_id}:{model}:{lang}:{normalised_query}"` so model swaps cannot serve stale
vectors and cross-tenant hits are impossible. Per-process; guarded by a global lock.

Public interface: `get_cached_embedding`, `put_cached_embedding`, `clear_all_caches` (tests),
`get_cache_stats`.

Depends on: `qortia.config` (max size / TTL / model). Used by `qortia.embeddings.embed_query`.

### workers

CLI entrypoint (`qortia-worker` / `just worker`) that runs background loops outside the API
process: embedding fill, archival, idle-reflection trigger, weekly summary. Same env as the API.
`--only embed|archive|idle-reflect|weekly-summary` selects a subset.

Depends on: `qortia.common`, `qortia.db`, `qortia.embeddings`, `qortia.reflect`, `qortia.knowledge`.

### entity_graph

Maintains the entity graph ("Obsidian layer"): incrementally updates a running per-entity summary
(`_update_entity_summary` — bootstraps from raw content on the first link, LLM-merges every 3rd
link thereafter, non-fatal on LLM failure), archives near-duplicate episodic/experiential
memories via cosine similarity (`_maybe_dedup_memory`, ADR-105 threshold), and batch-links
not-yet-graphed memories into `qortia_entities` (`_populate_graph_batch`, claiming batches of 50
rows with `FOR UPDATE SKIP LOCKED`).

Depends on: `qortia.config`, `qortia.auth.get_litellm_key`, `qortia.common` (LiteLLM client),
`qortia.db`. `_populate_graph_batch` is called from `qortia.reflect.run_embedding_worker`;
`_maybe_dedup_memory` and `_maybe_update_entity_summary` are called from
`qortia.reflect._embed_single_row`, which also re-exports them.

### eval_router

`/v1/internal/eval/*` endpoints that bypass normal auth (no Bearer/X-Agent-Id) so eval harnesses
can seed tenants/agents/memories by raw UUID and call the real recall/reflect/remember-org/
knowledge pipelines directly. Every handler 404s unless `config.settings.eval_mode` is true, and
`qortia.app` only mounts this router at all under that same flag, so it's inert in production.

Public interface: `POST /seed-agent`, `/seed-memory`, `/recall`, `/recall-full`, `/reflect`,
`/remember-org`, `/knowledge`.

Depends on: `qortia.config`, `qortia.auth` (AgentIdentity, provision_eval_litellm_key),
`qortia.db`, `qortia.knowledge`, and (via lazy imports to avoid import-time cycles)
`qortia.recall`, `qortia.reflect`, `qortia.remember`, `qortia.models`.

### knowledge

Org knowledge ingestion pipeline behind `/v1/knowledge` (chief-agent-only): splits incoming
content into heading-delimited, token-budgeted sections (`split_into_sections`/
`_paragraph_split`, backed by `estimate_tokens`), computes a PageIndex-style summary/entities/
questions per chunk (`extract_index_fields`), and content-hash dedups chunks against existing
`org_knowledge` rows so re-ingesting unchanged content is a no-op. It also owns spaCy NER loading
and entity extraction (`load_spacy_model`, `extract_entities`/`extract_entities_with_types`,
routing English through `en_core_web_sm` and hi/bn/ta/te/mr through the multilingual
`xx_ent_wiki_sm` pipeline), and the weekly-summary background task (`run_weekly_summary_task` →
`_summarise_tenant`, leader-elected via `qortia.leader`, staggered per tenant by a hash-based
day-of-week offset so all tenants don't summarise on the same day).

Depends on: `qortia.auth`, `qortia.common`, `qortia.db`, `qortia.leader`, `qortia.models`
(KnowledgeIngestRequest). External: `spacy`. `extract_entities_with_types` is called from
`qortia.remember`, `qortia.reflect`, and `qortia.eval_router`.

### leader

Postgres-advisory-lock leader election (`try_acquire_leader`) for singleton background jobs
across replicas — `pg_try_advisory_lock`/`pg_advisory_unlock` are plain Postgres built-ins, so
this needs no external coordination service. Defines `LOCK_KEY_WEEKLY_SUMMARY`.

Depends on: `asyncpg` only. Used by `qortia.knowledge.run_weekly_summary_task`.

### links

Cross-memory linking (Part 16i). Three functions: `_find_similar_memories` finds up to top-N
cosine-similar active `hindsight_memories` above a 0.70 threshold for a newly embedded memory;
`_upsert_memory_links` writes bidirectional rows into `memory_links` (`ON CONFLICT DO NOTHING`,
safe to call repeatedly); `_expand_with_links` takes a recall result set and appends linked
memories not already present, tagging each with `linked_via` pointing back at the source result.

Depends on: `qortia.db` (get_main_pool, tenant_transaction), `qortia.models` (RecallResult).
Called from `qortia.reflect._embed_single_row` (find + upsert, right after a memory is embedded)
and `qortia.recall._hybrid_recall_pipeline` (expand, via a lazy import).

### models

All Pydantic request/response schemas for the API: `MemoryItem`/`RememberRequest`/
`RememberResponse`, `RememberOrgRequest`/`RememberOrgResponse`, `ForgetRequest`/`ForgetResponse`,
`ContextResponse`/`ContextMemories`/`MemoryEntry`, `ReflectResponse`, `RecallRequest`/
`RecallResult`/`RecallResponse`, `KnowledgeIngestRequest` — plus the `IMPORTANCE`
default-importance-by-type table and the shared `_normalise_lang` BCP-47 helper used by several
validators. `RecallResult` carries private, non-serialised ranking signals (`_recall_count`,
`_last_recalled_at`, `_confidence_multiplier`, `_score`, `_embedding`) that the recall pipeline's
fusion/MMR stages read and mutate internally.

Depends on: `pydantic` only. Imported by nearly every other module (`remember`, `recall`,
`recall_helpers`, `reflect`, `knowledge`, `links`, `eval_router`).

### provisioning

Tenant/agent/API-key provisioning — deliberately CLI (`qortia-admin`) and direct-function-call
only, with no HTTP endpoint: the very first API key for a fresh tenant can't be created through
an API that itself requires an API key to authenticate, so this stays an out-of-band path with
direct DB access. Core functions: `create_tenant`, `create_agent`, `issue_api_key` (returns the
plaintext key once — only its SHA-256 hash is persisted), `revoke_api_key`. `main()` wraps these
in an `argparse` CLI (`create-tenant` / `create-agent` / `issue-key` subcommands).

Depends on: `qortia.config`, `qortia.auth.hash_api_key`, `asyncpg`.

### recall

`/v1/recall` — the hybrid search endpoint. Single-type queries (`type` = decision, lesson,
episodic, or short_term) take a fast single-strategy path (`_recall_decisions`,
`_recall_lessons`, `_recall_episodic`, `_recall_short_term`). Everything else runs
`_hybrid_recall_pipeline`: BM25 and vector search across the requested scopes (private/org/
knowledge) concurrently via `asyncio.gather`, an entity-graph adjacency boost plus 2-hop BFS
traversal (`qortia.recall_rerank._bfs_entity_traversal`), reciprocal-rank fusion of
private+org results and MMR diversification of knowledge candidates (`qortia.recall_helpers`),
cross-memory link expansion of the top results (`qortia.links`), and an optional LLM rerank
(`qortia.recall_rerank._llm_rerank`). After results are assembled the endpoint fires-and-forgets
recall-count/access-time tracking and, if an `X-Work-Order-Id` header is present, session-read
logging for later outcome-based confidence decay (`_record_work_order_outcome`).

Depends on: `qortia.auth`, `qortia.common`, `qortia.db`, `qortia.embedding_cache`,
`qortia.models`, `qortia.recall_helpers`, `qortia.recall_rerank`, `qortia.links` (lazy),
`qortia.knowledge.extract_entities` (lazy), `qortia.telemetry` (degraded-search counter).

```
                          POST /v1/recall
                                │
                type in {decision, lesson, episodic, short_term}?
                  ┌─────────────┴─────────────┐
                 yes                           no
                  │                             │
     single-strategy fast path          _hybrid_recall_pipeline:
   (_recall_decisions, etc.)                    │
                  │              ┌───────────────┴────────────────────┐
                  │              │ BM25 + vector search, per requested  │  asyncio.gather,
                  │              │ scope: private / org / knowledge      │  concurrent
                  │              └───────────────┬────────────────────┘
                  │                              ▼
                  │              entity-graph boost + 2-hop BFS
                  │              (recall_rerank._bfs_entity_traversal)
                  │                              ▼
                  │              RRF fuse private+org (recall_helpers._rrf_fuse)
                  │              MMR diversify knowledge (recall_helpers._mmr)
                  │                              ▼
                  │              cross-memory link expansion (qortia.links)
                  └───────────────┬──────────────┘
                                  ▼
                  optional LLM rerank (recall_rerank._llm_rerank)
                                  ▼
                  fire-and-forget: recall_count/access tracking,
                  work-order session-read logging (X-Work-Order-Id)
                                  ▼
                            RecallResponse
```

### recall_helpers

Pure functions supporting the recall pipeline — no I/O or database access by design (see module
docstring). SQL filter-clause builders that always return parameterised placeholders, never
interpolated values (`_entity_filter_clause`, `_type_filter_clause`, `_temporal_filter_clause`,
`_lang_filter_clause`); `dynamic_importance` (frequency + recency boosted importance);
`_rrf_fuse` (reciprocal rank fusion with an entity-adjacency boost); `_sort_by_importance`;
`_mmr` (maximal marginal relevance diversification); `_cosine`; `_keyword_boost`; and `_to_result`
(DB row → `RecallResult`).

Depends on: `qortia.models` (RecallResult) only. Imported by `qortia.recall` and by
`qortia.reflect` (`_cosine`, for stability-score computation).

### recall_rerank

The two heavier recall augmentations kept out of `recall.py`: `_llm_rerank` (asks
`config.settings.rerank_model` to reorder a result list as a JSON array of indices, falling back
to the original order on any failure) and `_bfs_entity_traversal` (a multi-hop breadth-first walk
across `qortia_entities` via co-occurring `linked_memory_ids`, decaying the boost score at each
hop).

Depends on: `qortia.config`, `qortia.auth` (get_litellm_key, AgentIdentity), `qortia.common`
(LiteLLM client), `qortia.db`, `qortia.models`. Used by `qortia.recall`.

### reflect

`/v1/reflect` — consolidates an agent's recent episodic/experiential memories (last 7 days, up to
30) together with its existing consolidated mental_models/lessons into one LLM call that returns
CREATE/UPDATE/RETAIN actions (`_call_litellm_reflect`, `_build_reflect_prompt`). Writes are
guarded against the LLM wiping all consolidated knowledge — `_write_reflections` refuses to prune
existing rows if the LLM returns zero active IDs, or fewer than 50% of the existing consolidated
rows — and computes a stability score (cosine similarity between old and new embedding) for each
UPDATE. This module also defines three background-task functions that nothing currently starts as
a running process (see Known Limitations): `run_archival_task` (tiers old low-importance episodic
memories to `'archive'`, purges expired short_term memories), `run_embedding_worker` (claims
unembedded rows across four tables with `FOR UPDATE SKIP LOCKED`, embeds via LiteLLM, then
triggers `entity_graph._populate_graph_batch` and `links` cross-linking), and
`run_background_reflection_trigger`/`_trigger_idle_reflections` (finds agents idle past a
configured window and runs `_reflect_agent`, the non-HTTP twin of `/v1/reflect`).

Depends on: `qortia.config`, `qortia.auth`, `qortia.common`, `qortia.db`, `qortia.entity_graph`,
`qortia.models`, `qortia.recall_helpers` (_cosine), `qortia.knowledge.extract_entities_with_types`
(lazy), `qortia.links` (lazy), `qortia.remember` (`build_temporal_grounding_instruction`,
`_fetch_agent_clearance`).

```
POST /v1/reflect  (or the idle-trigger's _reflect_agent)
        │
        ▼
 fetch recent episodic/experiential (≤30, last 7 days)
 + existing consolidated mental_model/lesson rows
        │
        ▼
 LLM call → {"reflections": [CREATE | UPDATE | RETAIN, ...]}
 (_call_litellm_reflect / _build_reflect_prompt)
        │
        ▼
 embed CREATE/UPDATE content (before the write transaction)
        │
        ▼
 write transaction (_write_reflections):
   - prune_safe guard (refuse to wipe on empty/low-coverage response)
   - supersede old consolidated rows not returned as active
   - insert new consolidated rows + stability_score
   - decrement agent.reflection_counter
        │
        ▼
 ReflectResponse(memories_written, reflection_counter)
```

### remember

`/v1/remember` (private episodic/experiential/mental_model/decision/lesson/short_term memories,
with exact-content-hash dedup for episodic/experiential within 24h), `/v1/remember-org` (shared
handoff/process/decision_log — process and decision_log are role-gated to "chief" agents and
upserted by tenant+type+title), `/v1/forget` (ownership-checked delete that also cleans up
entity-graph links and `memory_links` rows), and `/v1/context` (bootstrap context bundle: org
chart, processes, recent handoffs, latest weekly summary, top mental_models/decisions/lessons).
Also owns the extraction-prompt helpers shared with `reflect.py`
(`build_temporal_grounding_instruction`, `ATTRIBUTION_INSTRUCTION`,
`NEGATIVE_EXTRACTION_INSTRUCTION`, `build_extraction_prompt`), language auto-detection
(`_detect_lang`, via `langdetect`, degrades to `"en"` if unavailable or on failure), and
`_fetch_agent_clearance` (used by `reflect.py`'s idle-reflection trigger, which only has raw
agent/tenant UUIDs and no `AgentIdentity`).

Depends on: `qortia.auth`, `qortia.common`, `qortia.db`, `qortia.knowledge`
(extract_entities_with_types), `qortia.models`. External: `langdetect` (optional).

### router

Top-level `APIRouter` that composes the four core sub-routers — remember, reflect, recall,
knowledge — into a single mount point included by `qortia.app`. No logic of its own.

Depends on: `qortia.remember`, `qortia.reflect`, `qortia.recall`, `qortia.knowledge`.

### telemetry

Best-effort OpenTelemetry counters — currently one, `qortia_recall_degraded`, incremented
whenever recall falls back to a degraded search path (e.g. an embedding call failure). Falls back
to a silent no-op counter if `opentelemetry-api` isn't installed or configured; every call site
already wraps use of these in `try/except Exception: pass`, so the no-op fallback is a correct
mode, not a degraded one.

Depends on: `opentelemetry` (optional, imported lazily inside `_make_counter`).

## Known Limitations

- **No HTTP provisioning/admin API.** The first API key for a fresh tenant can't be minted
  through the API itself (nothing to authenticate that request with), so tenant/agent/API-key
  creation is CLI-only (`qortia-admin`) or direct function calls into `qortia.provisioning` — see
  that module's docstring for the rationale.
- **Clearance levels are global, not per-tenant.** `qortia_clearance_levels` is a single
  process-wide lookup table (`external`/`internal`/`restricted`), simplified from an originally
  per-tenant-customizable design; per-tenant clearance levels are a clean future extension, not
  implemented today.
- **LiteLLM virtual keys are mapped by env, not Vault.** `get_litellm_key(tenant_id)` returns
  `QORTIA_LITELLM_TENANT_KEYS[tenant_id]` when set, else the shared `QORTIA_LITELLM_API_KEY`.
  Embed calls also send `user` + `metadata.qortia.tenant_id` for gateway OTel (ADR-003).
  Auto-minting virtual keys via the LiteLLM Admin API is a later increment.
- **Background workers run in a separate process.** Use `qortia-worker` / `just worker`
  alongside the API. The app lifespan does not start workers (keeps request latency clean).
  Without the worker, memories stay unembedded and archival/idle-reflect/weekly-summary do not
  run — see `docs/how-to/embeddings.md`.
- **Eval harnesses haven't been run end-to-end yet.** The scripts under `evals/` (`run_alb.py`,
  `run_reh.py`, `run_pib.py`, `run_temporal_eval.py`, `run_longitudinal_eval.py`,
  `run_longmemeval.py`, `run_extraction_eval.py`) need a live qortia server plus real LiteLLM
  connectivity to exercise `qortia.eval_router`'s endpoints, and haven't been exercised end-to-end
  against the standalone extraction yet. Dogfood path after `just stack-up` + `stack-pull-model`
  is the prerequisite.
