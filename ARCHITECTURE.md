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

A third router, `qortia.admin_router` (`/v1/admin/*`, ADR-004), mounts the same way `qortia.eval_router` does — conditionally, on an env var (`QORTIA_ADMIN_TOKEN` instead of `eval_mode`) — but doesn't join the diagram above at all: it never touches the core memory pipeline or `tenant_transaction`. It goes straight from `qortia.auth.require_admin` to `qortia.provisioning` to the Postgres identity tables (`qortia_tenants`/`qortia_agents`/`qortia_api_keys`, which carry no RLS policies). See the `admin_router` section below for its own diagram.

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
call site already degrades to an empty entity list). It always mounts `qortia.router.router`,
additionally mounts `qortia.eval_router.router` when `config.settings.eval_mode` is true, and
additionally mounts `qortia.admin_router.router` when `config.settings.qortia_admin_token` is set
(ADR-004) — both extra routers are inert by default and opt-in only.

Depends on: `qortia.config`, `qortia.common` (LiteLLM client init/close), `qortia.db` (pool
init/close), `qortia.router`, `qortia.eval_router`, `qortia.admin_router`,
`qortia.knowledge.load_spacy_model`. External: FastAPI, uvicorn.

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
`get_platform_embed_key` (shared/master key for probes), `provision_eval_litellm_key` (no-op),
`require_admin` (ADR-004 — a different, simpler mechanism for a different problem: gates
`qortia.admin_router` on a single static `QORTIA_ADMIN_TOKEN` compared with
`secrets.compare_digest`, not a per-tenant DB-backed key; 404s if the token is unset, 401 on a
missing/wrong `Authorization: Bearer` header).

Depends on: `qortia.config`, `qortia.db`. Used via `Depends(require_agent)` by `remember.py`,
`recall.py`, `reflect.py`, `knowledge.py`, and `eval_router.py`; `require_admin` is used by
`admin_router.py` via `APIRouter(dependencies=[Depends(require_admin)])`.

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

### admin_router

`/v1/admin/*` platform-admin provisioning over HTTP (ADR-004) — the same `create_tenant`/
`create_agent`/`issue_api_key` functions the `qortia-admin` CLI calls, reached by a caller with no
shell into wherever Qortia runs. Gated end-to-end by a static `QORTIA_ADMIN_TOKEN`:
`qortia.app` only mounts this router when the env var is set, and `qortia.auth.require_admin` —
applied once via `APIRouter(dependencies=...)`, not per-handler, since there's no identity payload
worth injecting — 404s outright when it's unset, mirroring `qortia.eval_router`'s
belt-and-suspenders posture for a different flag. `admin_create_agent` and `admin_issue_api_key`
both check the target tenant exists first (`SELECT 1 FROM qortia_tenants ...`) and 404 rather than
letting a foreign-key violation surface as a raw 500.

```
        POST /v1/admin/{tenants,agents,keys}
                        │
                        ▼
        qortia.auth.require_admin (router-level dependency)
        no QORTIA_ADMIN_TOKEN configured → 404 · wrong/missing Bearer → 401
                        │
                        ▼
        agents/keys only: SELECT 1 FROM qortia_tenants WHERE id = $1
        → 404 "Tenant not found" if missing
                        │
                        ▼
        qortia.provisioning.{create_tenant, create_agent, issue_api_key}
        (get_main_pool() directly — no RLS on the identity tables)
                        │
                        ▼
        200 {tenant_id} / {agent_id} / {api_key}
```

Public interface: `POST /tenants` (`{name}` → `{tenant_id}`), `POST /agents` (`{tenant_id, name,
clearance_level, division}` → `{agent_id}`), `POST /keys` (`{tenant_id}` → `{api_key}`) — all
under the `/v1/admin` prefix, all requiring `Authorization: Bearer <QORTIA_ADMIN_TOKEN>`, none
requiring `X-Agent-Id` (admin routes are platform-level, not tenant/agent-scoped).

Depends on: `qortia.auth.require_admin`, `qortia.db.get_main_pool`, `qortia.provisioning`
(`create_tenant`, `create_agent`, `issue_api_key`). Mounted by `qortia.app` when
`config.settings.qortia_admin_token` is set.

### common

Small shared runtime pieces used across the request path: the module-level `httpx.AsyncClient`
to LiteLLM (`init_litellm_client`/`close_litellm_client`/`get_litellm_client`), a back-compat
`EMBEDDING_MODEL` alias (prefer `qortia.embeddings.embedding_model()`), and
`assert_agent_active`, which every write/read endpoint calls first to 403 on a non-active agent.

Depends on: `qortia.config` (litellm_url / embedding_model), `httpx`, `asyncpg`, `fastapi`.

### chat

Single owner of LiteLLM `/chat/completions` calls, the same "one module owns this endpoint shape"
discipline `embeddings.py` already applies to `/embeddings` (ADR-002) — extracted after three call
sites (`recall_rerank._llm_rerank`, `entity_graph._update_entity_summary`,
`reflect._call_litellm_reflect`) turned out to each hand-roll the same request body, response
parsing (`choices[0].message.content`), and usage-token logging slightly differently, which is
exactly how `entity_graph.py`'s separate hardcoded model string went unnoticed (see `config`
below).

```
chat_completion(model, prompt, litellm_key, timeout, json_mode?, max_tokens?, log_event?)
        │
        ▼
 POST /chat/completions ──► 200? ──no──► raise ChatCompletionError("LiteLLM error: {status}")
        │ yes
        ▼
 choices[0].message.content present? ──no──► raise ChatCompletionError("malformed …")
        │ yes
        ▼
 log_event set? → log usage (prompt/completion tokens) ──► return content: str
```

What's *not* shared: failure policy. `chat_completion` only distinguishes "got content" from
"didn't" (`ChatCompletionError`) — each caller still decides whether that's swallow-and-fall-back
(`_llm_rerank`, `_update_entity_summary`: broad `except Exception`, return the pre-LLM value) or
propagate-as-HTTP-error (`reflect.reflect`'s explicit endpoint: catches `ChatCompletionError`,
re-raises `HTTPException(500, …)`). Consolidating that too would have meant picking one failure
policy for three call sites that genuinely need different ones — see the `config` section's note
on `rerank_model`'s per-consumer guards for why that split is deliberate, not an oversight.

Depends on: `qortia.auth` (`litellm_auth_headers`), `qortia.common` (LiteLLM client). Used by
`recall_rerank`, `entity_graph`, `reflect`. Do not POST `/chat/completions` from any other module —
call `chat_completion`.

### config

Env- and (now) file-driven runtime settings: the `Settings` dataclass, `load_settings()`, and the
module-level `settings` singleton. Includes database URL, LiteLLM URL/API key,
**`embedding_model` / `embedding_dimension`** (defaults `bge-m3` / `1024`, structurally pinned to
migrations/V1's `vector(1024)` — ADR-002), dedup similarity threshold, embedding cache size/TTL,
eval_mode, rerank model, reflection threshold, idle-reflection interval/window, and the
recall-tuning group below.

`rerank_model` is shared, unconfigured-by-default state read by three independent consumers —
`recall_rerank._llm_rerank`, `reflect.py`'s two `_call_litellm_reflect` callers, and
`entity_graph._update_entity_summary` (which had its own separately hardcoded
`"anthropic/claude-3-haiku-20240307"` string, disconnected from this setting entirely, until it
was found and routed through here) — each with its own empty-model guard skipping the call
instead of hitting a network round-trip guaranteed to fail. All three now go through
`qortia.chat.chat_completion` for the actual request/response mechanics (see `chat` above), but
each still owns its own guard and failure policy: `/v1/reflect` (an explicit, agent-authed
request) 503s rather than skipping silently, since there's a real caller here to tell, unlike the
automatic background trigger or best-effort entity-summary maintenance.
No vendor default is shipped for this setting — unlike `embedding_model`, nothing about it is
load-bearing. For a local Ollama-only stack (no LiteLLM gateway in front — see `QORTIA_LITELLM_URL`
in `.env.example`), a real chat model has to be pulled directly into Ollama; verified live against
real `/v1/reflect` data that `qwen2.5:0.5b`/`:1.5b` don't reliably follow the structured JSON
contract reflection needs (hallucinated actions, non-numeric fields) but `qwen2.5:3b` does.

Precedence: env var > optional TOML file (`QORTIA_CONFIG_FILE`, default `qortia.toml` in the
working directory, gitignored — `qortia.example.toml` is the committed template) > code default.
A missing file reproduces pure-env-var behaviour exactly; a malformed one logs a warning and is
treated as absent rather than crashing startup. Secrets and per-deployment topology
(`database_url`, `litellm_api_key`, `qortia_admin_token`, …) are deliberately not exposed to the
file layer — only tuning knobs are, so a config file can never become a second place a credential
leaks into git.

Recall tuning (`recall_rrf_k`, `recall_search_fetch_multiplier`,
`recall_{private,org,knowledge}_result_limit`, `recall_default_max_chars`, `[recall]` table in the
TOML file) used to be hardcoded module constants in `recall_helpers.py`, read once at import time.
Moved here — and read from `config.settings` at point of use in `recall_helpers`/`recall.py`, not
cached — while adding the response char budget `/v1/recall` never had at all: measured unbounded
at 38,961 chars/call average against a real 276-document corpus for 5.5% precision (agnova's
`evals/run_scale_eval_qortia.py`). `recall_default_max_chars` (8000) is what a caller gets when it
doesn't pass `max_chars` explicitly — the fix applies by default, not only to callers who know to
opt in; pass a non-positive `max_chars` explicitly to opt out.

Depends on: stdlib `os`/`dataclasses`/`tomllib` only — no internal imports, no new dependency
(`tomllib` is stdlib as of Python 3.11; this project requires ≥3.12).

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

Depends on: `qortia.config`, `qortia.auth.get_litellm_key`, `qortia.chat` (LLM completions),
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

### logging_config

One `configure()`, called by both entrypoints before any other qortia module's logger calls can
fire, so `app` and `workers` produce the same `<timestamp> <LEVEL> <message>` shape instead of
drifting independently — `workers.py` used to `logging.basicConfig(...)` on its own, `app.py`
configured nothing at all, and every `qortia.*` logger in the `app` process fell through to
Python's `logging.lastResort` (no formatter, hardcoded WARNING floor): INFO-level calls vanished
entirely, WARNING+ printed as a bare `str(dict)` with no timestamp/level/name, and none of it
matched `workers.py`'s own format. `configure()` also reformats uvicorn's own
`uvicorn`/`uvicorn.error`/`uvicorn.access` loggers in place — those are a separate system
(uvicorn's own `dictConfig`, `propagate=False`, already attached before the ASGI lifespan this
module's caller runs inside ever fires) that `basicConfig()` alone cannot reach — so uvicorn's
access-log lines carry the same shape as every other line in the stream rather than their own
`INFO:     <message>` format. `QORTIA_LOG_LEVEL` (default `INFO`) sets the floor. Idempotent, so
importing both entrypoints in one process (tests) doesn't double-attach handlers.

Depends on: stdlib `logging`/`os` only. Called from `qortia.app`'s `lifespan` (first line) and
`qortia.workers.main` (first line).

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

Tenant/agent/API-key provisioning functions, called from two places (ADR-004): the `qortia-admin`
CLI (`main()`, direct DB access, no HTTP at all — the only out-of-band path for the very first API
key a fresh tenant gets, since nothing can authenticate that request through a tenant-scoped API
yet) and `qortia.admin_router` (`/v1/admin/*`, gated by a separate platform-level
`QORTIA_ADMIN_TOKEN` — a different bootstrapping answer, for callers with no shell into wherever
Qortia runs). Core functions: `create_tenant`, `create_agent` (`name` is optional — added
alongside `admin_router`, also exposed as CLI `--name`), `issue_api_key` (returns the plaintext
key once — only its SHA-256 hash is persisted), `revoke_api_key` (CLI/direct-call only; not
exposed over HTTP — see Known Limitations). `main()` wraps `create_tenant`/`create_agent`/
`issue_api_key` in an `argparse` CLI (`create-tenant` / `create-agent` / `issue-key` subcommands).

Depends on: `qortia.config`, `qortia.auth.hash_api_key`, `asyncpg`. Used by `qortia.admin_router`
in addition to this module's own CLI.

### recall

`/v1/recall` — the hybrid search endpoint. Single-type queries (`type` = decision, lesson,
episodic, or short_term) take a fast single-strategy path (`_recall_decisions`,
`_recall_lessons`, `_recall_episodic`, `_recall_short_term`). Everything else runs
`_hybrid_recall_pipeline`: BM25 and vector search across the requested scopes (private/org/
knowledge) concurrently via `asyncio.gather`, an entity-graph adjacency boost plus 2-hop BFS
traversal (`qortia.recall_rerank._bfs_entity_traversal`), reciprocal-rank fusion of
private+org results and MMR diversification of knowledge candidates (`qortia.recall_helpers`),
cross-memory link expansion of the top results (`qortia.links`), and an optional LLM rerank
(`qortia.recall_rerank._llm_rerank`). After rerank, a char budget
(`recall_helpers._apply_char_budget`, `config.settings.recall_default_max_chars` unless the
caller passes `max_chars` explicitly) drops whole lowest-ranked results — never truncates one —
until the combined response fits; `/v1/recall` had no response-size cap of any kind before this
(see the `config` section above for the measurement that found it). Only *then* does the endpoint
fire-and-forget recall-count/access-time tracking and, if an `X-Work-Order-Id` header is present,
session-read logging for later outcome-based confidence decay (`_record_work_order_outcome`) — so
those side effects only touch results actually returned to the caller, not ones the budget dropped.

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
                  char budget: drop whole lowest-ranked results
                  (recall_helpers._apply_char_budget, default_max_chars)
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

Depends on: `qortia.config`, `qortia.auth` (get_litellm_key, AgentIdentity), `qortia.chat`
(LLM completions), `qortia.db`, `qortia.models`. Used by `qortia.recall`.

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

Depends on: `qortia.config`, `qortia.auth`, `qortia.chat` (LLM completions), `qortia.db`,
`qortia.entity_graph`, `qortia.models`, `qortia.recall_helpers` (_cosine),
`qortia.knowledge.extract_entities_with_types` (lazy), `qortia.links` (lazy), `qortia.remember`
(`build_temporal_grounding_instruction`, `_fetch_agent_clearance`).

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

- **HTTP admin provisioning is a single static token, not a revocable credential.**
  `qortia.admin_router` (`/v1/admin/*`, ADR-004) closes the original "no HTTP path for
  provisioning" gap for callers with no shell into wherever Qortia runs, but `QORTIA_ADMIN_TOKEN`
  is one process-wide secret — rotating it means changing the env var and restarting, and a leak
  grants tenant/agent/key creation across every tenant (not memory-data access; `admin_router`
  never touches `tenant_transaction`). `revoke_api_key` also still isn't exposed over HTTP, only
  CLI/direct-call. A DB-backed, per-caller, revocable admin-token scheme (and HTTP key revocation)
  are reasonable future increments, not needed for a single caller and not built here.
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
- **`xx_ent_wiki_sm` (Indic NER) was an unpinned dependency until 2026-08-11 — every memory in
  all five supported Indic languages silently got zero entities.** `knowledge._INDIC_MODEL`
  routes `hi`/`bn`/`ta`/`te`/`mr` to `spacy.load("xx_ent_wiki_sm")`, but nothing in
  `pyproject.toml` installed that model (only `en_core_web_sm` was pinned) — `spacy.load()` 404'd
  in every environment built from this repo, `app`'s best-effort startup load swallowed it
  (`spacy_model_load_failed`, logged at WARNING, not a boot failure — see the `app` section
  above), and `remember()`'s per-item `try/except` around `extract_entities_with_types` swallowed
  it again per request (`ner_extraction_failed`, also WARNING). No 5xx anywhere; `/v1/remember`
  returned 200 with a valid id every time. Caught empirically by scoring `remember()`-stored
  entities against WikiANN gold spans: 0/16 gold entities recovered on an initial Hindi sample.
  This is what the swallow-everywhere design in the paragraph above costs when the failure isn't
  transient: the architecture correctly treats NER as best-effort so a bad sentence or a slow
  model never fails a `remember()` call, but the same property means a total, permanent,
  environment-wide failure looks identical to an occasional one, both in the API response and in
  a WARNING-level log line that also carries routine traffic (`ner_lang_unsupported`, the designed
  fallback-to-English path for any language outside the five, was logged at the same WARNING level
  as this — fixed to `logger.info` under the clearer name `ner_lang_fallback_to_en`, see
  `knowledge.py`). Fixed by pinning `xx-ent-wiki-sm` in `pyproject.toml` alongside
  `en-core-web-sm`.
  **A second bug surfaced fixing the first:** `_get_indic_pipeline` cached the loaded spaCy
  pipeline by `lang`, but every one of the five Indic languages maps to the same
  `"xx_ent_wiki_sm"` model (`_INDIC_MODEL`) — so exercising all five in one process loaded and
  held five independent copies of one model, reproducibly OOM-killing the process on the fifth.
  Fixed by caching on model name instead; `qortia-app`'s RSS now stays flat (~410–440MiB
  measured) across all five languages instead of growing with each one.
  **Still open even with both fixed:** `xx_ent_wiki_sm` itself is a small, dated
  (WikiNER-trained) multilingual model, and per-language quality against a 300-example-per-language
  WikiANN sample is uneven and generally weak: recall 7–14% across all five Indic languages
  (Telugu weakest at 7.0%, Marathi best at 12.8%), against 66.5% for English (`en_core_web_sm`, a
  different model) and, unexpectedly, 71.8% for German routed through the same English fallback
  path — precision is markedly lower there (0.566 vs English's 0.858), so the higher recall isn't
  a signal the fallback path is *better*, more that it's more permissive. Whether the Indic
  numbers are fixed by a better model (e.g. an Indic-specific NER model) and whether the model
  should be operator-configurable (same pattern as `QORTIA_EMBEDDING_MODEL`) is open — the
  WikiANN comparison above is reusable against any candidate before committing to one.
