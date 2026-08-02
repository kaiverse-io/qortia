---
kind: architecture
status: active
owner: platform
last_reviewed: 2026-06-01
---

# Qortia — Memory Service (Consolidated Design)

**Status:** Core pipeline implemented — ADR-125 (causal tracking) and ADR-078 (bi-temporal filtering) shipped 2026-06-01
**Scope:** Complete memory architecture for the the platform AI workforce platform
**Last updated:** 2025-07-26

> This document is the single source of truth for Qortia's architecture, incorporating
> the original design spec, all competitive intelligence from Mem0/Memorizz/Graphiti
> reviews, shipped enhancements, and approved design revisions. Implementation plans
> (Parts 11–14) remain the build-sequence reference.

---

## 1. Identity

Qortia is the external memory service for all the platform agents. Agents are stateless
containers — they persist nothing locally. Every memory operation (read, write, search,
reflect) goes through Qortia's HTTP API.

**Qortia is not a database.** It is a memory interface — it decides what to query, how
to blend results, and returns ranked context ready for the LLM. No agent touches
Postgres, pgvector, or LiteLLM directly.

**The "Autonomous Employee" Philosophical Fork:**

Unlike "Shallow Agents" (e.g., Glean, Rovo) which treat agents as smart search bars that use tools to fetch data into context, the platform treats agents as **Autonomous Architects**.

| Dimension | Shallow Agent (e.g., Glean) | the platform AI Employee |
|---|---|---|
| **Cognition** | LLM is the primary analyst. | LLM is the architect/engineer. |
| **Strategy** | Context-stuffing (RAG). | Build-to-Solve (provisioning). |
| **Execution** | Constrained sandbox interpreter. | Full compute infrastructure (K8s). |
| **Output** | Transient response/plot. | Production code & persistent services. |

**Architectural positioning vs competitors:**

| Dimension | Qortia | Mem0 V3 | Memorizz | Graphiti |
|---|---|---|---|---|
| Deployment | Multi-tenant platform service | Embedded library | Embedded library | Library + graph DB |
| Isolation | RLS + JWT-bound identity | JSONB payload filter | None | `group_id` app-level |
| Reflection | Automated consolidation | None | None | Per-write edge invalidation |
| Importance | Formula-based `dynamic_importance()` | None | LLM call per memory | None |
| Delegation | Work orders (state machine) | None | None | None |

**Design principles (from `.amazonq/rules/agent-identity-rule.md`):**
- Deterministic over probabilistic — formula-based importance, spaCy NER, no LLM calls for infrastructure operations
- Database-enforced isolation — RLS on every tenant-scoped table, never application-level filtering alone
- Single database engine — PostgreSQL + pgvector handles vector, BM25, relational, and entity queries
- Crash safety by construction — supersede-first ordering, atomic transactions, no partial state

---

## 2. Three Memory Scopes

### 2.1 Private Memory (`hindsight_memories`)

Per-agent. Isolated by `(tenant_id, agent_id)`. RLS enforced at database layer with
both permissive (tenant) and restrictive (agent) policies.

Contains the 5-type memory hierarchy (§3).

### 2.2 Org Memory (`org_memory`)

Shared across all agents in a tenant. Write access is role-gated. **Read access is
RBAC-gated (ADR-080)** — agents only see rows their clearance level and division permit.

| Type | Writer | Semantics |
|---|---|---|
| `org_chart` | Platform only (on provision/delete) | Agent roster — always `min_clearance='external'`, `audience='{all}'` so every agent can see it |
| `process` | Chief agent only | How work gets done — upsert on `(tenant_id, type, title)` |
| `decision_log` | Chief agent only | Org-level decisions — upsert on `(tenant_id, type, title)` |
| `handoff` | Any agent (own work only) | What was completed and handed off — append-only |
| `weekly_summary` | Platform background task | Deterministic concatenation of handoffs — no LLM, zero tokens |

**RBAC axes (ADR-080):** Each row carries `min_clearance TEXT` and `audience TEXT[]`.
An agent reads a row only when both hold:
- `agent.clearance_order >= row.min_clearance_order` (hierarchical, inclusive)
- `agent.division = ANY(row.audience) OR 'all' = ANY(row.audience)` (compartment)

Default config: `external=1`, `internal=2`, `restricted=3`. All existing rows default
to `min_clearance='internal'`, `audience='{all}'`. Existing behaviour is preserved for
tenants that do not customise their RBAC config.

### 2.3 Org Knowledge (`org_knowledge`)

Shared document corpus. All agents can search. Chief agent only can ingest.

- Section-aware splitting by markdown headings — logical units, not fixed-token slices
- PageIndex fields (`index_summary`, `index_questions`, `index_entities`) computed
  synchronously at ingest via spaCy `en_core_web_sm` — zero LLM tokens, ~5-10ms/section
- Embedding computed on `index_summary` (not raw content) — async worker
- BM25 searches `index_tsv` (reasoning artifacts), not raw `content`
- Dedup via SHA-256 `content_hash` — identical sections copy embeddings, always recompute PageIndex

---

## 3. Memory Type Hierarchy

The 5-type hierarchy is Qortia's primary structural advantage over flat-fact frameworks
(Mem0, Memorizz). Each type has distinct write semantics, recall strategy, and importance weight.

| Type | Importance | Recall Strategy | Boot Context | Reflection Role |
|---|---|---|---|---|
| `episodic` | 0.3 | Temporal + BM25 fallback | Never — raw events excluded | Input to reflection |
| `experiential` | 0.6 | Hybrid (BM25 + vector + RRF) | Never — consolidated versions only | Input to reflection |
| `mental_model` | 0.8 | Hybrid (BM25 + vector + RRF) | Top 20 by importance (consolidated only) | Output of reflection |
| `decision` | 0.9 | BM25 + recency sort | Top 15 by recency (all rows) | Never reflected — already finished artifacts |
| `lesson` | 0.95 | Vector similarity only | Top 20 by importance (consolidated only) | Output of reflection |

**Importance is platform-assigned, never agent self-assessed.** This is a retrieval
signal for boot context ranking and recall scoring — not a vanity metric.

**Type-routed recall** (ADR-020): Each type routes to its optimal retrieval strategy.
Decisions use BM25+recency (latest decision on a topic, not most semantically similar).
Lessons use vector-only (experiential patterns match by meaning). This is structurally
superior to monolithic search — validated by the competitive review.

---

## 4. Data Model

### 4.1 Database Strategy

**Decision: Stay with PostgreSQL + pgvector.** (ADR-002)

| Alternative | Evaluated | Rejected Because |
|---|---|---|
| MongoDB | Memorizz uses it | No RLS — tenant isolation regresses to application-level. Adds second database. Zero capability gain over JSONB. |
| Neo4j / graph DB | Graphiti requires it | No RLS. GPL (Community) or $$$ (Enterprise). Heavy operational dependency. Neither Mem0 nor Memorizz validates graph traversal for core recall. |
| Apache AGE | Graphiti patterns | Reserved as future option if 2-hop entity traversal via CTEs becomes unmaintainable. Runs inside Postgres — preserves RLS boundary. |

pgvector handles all three recall signals in one engine:
- **Semantic:** HNSW index with `vector_cosine_ops`, DiskANN for larger-than-memory indexes (v0.8+)
- **BM25:** `tsvector` generated columns with GIN indexes, `ts_rank_cd` + `plainto_tsquery`
- **Entity:** JSONB `entities` column with GIN index for `?|` containment, `qortia_entities` table with HNSW for entity boost

### 4.2 Embedding Model

Single model for all languages: `BAAI/bge-m3` served via host ollama, aliased as
`text-embedding-3-small` in LiteLLM (ADR-081). Outputs 1024-dim vectors.

| Language | Model | Routing |
|---|---|---|
| All languages | `text-embedding-3-small` → `bge-m3` via LiteLLM | Single model |

`bge-m3` supports 100+ languages including all Indic languages (hi, ta, bn, te, kn,
ml, mr, gu, pa, or, as). No per-language routing. Startup dimension validation checks
the model returns 1024-dim vectors before any data is written (ADR-081).

`qortia_entities` uses the same model — `lang` column is absent on that table.

### 4.3 Core Tables

**`hindsight_memories`** — Private per-agent memory (append-only, never updated after write):
- RLS: permissive tenant policy + restrictive agent read policy (`qortia_platform` exempt)
- Key columns: `type`, `content`, `content_tsv` (generated), `embedding`, `importance`,
  `is_consolidated`, `entities` (JSONB), `recall_count`, `last_recalled_at`, `stability_score`,
  `tier` (active/archive), `expires_at` (short_term TTL), `valid_from`, `valid_until` (temporal bounds, ADR-078),
  `confidence_multiplier FLOAT NOT NULL DEFAULT 1.0` (outcome-driven decay, ADR-125)
- Superseded memories: `valid_until IS NOT NULL` — excluded from default recall (`valid_until IS NULL OR valid_until > now()`), queryable via `as_of`

**`org_memory`** — Shared tenant memory:
- RLS: `tenant_visibility_read` (two-axis RBAC: clearance order + division, ADR-080) + platform unrestricted write
- Columns: `min_clearance TEXT DEFAULT 'internal'`, `audience TEXT[] DEFAULT '{all}'`,
  `valid_from TIMESTAMPTZ`, `valid_until TIMESTAMPTZ` (ADR-078 — V28 migration),
  `confidence_multiplier FLOAT NOT NULL DEFAULT 1.0` (ADR-125 — V27 migration)
- `org_chart` rows hardcoded to `min_clearance='external'`, `audience='{all}'` — all agents can see the roster
- Unique indexes for upsertable types: `(tenant_id, type, title)` WHERE `type IN ('process', 'decision_log')`
- Unique index for org_chart: `(tenant_id, author_id)` WHERE `type = 'org_chart'`

**`qortia_session_reads`** — ADR-125 causal read log (V27 migration):
- `(work_order_id, memory_id, tenant_id, agent_id, recalled_at)` — one row per recalled memory per WO
- Fire-and-forget write on every `POST /v1/recall` that carries `X-Work-Order-Id` header
- RLS + FORCE ROW LEVEL SECURITY; never blocks recall latency

**`qortia_outcome_records`** — ADR-125 WO outcome log (V27 migration):
- `(work_order_id UNIQUE, outcome, memory_count, tenant_id, agent_id, recorded_at)`
- Written by `work_orders/router.py` on WO completion/failure; triggers `confidence_multiplier` update
- RLS + FORCE ROW LEVEL SECURITY

**`org_knowledge`** — Document corpus:
- RLS: `tenant_visibility_read` (same two-axis RBAC as org_memory, ADR-080) + platform unrestricted write
- New columns: `min_clearance TEXT DEFAULT 'internal'`, `audience TEXT[] DEFAULT '{all}'`
- PageIndex fields: `index_summary`, `index_questions`, `index_entities`, `index_tsv` (generated)
- Unique constraint: `(tenant_id, source_path, chunk_index)`

**`qortia_entities`** — Entity graph (Enhancement 3, Phase 1):
- `entity_text`, `entity_type`, `embedding`, `linked_memory_ids UUID[]`, `summary TEXT`
- `max_clearance_order INTEGER DEFAULT 2` — highest clearance order of any linked memory (ADR-080 G8 fix)
  prevents entity boost from leaking existence of restricted memories to lower-clearance agents
- HNSW index on embedding, GIN index on `linked_memory_ids`
- Tenant-scoped with optional `agent_id` (NULL = org-scoped)
- `summary`: running natural-language summary of the entity, maintained by `_populate_graph_batch`.
  NULL until first memory link. Bootstrapped from memory content on 1st link (no LLM).
  Updated via LLM on every 3rd link (count % 3 == 0). Surfaced in `RecallResult.entity_summary`.
- Population runs in the embedding worker (not the write path) to avoid write amplification

**`memory_history`** — Append-only audit trail:
- Operations: `remember`, `remember_org`, `forget`, `knowledge_ingest`, `knowledge_delete`, `reflect`
- Agent-initiated operations only — platform-internal writes exempt

**`memory_links`** — Cross-memory similarity links (Part 16i, ADR-077):
- Bidirectional links between `hindsight_memories` rows with cosine similarity >= 0.70
- Populated by the embedding worker after each successful `hindsight_memories` embedding write (top-3 per memory)
- No `agent_id` column — tenant isolation via RLS `tenant_isolation` policy; agent isolation inherited from `hindsight_memories`
- Traversed by `_expand_with_links()` in the recall pipeline to surface causal context (e.g. lesson → decision that caused it)
- Linked results carry `linked_via: str` in `RecallResult` for LLM provenance
- Cleaned up atomically in `forget()` — no dangling link rows

### 4.4 RLS Pattern

```python
@asynccontextmanager
async def tenant_transaction(
    pool: asyncpg.Pool,
    tenant_id: UUID,
    agent_id: UUID | None = None,
    memory_clearance_order: int | None = None,  # ADR-080
    agent_division: str | None = None,  # ADR-080
) -> AsyncGenerator[asyncpg.Connection, None]:
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.tenant_id = '{tenant_id}'")
            if agent_id:
                await conn.execute(f"SET LOCAL app.agent_id = '{agent_id}'")
            order = memory_clearance_order if memory_clearance_order is not None else 2
            division = agent_division or "all"
            await conn.execute(f"SET LOCAL app.memory_clearance_order = '{order}'")
            await conn.execute(f"SET LOCAL app.agent_division = '{division}'")
            yield conn
```

`SET LOCAL` is transaction-scoped. Safe with PgBouncer transaction pooling.
Every memory operation uses this — no raw `pool.acquire()` for tenant-scoped queries.
Background tasks pass `None` for both RBAC params — they run as `qortia_platform`
which bypasses `tenant_visibility_read` via the `platform_write` policy.

Authoritative DDL lives in `docs/platform/01-data-model.md`.

---

## 5. API Surface

All endpoints are called by `mcp_bridge.py` inside the agent container.
No agent ever touches Postgres or LiteLLM directly. Every agent-authenticated
endpoint checks `agent status = active` before executing (ADR-043).

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /v1/context` | Agent | Boot context assembly — called once per boot by `agent-start.sh` |
| `POST /v1/recall` | Agent | Mid-session hybrid search across all scopes |
| `POST /v1/remember` | Agent | Write private memory (batch, atomic) |
| `POST /v1/remember-org` | Agent | Write org memory (role-gated) |
| `POST /v1/forget` | Agent | Expire a private memory |
| `POST /v1/reflect` | Agent | Trigger reflection — LLM synthesis of consolidated memories |
| `POST /v1/knowledge` | Agent (chief) | Ingest document into knowledge corpus |
| `DELETE /v1/knowledge/{path}` | Agent (chief) | Remove all chunks for a source |

---

## 6. Write Pipelines

### 6.1 Private Memory Write (`POST /v1/remember`)

```json
{
  "memories": [
    {"type": "episodic", "content": "...", "source_task_id": "<uuid>", "metadata": {}}
  ]
}
```

**`importance` is NOT a request body field.** Assigned by platform per type at write time.

**Write path (single atomic transaction — ADR-028):**
1. Extract NER entities from each memory's content via spaCy (best-effort, ~1ms/memory)
2. INSERT all memories — `content_tsv` generated automatically, BM25 ready instantly
3. If batch contains episodic memories: atomically increment `reflection_counter`
4. All inserts + counter increment succeed or all roll back — no partial batch success
5. Append `memory_history` rows
6. `embedding = NULL` — background worker fills asynchronously

### 6.2 Org Memory Write (`POST /v1/remember-org`)

```json
{"type": "handoff | process | decision_log", "title": "...", "content": "..."}
```

- `process`, `decision_log`: chief-only, upsert on `(tenant_id, type, title)`
- `handoff`: any agent (own work only), append-only
- NER entity extraction at write time (same as private memory)

### 6.3 Knowledge Ingestion (`POST /v1/knowledge`)

Chief-only. Section-aware splitting. PageIndex computed synchronously via spaCy.

1. Split by markdown headings (`##`, `###`). Oversized sections split at paragraph boundaries.
2. Dedup: if same chunk count AND all `content_hash` values match → return early (no work)
3. Otherwise: delete existing chunks → insert new chunks with PageIndex fields
4. Per-section: `content_hash` (SHA-256), `index_summary` (first 2 sentences),
   `index_entities` (spaCy NER), `index_questions` (heading + noun chunks)
5. Hash-match dedup: identical content copies embedding only, always recomputes PageIndex (ADR-031)
6. Embedding worker picks up `embedding IS NULL` rows on next cycle

### 6.4 Forget (`POST /v1/forget`)

Marks a private memory as forgotten. Agent-scoped — can only forget own memories.
When entity graph (§10.1) ships: must strip the memory UUID from `qortia_entities.linked_memory_ids`.

---

## 7. Recall Pipeline

The most algorithmically complex endpoint. Type-routed search strategies, parallel
BM25 + vector, RRF fusion, entity boost, MMR dedup, optional LLM re-ranking.

### 7.1 Request

```json
{
  "query": "what did we decide about rate limiting",
  "scope": "private | org | knowledge | all",
  "type": "episodic | experiential | mental_model | decision | lesson | null",
  "rerank": false,
  "entities": ["AuthService", "Scout"]
}
```

- `type` filter applies to `hindsight_memories` only — org/knowledge never filtered by type ([recall/remember response schemas](qortia/04-api-contracts.md#post-v1recall--authoritative-schema))
- `entities` narrows via `entities ?| $N::text[]` on private/org tables; knowledge uses `index_tsv` implicitly
- `agent_id` and `tenant_id` from JWT — never in request body

### 7.2 Type-Routed Search Strategies (ADR-020)

| Type | Strategy | Rationale |
|---|---|---|
| `decision` | BM25 + recency sort | Latest decision on a topic, not most semantically similar |
| `lesson` | Vector similarity only | Experiential patterns match by meaning better than keyword |
| `episodic` | Temporal range (7 days) + BM25 fallback | Recency matters most for episodic |
| `mental_model`, `experiential`, `null` | Full hybrid pipeline | Maximum recall quality |

### 7.3 Full Hybrid Pipeline

When no type filter or `mental_model`/`experiential`:

```
Query → embed (LiteLLM, graceful degradation to BM25-only on failure)
      → parallel searches based on scope:
          BM25 + Vector on hindsight_memories (private)
          BM25 + Vector on org_memory (shared)
          BM25 + Vector on org_knowledge.index_tsv / embedding (corpus)
      → RRF fusion on memory results × dynamic_importance
      → Entity boost from qortia_entities (3rd signal — §10.1)
      → Entity summary attached to top result (entity_summary field on RecallResult)
      → Cross-memory link expansion: top-5 results expanded with linked memories (linked_via set, ADR-077)
      → Temporal filter: valid_until IS NULL (default) or as_of range (ADR-078)
      → Keyword boost on knowledge candidates: _score *= (1 + token_overlap(query, content))
      → MMR dedup on knowledge results (lambda=0.5, dedup_threshold=0.85, min_score=0.30)
      → LLM re-rank if total results < 3 OR rerank=true (ADR-020)
      → fire-and-forget access tracking (recall_count++, last_recalled_at=now())
      → merge memory + knowledge results, return
```

**Graceful degradation:** If `_embed_query` fails (LiteLLM unavailable), fall back to
BM25-only across all scopes. Never return 500 due to LiteLLM being down ([two-scope memory](qortia/04-api-contracts.md#two-scope-memory)).

**Re-rank failure is non-fatal:** `_llm_rerank` catches all exceptions and returns
original order. Uses agent's configured model — never free-worker (confidential content).

### 7.4 Dynamic Importance (ADR-055 + ADR-125)

```python
def dynamic_importance(
    recall_count: int,
    last_recalled_at: datetime | None,
    base_importance: float,
    confidence_multiplier: float = 1.0,  # ADR-125: outcome-driven scaling
) -> float:
    frequency_boost = log1p(recall_count) / 10.0  # log scale, ~0.3 at 1000 recalls
    recency_boost = linear_decay(last_recalled_at, 30)  # 0→0.2 over 30 days
    raw = min(1.0, base_importance + frequency_boost + recency_boost)
    return max(0.0, min(1.0, raw * confidence_multiplier))  # outcome scaling last
```

`confidence_multiplier` is stored on each `hindsight_memories` and `org_memory` row (default 1.0).
Updated by `_record_work_order_outcome()` when a work order completes/fails:
- `SUCCESS × 1.05` (cap 1.0) · `MINOR_FAILURE × 0.85` · `CRITICAL_FAILURE × 0.60` (floor 0.10)

Applied as multiplier in RRF fusion: `final_score = rrf_score × dynamic_importance`.
Access tracking is fire-and-forget after response assembly (separate connection, non-blocking).

### 7.5 RRF Fusion

```
RRF_K = 60  (standard constant)
score(result) = Σ 1/(RRF_K + rank_in_list) across all lists containing the result
final_score = score × dynamic_importance
```

Groups results by ID, sums RRF scores across BM25 and vector lists, re-ranks by
`final_score`. Principled fusion — score-magnitude-invariant, well-studied.

### 7.6 MMR for Knowledge Corpus

Knowledge results use Maximal Marginal Relevance to balance relevance with diversity:
- `min_score: 0.30` — candidates below this excluded before MMR (lower than memory
  candidates because paraphrased queries produce lower raw cosine scores against
  knowledge chunks than against episodic memories — see ADR-074)
- `lambda: 0.5` — equal weight relevance vs diversity
- `dedup_threshold: 0.85` — near-duplicate sections dropped
- Selects top 4 results after diversity filtering

**Keyword boost (ADR-074):** Before MMR, each knowledge candidate's `_score` is
multiplied by `(1.0 + _keyword_boost(query, content))`. `_keyword_boost` returns
the fraction of unique query tokens (> 2 chars, case-insensitive) found in the
content. This lifts candidates that share surface tokens with the query, preventing
paraphrased queries from burying the correct knowledge chunk below the `min_score`
threshold. Applied to knowledge candidates only — episodic/org candidates use RRF
fusion with `dynamic_importance` as their scoring signal.

### 7.7 Entity Filter

When `entities` present in request: `AND entities ?| $N::text[]` on private/org queries.
GIN-indexed JSONB containment. Knowledge entity filtering via BM25 on `index_tsv` (which
includes `index_entities`). Helper function `_entity_filter_clause` shifts parameter
numbers to avoid positional collisions.

**SQL injection defence:** Type filter uses parameterised clause with whitelist
validation in `_type_filter_clause` — independent of Pydantic enum validation.
Defence-in-depth, not redundancy.

---

## 8. Boot Context Assembly

Called once at container start by `agent-start.sh`. Assembles the system prompt
context package. No token budget — all data is included.

Assembly order (fixed):
1. **Org chart:** Who the other agents are (`type = 'org_chart'`)
2. **Org processes:** How work gets done (`type = 'process'`)
3. **Recent handoffs:** What the team has been working on (top 5 by recency)
4. **Weekly summary:** Synthesised view of the past week (top 1 by recency)
5. **Private consolidated memories:** The agent's synthesised knowledge
   - `mental_model`: top 20 by importance DESC (WHERE `is_consolidated = true`)
   - `decision`: top 15 by recency DESC (all rows, no `is_consolidated` filter)
   - `lesson`: top 20 by importance DESC (WHERE `is_consolidated = true`)

**Why `is_consolidated` matters:** Raw episodic and experiential memories are never
in boot context. They must be synthesised by reflection first. Decisions are the
exception — they are point-in-time records and do not require reflection.

---

## 9. Reflection Pipeline (Agent-Driven)

Agents run their own reflection. No platform cron. No coordination.
Trigger: after every 10 new episodic memories written since last reflection (`REFLECTION_THRESHOLD = 10`).

### 9.1 Execution Flow in `mcp_bridge.py`

Reflection is triggered automatically inside `mcp_bridge.py` and runs as a non-blocking
background task. The `remember()` call returns immediately to the LLM.

**In-process state:**
- `_episodic_counter`: Syncs from Postgres at boot. Increments on confirmed writes.
- `_reflection_in_progress`: Mutex lock preventing concurrent reflection tasks.

**Flow:**
1. `_episodic_counter` hits 10 (and `_reflection_in_progress` is False).
2. `asyncio.create_task(_run_reflect())` fires.
3. Lock is set: `_reflection_in_progress = True`.
4. Calls `POST /v1/reflect` via HTTP client with a strict 120s timeout.
5. **On success:** `_episodic_counter` synced directly from the HTTP response JSON.
6. **On failure or timeout:** 
   - HTTP exception is swallowed locally to prevent crashing the agent container.
   - Structured log emitted: `{"event": "reflection_failed", "error": "..."}`.
   - Transaction rolled back server-side.
7. **Finally:** `_reflection_in_progress = False` (retries on next trigger).

### 9.2 The Transaction (Crash Safety)

The entire DB operation runs in a single transaction with **supersede-first ordering** (ADR-027).

1. Fetch recent episodic/experiential + ALL existing consolidated memories
2. Call LiteLLM with agent's model to synthesise new mental models and lessons
3. Transaction starts:
   - **Supersede (FIRST):** Set `is_consolidated = false` and `valid_until = now()` on ALL previously consolidated rows (ADR-027, ADR-078)
   - **Write:** Insert new memories from LLM with `is_consolidated = true`
   - **Decrement:** `reflection_counter = GREATEST(reflection_counter - 10, 0)`
   - **Audit:** Append to `memory_history`
4. Commit

**Why supersede first?** If the process crashes after (1) but before (2), Postgres
rolls back. The old consolidated set remains intact and `is_consolidated = true`.
If we wrote first and crashed before superseding, the agent would have two concurrent
consolidated sets, leading to duplicate/contradictory boot context.

---

---

## 10. Weekly Org Summary (Platform Task)

Runs weekly per tenant as an asyncio background task inside the FastAPI monolith.
No `pg_cron`. No K8s CronJob. Works in Docker Compose and K8s identically.

**Execution:**
- Staggered by `hash(tenant_id) % 7` to prevent all tenants firing simultaneously.
- Restart-safe: tracked via `qortia_tenants.weekly_summary_last_run_at`.
- Multi-replica safe: uses `SELECT id FROM qortia_tenants ... FOR UPDATE SKIP LOCKED`.
- If another replica holds the lock for a tenant, the current replica skips it.

**Logic:**
- Fetches all `handoff` records from the past 7 days. If < 3, skips (not enough to summarize).
- Deterministic concatenation: `[{agent_name} | {date}]\n{content}`.
- Zero LLM calls. Zero token cost.
- Writes to `org_memory` as `type: weekly_summary`.

---

## 11. What Replaces What From the agent harness

| Current (the agent harness native) | Qortia replacement |
|---|---|
| `<workspace>/.openclaw/memory/main.sqlite` | `hindsight_memories` + `org_knowledge` tables |
| `chunks` table (raw text + blob embedding) | `org_knowledge.content` + `embedding` |
| `chunks_fts` (SQLite FTS5) | `content_tsv` (Postgres tsvector, GIN index) |
| `chunks_vec` (sqlite-vec ANN) | `embedding` (pgvector HNSW) |
| `embedding_cache` (dedup by content hash) | `org_knowledge.content_hash` dedup |
| bge-m3 via Ollama (local) | `text-embedding-3-small` via LiteLLM |
| Workspace files (`bank/`, `research/`) | `org_knowledge` via `POST /v1/knowledge` |
| Ollama dependency (local infra) | LiteLLM gateway (cloud-native, fallback chain) |

---

## 12. What Qortia Does Not Do

- **Does not store conversation history:** the agent harness manages that in-process.
- **Does not chunk agent memories:** `hindsight_memories` entries are atomic units written by the agent.
- **Does not run reflection:** Agents trigger their own reflection via `mcp_bridge.py`.
- **Does not manage agent identity:** Handled by the auth service + Vault.
- **Does not handle tool credentials:** Handled by Vault.

---

## 13. Streaming Hindsight & Replayability (Workforce OS)

In the the platform Workforce OS, memory is not just a static record—it is a **durable stream of thought**. This architecture enables "immortal" agents through event-driven state recovery.

### 12.1 The Problem: The "Amnesia" Trap
If an agent container crashes or a K8s pod is evicted mid-task, the agent loses its **Episodic Memory** (the context of what it was just doing). Traditional RAG cannot recover this because the memories weren't "consolidated" or even "written" yet.

### 12.2 The Solution: Event-Stream Persistence
Qortia treats the `memory_history` and `work_orders` tables as an append-only log (The "Workforce Bus"). 
- **Granular Commits:** Agents are instructed to emit "Thought Events" to the platform mid-task.
- **State Re-hydration:** When a replacement agent is provisioned to take over an orphaned Work Order, the platform fetches the last 30 minutes of "Thought Events" from the stream.
- **Seamless Recovery:** The new agent "replays" these events into its local context, allowing it to "remember" the exact reasoning steps of its predecessor.

### 12.3 Benefits for the AI Employee
- **Resilience:** AI Employees can survive infrastructure failures without losing progress.
- **Auditability:** Humans can "rewind" the stream to see the exact moment a reasoning error occurred.
- **Observability:** Analysts can "peek" at the Compute Agent's internal stream in real-time via the Blackboard pattern.

---

## 14. Enhancement Roadmap

Post-MVP Qortia enhancements are tracked in [`docs/generated/enhancement-index.md`](../generated/enhancement-index.md).
Filter by `owner: platform` and `status: post-mvp` for the deferred backlog.

---
**End of Document**
