---
kind: architecture
status: active
owner: platform
last_reviewed: 2026-05-30
source: migrated from design-clarity-monolith.md Q10–Q97 (qortia-scoped)
---

# Qortia API Contracts

Authoritative request/response schemas, behavioral invariants, and data model decisions for
all Qortia endpoints (recall, remember, reflect, forget, knowledge, org memory). Migrated
from the design-clarity-monolith. This document supersedes all earlier Qortia design notes.

---

## Memory Model

### Two-Scope Memory

| Scope | Table | Who writes | Who reads |
|---|---|---|---|
| `private` | `hindsight_memories` | The agent itself | The agent itself |
| `org` | `org_memory` | Any agent (type-gated) | All agents in tenant |
| `knowledge` | `org_knowledge` | Chief only (via POST /v1/knowledge) | All agents in tenant |

**hindsight_memories types:** `episodic`, `experiential`, `mental_model`, `decision`, `lesson`
**org_memory types:** `org_chart`, `process`, `decision_log`, `handoff`, `weekly_summary`

`org_chart` and `weekly_summary` are platform-internal — no agent endpoint writes them.

### All Qortia Endpoints: Active Agent Check

All agent-authenticated Qortia endpoints check `agent status = active` before executing.

**Affected endpoints:** `GET /v1/context`, `POST /v1/recall`, `POST /v1/remember`,
`POST /v1/remember-org`, `POST /v1/reflect`, `POST /v1/forget`,
`POST /v1/knowledge`, `DELETE /v1/knowledge/{source_path}`

**Check:**
```python
agent = await conn.fetchrow(
    "SELECT status FROM qortia_agents WHERE id = $1 AND tenant_id = $2", agent_id, tenant_id)
if agent is None or agent["status"] != "active":
    raise HTTPException(status_code=403, detail="Agent is not active")
```

**Rationale:** deletion flow sets `status = inactive` at step 1 before tearing down K8s.
Without this check, a deleted agent can write org memory during the window (step 1–step 3).
`handoff`/`process`/`decision_log` org memory writes survive agent deletion via `ON DELETE SET NULL`
on `author_id` — producing orphaned rows never intentionally authored.

### GET /v1/context — Auth Contract

**Auth:** agent only. User auth returns 403.

No path parameter. `agent_id` and `tenant_id` derived exclusively from validated JWT.
An agent can only fetch its own boot context — no mechanism to request another agent's context.
Called once per boot by `agent-start.sh`. Never called mid-session.

---

## A2A Tracing Header

`POST /v1/recall` and `POST /v1/remember` accept an optional `X-Work-Order-Id` header.
When present, the platform includes `work_order_id` in the `recall_executed` and
`remember_written` structured log events, enabling end-to-end correlation of Qortia
operations back to the originating work order in the observability pipeline.

`mcp_bridge.py` injects this header automatically via `_the platformAuth.auth_flow()` whenever
`_active_work_order_id` is set — agents do not need to pass it manually.

---

## Recall

### POST /v1/recall — Authoritative Schema

**Auth:** agent only. User auth returns 403.

**Request body:**
```json
{
  "query":    "<string>",                // required — non-empty, drives ranking
  "scope":    "private|org|knowledge|all", // optional — default: "all"
  "type":     "<string>",               // optional — episodic|experiential|mental_model|decision|lesson
  "rerank":   true,                     // optional — bool, default false
  "entities": ["<string>", ...]         // optional — entity filter list
}
```

| Field | Required | Default | Validation |
|---|---|---|---|
| `query` | yes | — | non-empty string — 422 if missing or empty |
| `scope` | no | `all` | one of `private\|org\|knowledge\|all` — 422 if invalid |
| `type` | no | null | one of `episodic\|experiential\|mental_model\|decision\|lesson` — 422 if invalid. Org memory types are not valid filter values. |
| `rerank` | no | `false` | forces LLM re-rank regardless of result count. When false, auto-triggers only when total results < 3. |
| `entities` | no | null | list of non-empty strings — 422 if any element is empty. `[]` treated as null (no filter). |

**Interaction rules:**
- `query` always required — entities-only recall is invalid (422)
- `type` applies only to `hindsight_memories`. `org` and `knowledge` scopes searched without type filter.
- `scope: knowledge` + `type: <any>` → type filter ignored (`org_knowledge` has no type field)
- `entities` filters `hindsight_memories` and `org_memory` only; `org_knowledge` entity filtering handled implicitly by BM25 on `index_tsv`

**Response 200:**
```json
{
  "results": [
    {
      "id":         "<uuid>",
      "type":       "episodic|experiential|mental_model|decision|lesson|knowledge",
      "scope":      "private|org|knowledge",
      "content":    "<string>",
      "importance": 0.85,
      "created_at": "<iso8601>"
    }
  ]
}
```

Flat ranked list — results from all scopes merged and ranked by RRF score × dynamic_importance.
`importance` in response is the static base value — dynamic_importance is a ranking signal only.
`recall_count` and `last_recalled_at` are internal signals — never returned.
Empty `results` array is valid.

### org_knowledge Recall: MMR Parameters

org_knowledge uses MMR (Maximal Marginal Relevance) to reduce redundancy in knowledge results.

MMR balances relevance vs diversity:
```python
mmr_score = lambda_param * similarity - (1 - lambda_param) * max_similarity_to_selected
```

**Authoritative parameters (from `recall_helpers.py::_mmr`):**
- `lambda_ = 0.5` — equal weight on relevance and diversity
- Candidate pool: all embedded knowledge candidates before MMR re-ranking
- Final selection: top-k from MMR (k = caller's limit, default 10)

MMR runs on `org_knowledge` only. `hindsight_memories` and `org_memory` use RRF fusion.

### NER Entity Extraction

spaCy `en_core_web_sm` extracts named entities at write time for `POST /v1/remember` and
`POST /v1/remember-org`. Stored in `entities JSONB` column on both tables.

**Entity labels extracted:**
`ORG`, `PERSON`, `PRODUCT`, `GPE`, `NORP`, `FAC`, `WORK_OF_ART` — capped at 20 per row. Deduplicated.

**Shared utility (`qortia/knowledge.py`):**
```python
ENTITY_LABELS = frozenset({"ORG", "PERSON", "PRODUCT", "GPE", "NORP", "FAC", "WORK_OF_ART"})

def extract_entities(text: str) -> list[str]:
    doc = get_nlp()(text)
    return list(dict.fromkeys(
        ent.text for ent in doc.ents if ent.label_ in ENTITY_LABELS
    ))[:20]
```

`get_nlp()` returns the already-loaded model instance (loaded once at platform startup step 4).

**Best-effort:** if spaCy raises on malformed input, platform logs warning and writes `entities=[]`. Write never fails due to NER failure.

**Entity filter SQL (when `entities` present in recall request):**
```sql
AND entities ?| $N::text[]   -- JSONB ?| operator: true if column contains at least one key
```
GIN indexes (`idx_hindsight_entities`, `idx_org_memory_entities`) support this efficiently.

**Schema additions:**
```sql
entities JSONB NOT NULL DEFAULT '[]'
CREATE INDEX idx_hindsight_entities ON hindsight_memories USING GIN (entities);
CREATE INDEX idx_org_memory_entities ON org_memory USING GIN (entities);
```

**Migration:** `migrations/V2__ner_entities.sql`

**Platform-internal org_memory writes:**
- `org_chart`: entities extracted from formatted org chart string
- `weekly_summary`: `entities='[]'` — summary is a concatenation; individual handoffs already carry entities

### Dynamic Importance Scoring

Static `importance` (assigned at write time by memory type) is augmented with a dynamic signal
computed from access frequency and recency. Replaces static lookup in RRF fusion.

**Columns added to `hindsight_memories` and `org_memory`:**
```sql
recall_count          SMALLINT NOT NULL DEFAULT 0   -- SMALLINT sufficient (max 32767)
last_recalled_at      TIMESTAMPTZ
confidence_multiplier FLOAT    NOT NULL DEFAULT 1.0  -- ADR-125: outcome-driven decay
```

**Access tracking — fire-and-forget after recall (`_record_recall_access`):**
```python
async def _record_recall_access(table: str, row_ids: list[UUID]) -> None:
    async with main_pool.acquire() as conn:
        await conn.execute(f"""
            UPDATE {table}
            SET recall_count = recall_count + 1, last_recalled_at = now()
            WHERE id = ANY($1::uuid[])
        """, row_ids)
```

**ADR-125 causal read logging — fire-and-forget when `X-Work-Order-Id` header present (`_log_session_reads`):**
Inserts one row per recalled memory into `qortia_session_reads(work_order_id, memory_id, ...)`.
Used by the WO outcome recorder to identify which memories were implicated in a succeeded/failed WO.

**ADR-125 outcome recording — triggered by `work_orders/router.py` on WO completion (`_record_work_order_outcome`):**
Queries `qortia_session_reads` for the WO, applies `confidence_multiplier` decay to all implicated
`hindsight_memories` rows, inserts one `qortia_outcome_records` row.
Decay: `SUCCESS × 1.05` (cap 1.0) · `MINOR_FAILURE × 0.85` · `CRITICAL_FAILURE × 0.60` (floor 0.10).

**dynamic_importance formula (from `recall_helpers.py`):**
```python
def dynamic_importance(
    recall_count: int,
    last_recalled_at: datetime | None,
    base_importance: float,
    confidence_multiplier: float = 1.0,   # ADR-125: outcome-driven scaling
) -> float:
    frequency_boost = math.log1p(recall_count) / 10.0
    recency_boost = 0.0
    if last_recalled_at:
        days_since = (datetime.now(timezone.utc) - last_recalled_at).days
        recency_boost = max(0.0, 1.0 - (days_since / 30.0)) * 0.2
    raw = min(1.0, base_importance + frequency_boost + recency_boost)
    return max(0.0, min(1.0, raw * confidence_multiplier))   # outcome scaling last
```

`confidence_multiplier` is read from the DB column, never exposed in the API response
(stored as `RecallResult._confidence_multiplier` private attr).

**Wired into `_rrf_fuse`:** `final_score` replaces static `importance` with
`dynamic_importance(r._recall_count, r._last_recalled_at, r.importance, r._confidence_multiplier)`.

**No effect on boot context assembly:** `GET /v1/context` uses fixed `ORDER BY importance DESC` /
`ORDER BY created_at DESC`. Dynamic importance is recall-time only.

**Migration:** `migrations/V3__recall_tracking.sql` — ALTER TABLE statements across memory tables.
V27 adds `confidence_multiplier` column to `hindsight_memories` and `org_memory`, and creates
`qortia_session_reads` and `qortia_outcome_records` tables.

---

## Remember

### POST /v1/remember — Field Validation

**Auth:** agent only.

**Field table:**

| Field | Required | Type | Validation |
|---|---|---|---|
| `type` | yes | string enum | `episodic\|experiential\|mental_model\|decision\|lesson` — 422 otherwise |
| `content` | yes | string | non-empty — 422 if missing or empty |
| `source_task_id` | no | UUID string | valid UUID format if present — 422 if malformed. No FK check. |
| `metadata` | no | JSON object | must be object type if present — 422 if array or scalar |

**memories array:** must be non-empty — 422 if empty. No upper bound enforced at API layer.

**Response 200:**
```json
{"ids": ["<uuid>", "<uuid>"]}
```
One UUID per memory written. `mcp_bridge.py` returns `f"Remembered {len(ids)} memories."` to LLM.

### remember() Batch Atomicity

One transaction for the entire batch. All inserts + counter increment execute atomically.

```sql
-- All memory inserts (one per memory in batch)
INSERT INTO hindsight_memories (...) VALUES (...);
-- ...

-- Single counter increment (episodic count only, once per batch)
UPDATE qortia_agents
SET reflection_counter = reflection_counter + $episodic_count, updated_at = now()
WHERE id = $agent_id;
```

All succeed or all roll back. No partial batch success.
In-process `_episodic_counter` incremented by `episodic_count` only after 200 — never optimistically.

### POST /v1/remember-org — Write Semantics

**Auth:** agent only. Valid types: `handoff | process | decision_log`. `org_chart` and `weekly_summary` are platform-internal — return 422 if submitted.

**Write semantics by type:**

| Type | Semantics |
|---|---|
| `process` | Upsert on `(tenant_id, type, title)` — replaces existing row with same title |
| `decision_log` | Upsert on `(tenant_id, type, title)` — replaces existing row with same title |
| `handoff` | Append always — each handoff is a distinct historical event |

**Rationale for upsert:** boot context loads all `process` and `decision_log` rows for the tenant.
Multiple rows with the same title would load stale versions alongside current — correctness problem.

**Schema requirement:** unique partial index `org_memory_upsertable_per_title` on
`(tenant_id, type, title) WHERE type IN ('process', 'decision_log')`.

**memory_history on upsert:** one row appended per call regardless of insert vs update.
`target_id` = existing UUID on update, new UUID on insert.

**Response 200:**
```json
{"id": "<uuid>"}
```
`mcp_bridge.py` returns `f"Org memory written: {id}"` to LLM.

### POST /v1/remember-org — Validation Order

1. **Enum check first:** if `type` not in `{handoff, process, decision_log}` → 422
2. **Role check second:** if type requires chief and caller is not chief → 403

422 fires before role enforcement — no information about role requirements leaked to invalid type submitters.

---

## Reflect

### Reflection: Agent-Driven

**Trigger:** after every 10 new episodic memories since last reflection.
**Counter:** `qortia_agents.reflection_counter` (Postgres). Tracked in-process in `mcp_bridge.py`.

**Counter invariant:** Postgres always incremented on every episodic write (inside the remember transaction).

**Full flow:**
1. `POST /v1/remember` (atomic: insert rows + increment counter) → on 200: `_episodic_counter += episodic_count`
2. At 10: `asyncio.create_task(_run_reflect())` — non-blocking
3. Platform fetches recent episodic + experiential memories (last 7 days, up to 30 rows)
4. Platform fetches ALL existing consolidated mental models + lessons (`is_consolidated = true`)
5. `POST /v1/reflect` — platform calls LiteLLM with agent's configured model. LLM returns strict JSON: `{memories: [{type, content, importance}]}`. If malformed or empty: abort, return 500, retry on next trigger.
6. Platform executes single transaction **in supersede-first order:**
   a. Flip ALL previously consolidated mental models + lessons to `is_consolidated = false` (supersede FIRST — crash here leaves old set intact)
   b. Write new mental models + lessons with `is_consolidated = true`
   c. Decrement `reflection_counter`: `GREATEST(counter - REFLECTION_THRESHOLD, 0)`
   d. Append one `memory_history` row per new memory written (`operation = 'reflect'`)
7. Response includes `{memories_written, reflection_counter}`. In-process `_episodic_counter` set to `reflection_counter` from response.

**Decrement, not reset:** episodic memories written during reflection are preserved.
If 3 arrived during reflection: Postgres holds 13, reflect returns `reflection_counter: 3`, in-process becomes 3.

**Model:** agent's configured model from `domain_md.model`. Never downgraded.
**Result:** consolidated set = fresh synthesis of everything learned. Boot context capped at
20 mental models + 15 decisions + 20 lessons. Raw memories accumulate unlimited but never bloat boot context.

### reflect() Return Values

```
"Reflection complete."                          — POST /v1/reflect returned 200
"Reflection already in progress. No action taken."  — _reflection_in_progress was true; HTTP call skipped
```

Auto-trigger path (`asyncio.create_task`) calls `_run_reflect()` directly and discards return.
The `_reflection_in_progress` guard still protects against race between auto-trigger and manual call.

### _run_reflect() Failure Observability

```python
async def _run_reflect() -> None:
    global _episodic_counter, _reflection_in_progress
    if _reflection_in_progress:
        return
    _reflection_in_progress = True
    try:
        result = await asyncio.wait_for(
            _http_client.post("/v1/reflect", json={}, timeout=120.0), timeout=125.0)
        result.raise_for_status()
        _episodic_counter = result.json()["reflection_counter"]
    except Exception as exc:
        logger.error({
            "event": "reflection_failed", "agent_id": AGENT_ID,
            "tenant_id": TENANT_ID, "error": str(exc),
        })
        # Transaction never committed — neither side changed. Retries on next trigger.
    finally:
        _reflection_in_progress = False
```

**Alertmanager threshold:** same `agent_id` emits `reflection_failed` >3 times in 1 hour.

---

## Forget

### POST /v1/forget

Permanently deletes a single memory row. Agent-initiated only.

**Auth:** agent only.

**Request body:**
```json
{"id": "<uuid>"}
```

**Platform resolution logic:**
```
1. Query both tables by id within tenant (RLS scopes automatically):
   SELECT id, agent_id, author_id, type FROM hindsight_memories WHERE id = $1
   SELECT id, author_id, type         FROM org_memory           WHERE id = $1
   → 404 if not found in either

2. Authorization by table and type:
   hindsight_memories:
     agent_id must match calling agent → 403 if not

   org_memory:
     type = handoff:      author_id must match calling agent → 403 if not
     type = process:      calling agent must be chief → 403 if not
     type = decision_log: calling agent must be chief → 403 if not
     type = org_chart:    forbidden for all agents → 403 always
     type = weekly_summary: forbidden for all agents → 403 always

   org_knowledge: not reachable via forget.
     Use DELETE /v1/knowledge/{source_path} instead.

3. Compute content_hash of row content before delete (for audit)
4. DELETE the row
5. INSERT INTO memory_history:
   operation: forget
   target_table: hindsight_memories | org_memory
   target_id: deleted row UUID
   content_hash: SHA-256 of deleted content
   metadata: {type: "<memory_type>"}
```

**Response 200:** `{"id": "<uuid>"}`
**Response 404:** not found in this tenant
**Response 403:** agent does not own, or type is forbidden

---

## Knowledge

### org_knowledge Section Splitting Rules

Authoritative rules for `POST /v1/knowledge` section splitting pipeline.

**Heading levels:**
- Split on `##` and `###` only
- `#` = document title — content under it is pre-heading content
- `####` and deeper = stay merged into parent `###` section, not split boundaries

**Pre-heading content** (before first `##`/`###`):
- `estimate_tokens(content) >= 50` → treated as section with implicit title `"Introduction"`
- `estimate_tokens(content) < 50` → discarded

**No-heading documents:**
- Treat entire document as one flat section
- Apply paragraph-boundary split if `estimate_tokens(content) > 2000`
- Never reject — plain-text documents are valid

**Heading detection regex (authoritative):**
```python
HEADING_PATTERN = re.compile(r'^(#{2,3})\s+(.+)$', re.MULTILINE)
```

**Section token counting ([section token counting](qortia/04-api-contracts.md#org_knowledge-section-splitting-rules)):** word-count approximation (`len(content.split()) * 1.3`).
Rationale: synchronous spaCy call per section is too slow for large documents. Word-count
approximation is within 10–15% of actual tokens — sufficient for split decisions.

### DELETE /v1/knowledge — Audit Trail

`DELETE /v1/knowledge/{source_path}` appends one row to `memory_history`:

| Field | Value |
|---|---|
| `operation` | `knowledge_delete` |
| `target_table` | `org_knowledge` |
| `target_id` | `NULL` — bulk delete by source_path, no single UUID |
| `content_hash` | `NULL` — content gone before audit row is written |
| `metadata` | `{"source_path": "<path>", "chunks_deleted": <count>}` |

Only defined case where `target_id` is NULL in `memory_history`. Schema comment notes this explicitly.

---

## memory_history

### Scope: Agent Operations Only

`memory_history` logs agent-initiated operations only.

**Exempt (platform-internal):**
- `org_chart` writes on agent provision/delete
- `weekly_summary` writes from background task

**Logged (agent-initiated):**
- `POST /v1/remember` → `operation: remember`
- `POST /v1/remember-org` → `operation: remember_org`
- `POST /v1/forget` → `operation: forget`
- `POST /v1/knowledge` → `operation: knowledge_ingest`
- `DELETE /v1/knowledge/{source_path}` → `operation: knowledge_delete`
- `POST /v1/reflect` → `operation: reflect` (one row per new memory written)

`agent_id NOT NULL` FK makes platform-internal writes structurally unloggable without a sentinel — exempt is correct.

### Retention on Agent Deletion

`hindsight_memories` → `ON DELETE CASCADE` on `agent_id` FK — deleted with agent.
`org_memory` → `ON DELETE SET NULL` on `author_id` FK — rows survive, `author_id` becomes NULL.
`memory_history` → `ON DELETE CASCADE` on `agent_id` FK — audit trail deleted with agent.

Rationale: private memories are personal and should not outlive the agent. Org memory is
a shared team resource — handoffs, processes, decision logs authored by deleted agents
remain valid and should be preserved for the tenant.

---

## Data Model Decisions

### org_knowledge: content_tsv Removed

`content_tsv` and its GIN index removed from `org_knowledge`.

Recall uses `index_tsv` for BM25 — `content_tsv` was unreferenced in any query path.
Retaining it caused ~20–30% write overhead on a hot path for zero retrieval benefit.

```sql
-- REMOVED:
content_tsv  TSVECTOR GENERATED ALWAYS AS (to_tsvector('unicode61', content)) STORED,
CREATE INDEX ON org_knowledge USING GIN (content_tsv);
```

### org_knowledge Dedup: Embedding Only

When ingesting knowledge, deduplication uses embedding cosine similarity only.
`index_fields` (keywords, entities) are recomputed from scratch on each ingest — never deduped.

**Why:** embedding captures semantic meaning; re-indexing keywords/entities ensures they
reflect the current version of the content, not a cached prior version.

### org_knowledge Re-ingest: Dedup Before Delete

On `POST /v1/knowledge` with an existing `source_path`:
1. Run dedup check — find semantically identical sections already present
2. Delete sections from the old version that are NOT identical to any new section
3. Insert new sections (skipping duplicates already present)

**Why dedup before delete:** deleting all old sections then reinserting everything wastes
embedding compute on sections that haven't changed. Dedup-first preserves unchanged sections
in-place, only updating sections that actually changed.

### HNSW Index Parameters

Explicit parameters on all three HNSW indexes. Defaults are correct for current scale.
Documented explicitly so they are intentional, not implicit.

```sql
-- hindsight_memories
CREATE INDEX ON hindsight_memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- org_memory
CREATE INDEX ON org_memory USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- org_knowledge
CREATE INDEX ON org_knowledge USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

**Tuning threshold:** increase `hindsight_memories` to `m=32, ef_construction=128` when any
single agent's row count exceeds 50K. `REINDEX` only — no schema migration.
`org_memory` and `org_knowledge` are bounded by tenant activity — defaults hold indefinitely.

---

## Weekly Org Summary

Asyncio background task inside FastAPI monolith. No pg_cron. No K8s CronJob.

Tracked via `qortia_tenants.weekly_summary_last_run_at`. Multi-replica safe via `SELECT FOR UPDATE SKIP LOCKED`.

**Flow per tenant (weekly):**
1. Acquire row lock (skip if another replica holds it)
2. Re-check `weekly_summary_last_run_at` inside lock (guard TOCTOU)
3. Fetch all handoffs from past 7 days
4. If fewer than 3: skip
5. Build structured summary deterministically — sort by date, format with agent name + date header. No LLM call.
6. Write to `org_memory` as `type: weekly_summary`
7. Mark individual handoffs as consolidated in metadata
8. Update `weekly_summary_last_run_at = now()`

---

## Org Memory RBAC (ADR-080)

Two-axis RBAC on `org_memory` and `org_knowledge`, enforced at Postgres RLS layer.

**Axis 1 — Clearance level (hierarchical, inclusive).**
Tenants define ordered levels. Higher order includes all lower. Platform seeds three defaults:
`external=1`, `internal=2`, `restricted=3`. Stored in `qortia_clearance_levels`.
Agent assignment in `qortia_agents.clearance_level`.

**Axis 2 — Division / audience (set membership).**
Tenants define divisions. Memory rows carry `audience TEXT[]`. Agent must be in audience or
audience must include `all`. Platform seeds one default: `all`. Stored in `tenant_divisions`.

**Combined access rule (`tenant_visibility_read` RLS policy):**
```
readable = agent.clearance_order >= memory.min_clearance_order
           AND (agent.division = ANY(memory.audience) OR 'all' = ANY(memory.audience))
```

**G4 safety guard:** RLS policy defaults to order 2 (`internal`) when session variable is unset:
```sql
coalesce(nullif(current_setting('app.memory_clearance_order', true), ''), '2')
```
Prevents silent zero-read regression during rollout window.

**org_chart is always external:** hardcoded to `min_clearance='external'`, `audience='{all}'`.
Every agent in the fleet must see the roster. Not operator-configurable.

**Backward compatibility:** existing agents default to `internal` clearance and `all` division.
Existing memory rows default to `min_clearance='internal'`, `audience='{all}'`. No behavior change
for tenants that do not customise RBAC.

**Migration:** `V4__org_memory_rbac.sql`

**Deferred:** writing clearance/division to agent's Vault path at provisioning time. Currently
`qortia_agents` is source of truth; agents do not read clearance from Vault at boot.
