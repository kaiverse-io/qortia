---
kind: enhancement
owner: platform
last_reviewed: 2026-05-18
status: post-mvp
---

# Qortia: Temporal Fact Bounds on Hindsight Memories

**Status:** Implemented — `valid_from` / `valid_until` landed in V2__temporal_bounds.sql; `_temporal_filter_clause` active in `platform/app/qortia/recall.py`
**Scope:** `platform/migrations/V7__temporal_bounds.sql`,
`platform/app/qortia/models.py`, `platform/app/qortia/remember.py`,
`platform/app/qortia/reflect.py`, `platform/app/qortia/recall.py`,
`the agent runtime/mcp_bridge.py`
**ADR required:** Yes — changes recall semantics (superseded memories excluded
by default), new API contract (`as_of` parameter, `valid_from` on `MemoryItem`)
**Depends on:** None
**GitHub issue:** #73
**Research source:** `docs/research/zep-graphiti-review.md` §6 (Temporal Bounds)
and §11 (Actionable Pattern #1). Graphiti's `valid_at`/`invalid_at`/`expired_at`
on `EntityEdge` — the only memory framework reviewed that models temporal fact
evolution natively.

---

## 1. The Problem

the platform memories have a `created_at` timestamp but no concept of when a fact
*stopped being true*. This creates two concrete problems:

**Stale facts pollute recall.** An agent that learned "Alice is the project lead"
in January, then learned "Bob replaced Alice as project lead" in March, has both
facts in `hindsight_memories`. The reflection cycle marks the January memory as
`is_consolidated = true` but does not record *when* it became invalid. A recall
query in April returns both facts — the agent must reason about which is current.

**No temporal queries.** There is no way to ask "what did this agent believe about
Project X on 2026-03-15?" — a query that is essential for audit trails, debugging
agent decisions, and understanding why an agent acted a certain way at a point in time.

Graphiti solves this with `valid_at`/`invalid_at` timestamps on fact edges. The
same pattern maps directly to two nullable columns on `hindsight_memories` without
requiring a graph database.

---

## 2. Design

### 2.1 Schema — V7 migration

```sql
ALTER TABLE hindsight_memories
    ADD COLUMN valid_from  TIMESTAMPTZ,
    ADD COLUMN valid_until TIMESTAMPTZ;

-- NULL valid_from = "since the beginning"
-- NULL valid_until = "still current"
-- Both nullable — all existing rows are valid without change

CREATE INDEX ON hindsight_memories (agent_id, valid_from)
    WHERE valid_from IS NOT NULL;
CREATE INDEX ON hindsight_memories (agent_id, valid_until)
    WHERE valid_until IS NOT NULL;
```

### 2.2 Reflection cycle change

When `reflect.py` supersedes an old memory (marks `is_consolidated = true`),
it now also sets `valid_until = now()` on the superseded row:

```python
await conn.execute(
    """
    UPDATE hindsight_memories
    SET is_consolidated = true,
        valid_until = now()
    WHERE id = ANY($1::uuid[])
    """,
    superseded_ids,
)
```

This is the primary mechanism for populating `valid_until` — the reflection cycle
already identifies which memories are superseded by newer consolidated knowledge.

### 2.3 `remember` tool change

`MemoryItem` gains an optional `valid_from` field. The LLM can assert when a fact
became true:

```python
class MemoryItem(BaseModel):
    ...
    valid_from: datetime | None = None  # ISO 8601, optional
```

If absent, `valid_from` is set to `created_at` at write time (i.e. "this fact
became true when I observed it"). If present, the LLM-asserted value is stored
directly — enabling "Alice joined the team in March 2026" to be stored with
`valid_from = 2026-03-01` even if written in July.

### 2.4 Recall pipeline change

`RecallRequest` gains an optional `as_of` timestamp parameter:

```python
class RecallRequest(BaseModel):
    ...
    as_of: datetime | None = None  # ISO 8601, optional
```

When `as_of` is provided, all type-routed recall functions add:

```sql
AND (valid_from IS NULL OR valid_from <= :as_of)
AND (valid_until IS NULL OR valid_until > :as_of)
```

**Default behaviour (no `as_of`):** Only return currently-valid memories:

```sql
AND (valid_until IS NULL)
```

This is a **breaking change** in recall semantics. Currently, superseded memories
(`is_consolidated = true`) are returned unless the caller explicitly filters them.
After this change, memories with `valid_until` set are excluded from default recall.

The `is_consolidated` filter remains for backward compatibility — it is a separate
signal (whether the memory was processed by reflection) from `valid_until` (whether
the fact is still current). Both can be true independently.

### 2.5 MCP bridge change

`recall` tool gains optional `as_of` parameter (ISO 8601 string):

```python
Tool(name="recall", ..., inputSchema={..., "properties": {
    ...
    "as_of": {"type": "string", "description": "ISO 8601 timestamp. Return only memories valid at this point in time. Omit for current memories only."},
}})
```

`remember` tool: `valid_from` added to each memory item's schema.

---

## 3. Files Affected

| File | Change |
|---|---|
| `platform/migrations/V7__temporal_bounds.sql` | `valid_from`, `valid_until` columns + indexes |
| `platform/app/qortia/models.py` | `valid_from` on `MemoryItem`; `as_of` on `RecallRequest` |
| `platform/app/qortia/remember.py` | Write `valid_from` on INSERT (default to `created_at`) |
| `platform/app/qortia/reflect.py` | Set `valid_until = now()` on superseded memories |
| `platform/app/qortia/recall.py` | `as_of` filter on all type-routed recall functions; default `valid_until IS NULL` filter |
| `the agent runtime/mcp_bridge.py` | `as_of` on `recall` tool; `valid_from` on `remember` tool |

---

## 4. Test Gates

| Gate | What to verify |
|---|---|
| Unit — `test_qortia_models.py` | `valid_from` accepted on `MemoryItem`; `as_of` accepted on `RecallRequest` |
| Unit — `test_recall_pipeline.py` | `as_of` filter clause generated correctly; default `valid_until IS NULL` filter present |
| Integration | Write two conflicting memories → trigger reflection → older has `valid_until` set |
| Integration | Default recall excludes memories with `valid_until` set |
| Integration | `recall` with `as_of` before supersession returns the older memory |
| Integration | `recall` with `as_of` after supersession returns the newer memory only |
| Eval regression | `evals/run_reh.py` — Recall@5 ≥ 0.95, MRR ≥ 0.86 after default filter change |
| Platform unit tests | `cd platform && python3 -m pytest tests/unit/ -q` — 296/296 |
| Full stack health | `python3 scripts/local_agents.py` — 21/21 checks |

---

## 5. Known Constraints

**Breaking recall semantics.** The default `valid_until IS NULL` filter changes
what `recall` returns for agents that have been running for a while and have
superseded memories. The eval suite must be run before and after to confirm no
regression. The ADR must document this semantic change explicitly.

**`valid_from` is LLM-asserted.** The platform trusts the LLM's `valid_from`
value without validation. A malformed or future-dated `valid_from` is stored
as-is. The platform does not enforce `valid_from <= created_at`.

**Reflection is the primary supersession mechanism.** `valid_until` is only set
by the reflection cycle — not by the agent explicitly. An agent cannot directly
mark a memory as invalid; it must write a new memory and let reflection consolidate.
This is intentional — direct invalidation would allow agents to delete their own
history.

**`is_consolidated` and `valid_until` are independent.** A memory can be
`is_consolidated = true` (processed by reflection) but `valid_until IS NULL`
(still current — reflection confirmed it rather than superseding it). Both signals
are preserved.
