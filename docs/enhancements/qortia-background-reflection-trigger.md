---
kind: enhancement
owner: platform
last_reviewed: 2026-05-18
status: implemented
---

# Qortia: Proactive Background Reflection Trigger

**Status:** Implemented — 2026-05-30
**Scope:** `platform/app/qortia/reflect.py`, `platform/app/main.py`
**ADR required:** No — uses existing `reflect()` logic; no schema change, no new
endpoints, no API contract change
**Depends on:** None (but aligns with GitHub Epic #3 — Proactive Triggers)
**Research source:** "Context Architecture for Production AI Agents" (Redis, 2026) —
Pillar 3 "Improve": "Background extraction creates compounding value: context
improves automatically without adding latency to the critical path." Also:
"Managing Memory for AI Agents" (O'Reilly/Redis, 2026) — Ch. 2, "cascading memory
systems: allowing the agent itself to choose what to promote to long-term storage."

---

## 1. The Problem

The reflection cycle in the platform is agent-triggered. An agent accumulates episodic
memories, the `reflection_counter` increments, and when the counter reaches
`REFLECTION_THRESHOLD` (10), the agent is expected to call `POST /v1/reflect` to
consolidate those memories into `mental_model` and `lesson` types.

This design has a critical gap: **reflection only happens if the agent calls it.**

Agents that are idle, stuck in a work order, or simply not prompted to reflect will
accumulate raw episodic memories indefinitely without consolidation. The
`reflection_counter` can grow to 50, 100, or more with no automatic backstop.
The consequences:

- **Recall quality degrades.** The full hybrid pipeline's `_vector_private` and
  `_bm25_private` queries scan an ever-growing pool of unconsolidated episodic
  memories. Without reflection, there are no `mental_model` or `lesson` rows to
  surface in the `context` endpoint — the agent's working context becomes stale.

- **The compounding memory flywheel never starts.** The Redis research finding is
  that context systems should improve automatically with usage. In the platform, this
  compounding only happens if the agent is actively prompted to reflect. An agent
  that processes 100 work orders but never self-triggers reflection has no durable
  knowledge — only raw episodic observations.

- **No operator visibility.** There is no alert or metric for agents with a high
  `reflection_counter` that haven't reflected recently. The condition is invisible
  until recall quality degrades noticeably.

---

## 2. Design

### 2.1 Background Task: `run_reflection_trigger`

Add a new supervised background task to `reflect.py` that runs on a configurable
interval (default: every 6 hours) and identifies agents that:

1. Have `reflection_counter >= REFLECTION_THRESHOLD` (i.e. have enough new episodic
   memories to warrant reflection), AND
2. Have not reflected in the last `REFLECTION_IDLE_HOURS` (default: 24 hours) —
   i.e. the agent has not self-triggered reflection recently

For each qualifying agent, the task calls the existing `reflect()` logic directly
(not via HTTP — internal function call) on behalf of the agent.

```python
REFLECTION_TRIGGER_INTERVAL_SECONDS = 6 * 3600  # 6 hours
REFLECTION_IDLE_HOURS = 24


async def run_reflection_trigger() -> None:
    while True:
        await asyncio.sleep(REFLECTION_TRIGGER_INTERVAL_SECONDS)
        await _trigger_pending_reflections()


async def _trigger_pending_reflections() -> None:
    try:
        async with get_main_pool().acquire() as conn:
            agents = await conn.fetch(
                """
                SELECT id, tenant_id
                FROM qortia_agents
                WHERE status = 'active'
                  AND reflection_counter >= $1
                  AND (
                    updated_at < now() - interval '1 hour' * $2
                    OR updated_at IS NULL
                  )
                LIMIT 50
                """,
                REFLECTION_THRESHOLD,
                REFLECTION_IDLE_HOURS,
            )
        for row in agents:
            try:
                identity = AgentIdentity(agent_id=row["id"], tenant_id=row["tenant_id"])
                await reflect(agent=identity)
                logger.info(
                    {
                        "event": "reflection_triggered_background",
                        "agent_id": str(row["id"]),
                        "tenant_id": str(row["tenant_id"]),
                    }
                )
            except Exception as exc:
                logger.warning(
                    {
                        "event": "reflection_trigger_failed",
                        "agent_id": str(row["id"]),
                        "tenant_id": str(row["tenant_id"]),
                        "error": str(exc),
                    }
                )
    except Exception as exc:
        logger.warning({"event": "reflection_trigger_scan_failed", "error": str(exc)})
```

### 2.2 Registration in `main.py`

Register `run_reflection_trigger` with `start_supervised_tasks` alongside the
existing `run_embedding_worker` and `run_archival_task`:

```python
# platform/app/main.py — startup
start_supervised_tasks(
    [
        run_embedding_worker,
        run_archival_task,
        run_reflection_trigger,  # ← new
    ]
)
```

### 2.3 Idle Detection Logic

The `updated_at` column on `qortia_agents` is bumped on every `reflect()` call
(via `UPDATE qortia_agents SET reflection_counter = ..., updated_at = now()`).
This makes it a reliable proxy for "last reflected at" without adding a new column.

The condition `updated_at < now() - interval '1 hour' * REFLECTION_IDLE_HOURS`
correctly identifies agents that haven't reflected (or been updated for any reason)
in the idle window. The `OR updated_at IS NULL` guard handles newly provisioned
agents.

**Why 24 hours?** The reflection cycle is designed to consolidate a week of episodic
memories. Running it more frequently than daily produces diminishing returns and
increases LiteLLM cost. 24 hours is the minimum meaningful interval.

### 2.4 Tenant Isolation

The query uses `get_main_pool().acquire()` (not `tenant_transaction`) because it
reads from `qortia_agents` — an `auth.*` table that is explicitly excluded from RLS
(per `agent-maintenance-rule.md` Invariant #4). The subsequent `reflect()` call
uses `tenant_transaction` internally, which enforces tenant isolation at the
memory write level.

The `LIMIT 50` cap prevents a single trigger run from processing an unbounded
number of agents. With a 6-hour interval and a 50-agent cap, the task can process
200 agents per day — sufficient for the current scale.

---

## 3. Observability

Add a Prometheus counter for background-triggered reflections:

```python
# platform/app/metrics.py
background_reflections_total = Counter(
    "qortia_background_reflections_total",
    "Total number of background-triggered reflection cycles",
    ["tenant_id"],
)
```

Increment in `_trigger_pending_reflections` after each successful reflection.
Note: `tenant_id` as a label is acceptable here because tenant count is bounded
(enterprise multi-tenancy, not unbounded user-level cardinality).

---

## 4. Files Affected

| File | Change |
|---|---|
| `platform/app/qortia/reflect.py` | Add `run_reflection_trigger`, `_trigger_pending_reflections`, constants |
| `platform/app/main.py` | Register `run_reflection_trigger` with `start_supervised_tasks` |
| `platform/app/metrics.py` | Add `qortia_background_reflections_total` counter |

---

## 5. Test Gates

| Gate | What to verify |
|---|---|
| Unit test — `test_reflect.py` | `_trigger_pending_reflections` queries agents with `reflection_counter >= threshold` and `updated_at` beyond idle window |
| Unit test — `test_reflect.py` | Agents with `reflection_counter < threshold` are not triggered |
| Unit test — `test_reflect.py` | Agents with recent `updated_at` (within idle window) are not triggered |
| Integration test | Agent with `reflection_counter = 10` and `updated_at` 25h ago is reflected; `reflection_counter` decrements |
| Integration test | `start_supervised_tasks` patch in `conftest.py` covers `run_reflection_trigger` (no-op in tests) |
| Platform unit tests | `cd platform && python3 -m pytest tests/unit/ -q` — 798/798 |
| Full stack health | `python3 scripts/local_agents.py` — 21/21 checks |

---

## 6. Known Constraints

**LiteLLM cost.** Each background reflection call invokes the LiteLLM `/chat/completions`
endpoint. At 50 agents per trigger run and 4 runs per day, this is up to 200 LiteLLM
calls per day from the background task alone. At Haiku pricing this is negligible,
but operators should be aware. The `REFLECTION_TRIGGER_INTERVAL_SECONDS` constant
is configurable via `settings` if cost reduction is needed.

**Agent must be `status = 'active'`.** The query filters `status = 'active'` — agents
that are `inactive`, `provisioning`, or `error` are skipped. This is correct: calling
`reflect()` for an inactive agent would fail `assert_agent_active`.

**`updated_at` as a proxy for last-reflected-at.** `updated_at` is bumped by any
`qortia_agents` write, not just reflection. An agent that was updated for a different
reason (e.g. a status change) within the idle window will not be triggered even if
it hasn't reflected. This is acceptable — the false-negative rate is low and the
consequence is a delayed reflection, not a missed one.

---

## 7. Related

- `run_embedding_worker` and `run_archival_task` in `reflect.py` — existing
  background task pattern to follow
- `start_supervised_tasks` in `main.py` — registration pattern
- `testing-rule.md` — "Patch `start_supervised_tasks` itself to a no-op" in
  integration tests
- GitHub Epic #3 (Proactive Triggers) — this is the first proactive trigger
  implementation; the pattern established here should be reused for future triggers
