from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from qortia import config
from qortia.auth import AgentIdentity, get_litellm_key, require_agent
from qortia.common import get_litellm_client
from qortia.db import get_main_pool, tenant_transaction
from qortia.embeddings import embed_text as _get_embedding
from qortia.embeddings import validate_embedding_config as validate_embedding_dimensions
from qortia.entity_graph import (
    _maybe_dedup_memory,
    _populate_graph_batch,
)
from qortia.entity_graph import (
    _maybe_update_entity_summary as _maybe_update_entity_summary,
)
from qortia.entity_graph import (
    _update_entity_summary as _update_entity_summary,
)
from qortia.models import ReflectResponse

# Re-export for callers/tests that historically imported from reflect.
__all__ = ("validate_embedding_dimensions",)

logger = logging.getLogger(__name__)
router = APIRouter()

REFLECTION_THRESHOLD = 10  # overridden by settings.reflection_threshold at call site
EMBEDDING_BATCH_SIZE = 50
STABILITY_THRESHOLD = 0.95
DEDUP_SIMILARITY_THRESHOLD = (
    0.95  # calibrated for BGE-M3 1024-dim; ADR-105; see settings.qortia_dedup_similarity_threshold
)
DEDUP_LOOKBACK_DAYS = 7  # see settings.qortia_dedup_lookback_days


def _compute_stability_scores(
    new_memories: list[dict],  # type: ignore[type-arg]
    existing_embeddings: dict[str, list[float] | None],
) -> list[float | None]:
    """
    For each new memory (CREATE or UPDATE), compute cosine similarity between
    its embedding and the embedding of the existing row it supersedes.

    Returns a list of stability scores (one per new memory).
    - UPDATE with both embeddings present: cosine similarity in [0, 1]
    - CREATE or missing embeddings: None

    new_memories: list of dicts with keys: action, content, id (optional),
                  embedding (list[float] | None)
    existing_embeddings: {str(id): embedding | None} for existing consolidated rows
    """
    from qortia.recall_helpers import _cosine

    scores: list[float | None] = []
    for mem in new_memories:
        if mem["action"] != "UPDATE" or not mem.get("id"):
            scores.append(None)
            continue
        new_emb = mem.get("embedding")
        old_emb = existing_embeddings.get(mem["id"])
        if new_emb and old_emb:
            scores.append(_cosine(new_emb, old_emb))
        else:
            scores.append(None)
    return scores


# ── POST /v1/reflect ─────────────────────────────────────────


@router.post("/v1/reflect", response_model=ReflectResponse)
async def reflect(agent: AgentIdentity = Depends(require_agent)) -> ReflectResponse:  # noqa: B008
    from qortia.common import assert_agent_active

    clearance_order, agent_division = agent.clearance_order, agent.division

    # Fetch data outside the write transaction — no DB lock held during LLM call
    recent: list[dict[str, Any]] = []
    existing: list[dict[str, Any]] = []
    async with tenant_transaction(
        get_main_pool(),
        agent.tenant_id,
        agent.agent_id,
        memory_clearance_order=clearance_order,
        agent_division=agent_division,
    ) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)

        recent = await conn.fetch(
            """
            SELECT content FROM hindsight_memories
            WHERE agent_id = $1
              AND type IN ('episodic', 'experiential')
              AND tier = 'active'
              AND (expires_at IS NULL OR expires_at > now())
              AND created_at > now() - interval '7 days'
            ORDER BY created_at DESC LIMIT 30
        """,
            agent.agent_id,
        )

        existing = await conn.fetch(
            """
            SELECT id, type, content, embedding FROM hindsight_memories
            WHERE agent_id = $1
              AND type IN ('mental_model', 'lesson')
              AND is_consolidated = true
            ORDER BY importance DESC
        """,
            agent.agent_id,
        )

    model = config.settings.rerank_model
    litellm_key = await get_litellm_key(str(agent.tenant_id))
    reflections = await _call_litellm_reflect(
        model=model,
        recent=[r["content"] for r in recent],
        existing=[
            {"id": str(r["id"]), "type": r["type"], "content": r["content"]} for r in existing
        ],
        litellm_key=litellm_key,
        agent_id=agent.agent_id,
        tenant_id=agent.tenant_id,
    )

    # Build existing embedding index keyed by id for stability computation
    existing_embeddings: dict[str, list[float] | None] = {
        str(r["id"]): list(r["embedding"]) if r.get("embedding") else None for r in existing
    }

    # Embed CREATE and UPDATE content before the write transaction.
    # Needed for: (a) stability score computation, (b) populating the embedding column
    # so the embedding worker doesn't need a separate pass for reflected memories.
    new_embeddings: dict[int, list[float] | None] = {}
    for i, r in enumerate(reflections):
        if r["action"] in ("CREATE", "UPDATE"):
            try:
                new_embeddings[i] = await _get_embedding(r["content"], litellm_key)
            except Exception as exc:
                logger.warning({"event": "reflect_embed_failed", "error": str(exc)})
                new_embeddings[i] = None

    # Write transaction
    memories_written, new_counter = await _write_reflections(
        agent_id=agent.agent_id,
        tenant_id=agent.tenant_id,
        reflections=reflections,
        new_embeddings=new_embeddings,
        existing_embeddings=existing_embeddings,
        existing=existing,
        clearance_order=clearance_order,
        agent_division=agent_division,
    )

    return ReflectResponse(memories_written=memories_written, reflection_counter=new_counter)


async def _write_reflections(  # noqa: C901
    agent_id: UUID,
    tenant_id: UUID,
    reflections: list[dict[str, Any]],
    new_embeddings: dict[int, list[float] | None],
    existing_embeddings: dict[str, list[float] | None],
    existing: list[dict[str, Any]],
    clearance_order: int,
    agent_division: str | None,
) -> tuple[int, int]:
    """Write reflection results to DB. Returns (memories_written, new_counter)."""
    unstable_count = 0
    async with tenant_transaction(
        get_main_pool(),
        tenant_id,
        agent_id,
        memory_clearance_order=clearance_order,
        agent_division=agent_division,
    ) as conn:
        active_ids = []
        for r in reflections:
            if r["action"] in ("UPDATE", "RETAIN"):
                active_ids.append(UUID(r["id"]))

        prune_safe = True
        if existing:
            existing_consolidated_count = await conn.fetchval(
                """
                SELECT count(*) FROM hindsight_memories
                WHERE agent_id = $1
                  AND type IN ('mental_model', 'lesson')
                  AND is_consolidated = true
                  AND valid_until IS NULL
                """,
                agent_id,
            )
            if existing_consolidated_count > 0 and len(active_ids) == 0:
                logger.warning(
                    {
                        "event": "reflect_prune_aborted_empty",
                        "agent_id": str(agent_id),
                        "existing_consolidated": existing_consolidated_count,
                        "reason": "LLM returned no active IDs — refusing to wipe all memories",
                    }
                )
                prune_safe = False
            elif (
                existing_consolidated_count > 0
                and len(active_ids) < existing_consolidated_count * 0.5
            ):
                logger.warning(
                    {
                        "event": "reflect_prune_aborted_low_coverage",
                        "agent_id": str(agent_id),
                        "existing_consolidated": existing_consolidated_count,
                        "active_ids_count": len(active_ids),
                        "reason": "LLM returned <50% of existing consolidated rows — "
                        "skipping prune",
                    }
                )
                prune_safe = False

        if existing and prune_safe:
            await conn.execute(
                """
                UPDATE hindsight_memories
                SET is_consolidated = false,
                    valid_until = now()
                WHERE agent_id = $1
                  AND type IN ('mental_model', 'lesson')
                  AND is_consolidated = true
                  AND id != ALL($2::uuid[])
                """,
                agent_id,
                active_ids,
            )

        new_ids: list[tuple[object, object]] = []
        seen_hashes: set[str] = set()
        stable_count = 0
        unstable_count = 0
        for i, r in enumerate(reflections):
            if r["action"] == "RETAIN":
                continue

            content_hash = hashlib.sha256(r["content"].lower().strip().encode()).hexdigest()
            if content_hash in seen_hashes:
                logger.info({"event": "reflect_dedup_skipped", "hash": content_hash})
                continue
            seen_hashes.add(content_hash)

            new_emb = new_embeddings.get(i)
            stability: float | None = None
            if r["action"] == "UPDATE" and r.get("id"):
                old_emb = existing_embeddings.get(r["id"])
                if new_emb and old_emb:
                    from qortia.recall_helpers import _cosine

                    stability = _cosine(new_emb, old_emb)
                    if stability >= STABILITY_THRESHOLD:
                        stable_count += 1
                    else:
                        unstable_count += 1
            elif r["action"] == "CREATE":
                unstable_count += 1

            if r["action"] == "UPDATE":
                await conn.execute(
                    """
                    UPDATE hindsight_memories
                    SET is_consolidated = false
                    WHERE id = $1 AND agent_id = $2
                    """,
                    UUID(r["id"]),
                    agent_id,
                )

            try:
                from qortia.knowledge import extract_entities_with_types

                entities = extract_entities_with_types(r["content"], lang=r.get("lang", "en"))
            except Exception:
                entities = []

            row_id = await conn.fetchval(
                """
                INSERT INTO hindsight_memories
                    (tenant_id, agent_id, type, content, importance,
                     is_consolidated, entities, embedding, stability_score)
                VALUES ($1, $2, $3, $4, $5, true, $6, $7::vector, $8)
                RETURNING id
                """,
                tenant_id,
                agent_id,
                r["type"],
                r["content"],
                float(r["importance"]),
                json.dumps(entities),
                str(new_emb) if new_emb else None,
                stability,
            )
            new_ids.append((row_id, r.get("id")))

        new_counter = await conn.fetchval(
            """
            UPDATE qortia_agents
            SET reflection_counter = GREATEST(reflection_counter - $1, 0),
                updated_at = now()
            WHERE id = $2
            RETURNING reflection_counter
            """,
            config.settings.reflection_threshold,
            agent_id,
        )

        for row_id, parent_id in new_ids:
            metadata: dict[str, Any] = {}
            if parent_id:
                metadata["parent_id"] = parent_id
            await conn.execute(
                """
                INSERT INTO memory_history
                    (tenant_id, agent_id, operation, target_table, target_id, metadata)
                VALUES ($1, $2, 'reflect', 'hindsight_memories', $3, $4)
                """,
                tenant_id,
                agent_id,
                row_id,
                json.dumps(metadata),
            )

    logger.info(
        {
            "event": "reflect_incremental",
            "qortia.agent_id": str(agent_id),
            "qortia.tenant_id": str(tenant_id),
            "memories_written": len(new_ids),
            "stable_updates": stable_count,
            "unstable_updates": unstable_count,
        }
    )
    return len(new_ids), new_counter


async def _call_litellm_reflect(  # noqa: C901
    model: str,
    recent: list[str],
    existing: list[dict],  # type: ignore[type-arg]
    litellm_key: str,
    agent_id: UUID,
    tenant_id: UUID,
) -> list[dict]:  # type: ignore[type-arg]
    prompt = _build_reflect_prompt(recent, existing)

    async with asyncio.timeout(125.0):
        resp = await get_litellm_client().post(
            "/chat/completions",
            headers={"Authorization": f"Bearer {litellm_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            timeout=120.0,
        )

    if resp.status_code != 200:
        raise HTTPException(500, f"LiteLLM error: {resp.status_code}")

    try:
        raw_resp = resp.json()
        usage = raw_resp.get("usage", {})
        logger.info(
            {
                "event": "qortia_llm_reflect",
                "qortia.tenant_id": str(tenant_id),
                "model": model,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        )
        raw = raw_resp["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        reflections = parsed.get("reflections", [])
        if not reflections:
            raise ValueError("empty reflections array")
        for r in reflections:
            action = r.get("action")
            if action not in ("CREATE", "UPDATE", "RETAIN"):
                raise ValueError(f"invalid action: {action}")
            if action in ("UPDATE", "RETAIN") and not r.get("id"):
                raise ValueError(f"id required for {action}")
            if action in ("CREATE", "UPDATE"):
                if r.get("type") not in ("mental_model", "lesson"):
                    raise ValueError(f"invalid type: {r['type']}")
                if not isinstance(r.get("importance"), int | float):
                    raise ValueError("importance must be numeric")
                if not r.get("content"):
                    raise ValueError("content must not be empty")
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error({"event": "reflection_llm_malformed", "error": str(exc)})
        raise HTTPException(500, "Reflection failed: malformed LLM response")  # noqa: B904

    return reflections  # type: ignore[no-any-return]


def _build_reflect_prompt(recent: list[str], existing: list[dict]) -> str:  # type: ignore[type-arg]
    from qortia.remember import build_temporal_grounding_instruction

    recent_block = "\n".join(f"- {m}" for m in recent) or "(none)"
    existing_block = (
        "\n".join(f"- [{m['id']}] [{m['type']}] {m['content']}" for m in existing) or "(none)"
    )
    temporal_instruction = build_temporal_grounding_instruction()
    return f"""You are synthesising an agent's recent experiences into durable mental models
and lessons.

{temporal_instruction}

Recent episodic and experiential memories (last 7 days):
{recent_block}

Existing consolidated knowledge (will be superseded by your output):
{existing_block}

Your task is to refine the agent's knowledge. You can CREATE new models, UPDATE existing
ones with more detail, or RETAIN existing ones that are still accurate.

Return a JSON object with this exact schema:
{{
  "reflections": [
    {{"action": "CREATE", "type": "mental_model", "content": "...", "importance": 0.85}},
    {{"action": "UPDATE", "id": "uuid-here", "type": "lesson", "content": "updated content",
      "importance": 0.90}},
    {{"action": "RETAIN", "id": "uuid-here"}}
  ]
}}

Rules:
- action must be "CREATE", "UPDATE", or "RETAIN"
- type must be "mental_model" or "lesson" only
- importance is a float between 0.0 and 1.0
- content must be a non-empty string
- Return all knowledge items that should remain consolidated. Any existing ID NOT returned
  will be pruned (deleted).
- Synthesise patterns, not events
- Preserve named entities (people, systems, organisations) — they are recall anchors
- When a memory contains a temporal marker ("last quarter", "in March"), preserve it — do
  not strip temporal context
- Preserve specific resolved dates from source memories — do NOT convert them back to
  relative references
- Attribute observations to their source when relevant: "user reported X", "agent observed
  Y", "team decided Z"
- Preserve [User], [Observed], [Third-party] attribution prefixes from source memories

NEVER include:
- Specific dates or timestamps unless they define a durable pattern
- Pronouns or vague references ("they", "it", "the thing")
- Single-occurrence events that have not recurred
- Ephemeral details (ticket numbers, session IDs, temporary values)
- Abstract concepts without grounding ("things went well", "it was difficult")"""


# ── Archival background task ─────────────────────────────────


async def run_archival_task() -> None:
    while True:
        await asyncio.sleep(86400 * 7)  # weekly
        await _archive_old_episodic_memories()
        await _purge_expired_short_term_memories()


async def _archive_old_episodic_memories() -> None:
    try:
        async with get_main_pool().acquire() as conn:
            result = await conn.execute(
                """
                UPDATE hindsight_memories
                SET tier = 'archive'
                WHERE type = 'episodic'
                  AND tier = 'active'
                  AND created_at < now() - interval '30 days'
                  AND importance < 0.4
                  AND recall_count < 3
                """
            )
            archived = int(result.split()[-1])
            if archived > 0:
                logger.info({"event": "memories_archived", "count": archived})
    except Exception as exc:
        logger.warning({"event": "archival_task_failed", "error": str(exc)})


async def _purge_expired_short_term_memories() -> None:
    try:
        async with get_main_pool().acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM hindsight_memories
                WHERE type = 'short_term' AND expires_at < now()
                """
            )
            purged = int(result.split()[-1])
            if purged > 0:
                logger.info({"event": "short_term_memories_purged", "count": purged})
    except Exception as exc:
        logger.warning({"event": "short_term_purge_failed", "error": str(exc)})


# ── Embedding worker ─────────────────────────────────────────


async def run_embedding_worker() -> None:
    while True:
        await asyncio.sleep(10)
        await _process_embedding_batch()
        await _populate_graph_batch()


async def _process_embedding_batch() -> None:
    # Step 1: claim rows outside a long-held transaction
    rows: list[dict] = []  # type: ignore[type-arg]
    async with get_main_pool().acquire() as conn:
        async with conn.transaction():
            hindsight = await conn.fetch(
                """
                SELECT id, tenant_id, agent_id, content AS text_to_embed, lang,
                       'hindsight_memories' AS tbl
                FROM hindsight_memories
                WHERE embedding IS NULL AND embedding_attempts < 3
                ORDER BY tenant_id, created_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            """,
                EMBEDDING_BATCH_SIZE,
            )

            org_mem = await conn.fetch(
                """
                SELECT id, tenant_id, content AS text_to_embed, lang, 'org_memory' AS tbl
                FROM org_memory
                WHERE embedding IS NULL AND embedding_attempts < 3
                ORDER BY tenant_id, created_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            """,
                EMBEDDING_BATCH_SIZE,
            )

            org_know = await conn.fetch(
                """
                SELECT id, tenant_id, index_summary AS text_to_embed, lang, 'org_knowledge' AS tbl
                FROM org_knowledge
                WHERE embedding IS NULL AND embedding_attempts < 3
                ORDER BY tenant_id, created_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            """,
                EMBEDDING_BATCH_SIZE,
            )

            entities = await conn.fetch(
                """
                SELECT id, tenant_id, entity_text AS text_to_embed, 'qortia_entities' AS tbl
                FROM qortia_entities
                WHERE embedding IS NULL AND embedding_attempts < 3
                ORDER BY tenant_id, created_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            """,
                EMBEDDING_BATCH_SIZE,
            )

            rows = [
                dict(r) for r in list(hindsight) + list(org_mem) + list(org_know) + list(entities)
            ]

    # Group by tenant_id (standalone: one configured LiteLLM key; batch stays per-tenant)
    from collections import defaultdict

    rows_by_tenant = defaultdict(list)
    for row in rows:
        rows_by_tenant[str(row["tenant_id"])].append(row)

    for tenant_id_str, tenant_rows in rows_by_tenant.items():
        try:
            litellm_key = await get_litellm_key(tenant_id_str)
        except Exception as exc:
            logger.error(
                {
                    "event": "embedding_key_fetch_failed",
                    "tenant_id": tenant_id_str,
                    "error": str(exc),
                }
            )
            continue
        for row in tenant_rows:
            await _embed_single_row(row, litellm_key)


async def _embed_single_row(row: dict, litellm_key: str) -> None:  # type: ignore[type-arg]
    if not row.get("text_to_embed"):
        return
    # G5: skip embedding for short_term — BM25-only; embedding wastes storage (ADR-105)
    if row.get("tbl") == "hindsight_memories" and row.get("type") == "short_term":
        logger.info(
            {
                "event": "embedding_skipped",
                "reason": "short_term_mrl",
                "memory_id": str(row["id"]),
            }
        )
        return
    try:
        embedding = await _get_embedding(
            row["text_to_embed"],
            litellm_key,
            tenant_id=str(row["tenant_id"]),
        )
        async with get_main_pool().acquire() as conn:
            await conn.execute(
                f"UPDATE {row['tbl']} SET embedding = $1::vector WHERE id = $2",  # noqa: S608
                str(embedding),
                row["id"],
            )
        # Cross-memory linking (16i) — only for hindsight_memories
        if row["tbl"] == "hindsight_memories":
            from qortia.links import _find_similar_memories, _upsert_memory_links

            similar = await _find_similar_memories(
                memory_id=row["id"],
                embedding=embedding,
                tenant_id=row["tenant_id"],
                agent_id=row["agent_id"],
            )
            await _upsert_memory_links(row["id"], similar, row["tenant_id"])
            # G3: post-embed dedup for episodic/experiential only (ADR-105)
            if row.get("type") in ("episodic", "experiential"):
                await _maybe_dedup_memory(
                    row["id"], embedding, row["tenant_id"], row["agent_id"], row["type"]
                )
    except Exception as exc:
        async with get_main_pool().acquire() as conn:
            await conn.execute(
                f"UPDATE {row['tbl']} SET embedding_attempts = embedding_attempts + 1 "  # noqa: S608
                "WHERE id = $1",
                row["id"],
            )
            attempts = await conn.fetchval(
                f"SELECT embedding_attempts FROM {row['tbl']} WHERE id = $1",  # noqa: S608
                row["id"],
            )
        if attempts and attempts >= 3:
            logger.warning(
                {
                    "event": "embedding_failed",
                    "table": row["tbl"],
                    "row_id": str(row["id"]),
                    "attempts": attempts,
                    "error": str(exc),
                }
            )


# ── Background reflection trigger ────────────────────────────


async def run_background_reflection_trigger() -> None:
    """Supervised background task: triggers reflection for agents idle > window
    that have accumulated at least `reflection_threshold` new episodic memories."""
    while True:
        await asyncio.sleep(config.settings.idle_reflection_interval_s)
        await _trigger_idle_reflections()


async def _trigger_idle_reflections() -> None:
    try:
        async with get_main_pool().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT a.id AS agent_id, a.tenant_id
                FROM qortia_agents a
                WHERE a.status = 'active'
                  AND a.reflection_counter >= $1
                  AND a.updated_at < now() - ($2 * interval '1 hour')
                  AND EXISTS (
                      SELECT 1 FROM hindsight_memories hm
                      WHERE hm.agent_id = a.id
                      LIMIT 1
                  )
                LIMIT 50
                """,
                config.settings.reflection_threshold,
                config.settings.idle_reflection_window_h,
            )
        for row in rows:
            await _reflect_agent(UUID(str(row["agent_id"])), UUID(str(row["tenant_id"])))
    except Exception as exc:
        logger.warning({"event": "idle_reflection_trigger_failed", "error": str(exc)})


async def _reflect_agent(agent_id: UUID, tenant_id: UUID) -> None:
    """Run reflection for a single agent from the background trigger.

    Mirrors the HTTP /v1/reflect endpoint logic but accepts raw UUIDs rather than
    an AgentIdentity so it can be called without an inbound JWT context.
    """
    from qortia.remember import _fetch_agent_clearance

    try:
        clearance_order, agent_division = await _fetch_agent_clearance(agent_id, tenant_id)
        recent: list[dict[str, Any]] = []
        existing: list[dict[str, Any]] = []

        async with tenant_transaction(
            get_main_pool(),
            tenant_id,
            agent_id,
            memory_clearance_order=clearance_order,
            agent_division=agent_division,
        ) as conn:
            status = await conn.fetchval(
                "SELECT status FROM qortia_agents WHERE id = $1 AND tenant_id = $2",
                agent_id,
                tenant_id,
            )
            if status != "active":
                return

            recent = await conn.fetch(
                """
                SELECT content FROM hindsight_memories
                WHERE agent_id = $1
                  AND type IN ('episodic', 'experiential')
                  AND tier = 'active'
                  AND (expires_at IS NULL OR expires_at > now())
                  AND created_at > now() - interval '7 days'
                ORDER BY created_at DESC LIMIT 30
                """,
                agent_id,
            )
            existing = await conn.fetch(
                """
                SELECT id, type, content, embedding FROM hindsight_memories
                WHERE agent_id = $1
                  AND type IN ('mental_model', 'lesson')
                  AND is_consolidated = true
                ORDER BY importance DESC
                """,
                agent_id,
            )

        if not recent:
            return  # nothing to reflect on

        model = config.settings.rerank_model
        litellm_key = await get_litellm_key(str(tenant_id))

        reflections = await _call_litellm_reflect(
            model=model,
            recent=[r["content"] for r in recent],
            existing=[
                {"id": str(r["id"]), "type": r["type"], "content": r["content"]} for r in existing
            ],
            litellm_key=litellm_key,
            agent_id=agent_id,
            tenant_id=tenant_id,
        )

        existing_embeddings: dict[str, list[float] | None] = {
            str(r["id"]): list(r["embedding"]) if r.get("embedding") else None for r in existing
        }
        new_embeddings: dict[int, list[float] | None] = {}
        for i, r in enumerate(reflections):
            if r["action"] in ("CREATE", "UPDATE"):
                try:
                    new_embeddings[i] = await _get_embedding(r["content"], litellm_key)
                except Exception as exc:
                    logger.warning({"event": "reflect_embed_failed", "error": str(exc)})
                    new_embeddings[i] = None

        await _write_reflections(
            agent_id=agent_id,
            tenant_id=tenant_id,
            reflections=reflections,
            new_embeddings=new_embeddings,
            existing_embeddings=existing_embeddings,
            existing=list(existing),
            clearance_order=clearance_order,
            agent_division=agent_division,
        )
    except Exception as exc:
        logger.warning(
            {
                "event": "background_reflect_failed",
                "qortia.agent_id": str(agent_id),
                "qortia.tenant_id": str(tenant_id),
                "error": str(exc),
            }
        )
