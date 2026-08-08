"""Entity graph maintenance: summary updates, deduplication, and graph population."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from qortia import config
from qortia.auth import get_litellm_key, litellm_auth_headers
from qortia.common import get_litellm_client
from qortia.db import get_main_pool

logger = logging.getLogger(__name__)


async def _update_entity_summary(
    existing_summary: str | None,
    new_memory_content: str,
    litellm_key: str,
) -> str | None:
    """
    Maintain a running summary for an entity node.
    First call (no existing summary): return truncated memory content — no LLM.
    Subsequent calls: LLM merges existing summary with new memory content.
    Failure is non-fatal — returns existing_summary with a warning log.
    """
    if not existing_summary:
        return new_memory_content[:500]
    try:
        prompt = (
            f"Existing summary: {existing_summary}\n\n"
            f"New information: {new_memory_content[:300]}\n\n"
            "Update the summary to incorporate the new information. "
            "Be concise (2-3 sentences). Preserve named entities and temporal qualifiers."
        )
        async with asyncio.timeout(20.0):
            resp = await get_litellm_client().post(
                "/chat/completions",
                headers=litellm_auth_headers(litellm_key),
                json={
                    "model": "anthropic/claude-3-haiku-20240307",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                },
                timeout=15.0,
            )
        resp.raise_for_status()
        content: str = resp.json()["choices"][0]["message"]["content"]
        return content.strip()
    except Exception as exc:
        logger.warning({"event": "entity_summary_update_failed", "error": str(exc)})
        return existing_summary


async def _maybe_dedup_memory(
    memory_id: object,
    embedding: list[float],
    tenant_id: object,
    agent_id: object,
    memory_type: str,
) -> None:
    """Archive a memory if a near-duplicate (cosine >= DEDUP_SIMILARITY_THRESHOLD)
    exists within the last DEDUP_LOOKBACK_DAYS days for the same agent and type.
    Only called for episodic and experiential memories (ADR-105).
    """
    import json as _json

    async with get_main_pool().acquire() as conn:
        neighbour = await conn.fetchrow(
            """
            SELECT id, 1 - (embedding <=> $1::vector) AS similarity
            FROM hindsight_memories
            WHERE agent_id = $2
              AND type = $3
              AND tier = 'active'
              AND id != $4
              AND created_at > now() - interval '7 days'
              AND embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT 1
            """,
            str(embedding),
            agent_id,
            memory_type,
            memory_id,
        )
        if (
            neighbour
            and float(neighbour["similarity"]) >= config.settings.qortia_dedup_similarity_threshold
        ):
            await conn.execute(
                """
                UPDATE hindsight_memories
                SET tier = 'archive', metadata = metadata || $1
                WHERE id = $2
                """,
                _json.dumps({"dedup_of": str(neighbour["id"])}),
                memory_id,
            )
            logger.info(
                {
                    "event": "memory_dedup_archived",
                    "memory_id": str(memory_id),
                    "dedup_of": str(neighbour["id"]),
                    "similarity": float(neighbour["similarity"]),
                    "agent_id": str(agent_id),
                }
            )


async def _maybe_update_entity_summary(
    conn: Any,
    tenant_id: object,
    agent_id: object,
    entity_text: str,
    memory_content: str,
    is_org: bool,
) -> None:
    """
    Read the current link count and summary for the entity, then decide:
    - count == 1: bootstrap summary from memory content (no LLM)
    - count >= 3 and count % 3 == 0: call LLM to update summary
    - otherwise: no-op
    Failure is non-fatal.
    """
    try:
        if is_org:
            row = await conn.fetchrow(
                """
                SELECT array_length(linked_memory_ids, 1) AS link_count, summary
                FROM qortia_entities
                WHERE tenant_id = $1 AND entity_text = $2 AND agent_id IS NULL
                """,
                tenant_id,
                entity_text,
            )
        else:
            row = await conn.fetchrow(
                """
                SELECT array_length(linked_memory_ids, 1) AS link_count, summary
                FROM qortia_entities
                WHERE tenant_id = $1 AND agent_id = $2 AND entity_text = $3
                """,
                tenant_id,
                agent_id,
                entity_text,
            )
        if row is None:
            return
        link_count: int = row["link_count"] or 0
        existing_summary: str | None = row["summary"]

        if link_count == 1:
            new_summary: str | None = memory_content[:500]
        elif link_count >= 3 and link_count % 3 == 0:
            try:
                litellm_key = await get_litellm_key(str(tenant_id))
            except Exception as exc:
                logger.warning({"event": "entity_summary_key_fetch_failed", "error": str(exc)})
                return
            new_summary = await _update_entity_summary(
                existing_summary, memory_content, litellm_key
            )
        else:
            return

        if is_org:
            await conn.execute(
                """
                UPDATE qortia_entities SET summary = $1, updated_at = now()
                WHERE tenant_id = $2 AND entity_text = $3 AND agent_id IS NULL
                """,
                new_summary,
                tenant_id,
                entity_text,
            )
        else:
            await conn.execute(
                """
                UPDATE qortia_entities SET summary = $1, updated_at = now()
                WHERE tenant_id = $2 AND agent_id = $3 AND entity_text = $4
                """,
                new_summary,
                tenant_id,
                agent_id,
                entity_text,
            )
    except Exception as exc:
        logger.warning({"event": "entity_summary_write_failed", "error": str(exc)})


# ── Graph batch population ────────────────────────────────────


async def _populate_graph_batch() -> None:
    """
    Asynchronously links memories to the entity graph.
    Runs in the background to avoid write amplification on the main API path.
    """
    try:
        async with get_main_pool().acquire() as conn:
            # 1. Hindsight Memories (Private)
            rows = await conn.fetch(
                """
                SELECT id, tenant_id, agent_id, entities, content
                FROM hindsight_memories
                WHERE is_graphed = false AND entities != '[]'
                LIMIT 50
                FOR UPDATE SKIP LOCKED
            """
            )
            for row in rows:
                async with conn.transaction():
                    try:
                        entity_pairs: list[tuple[str, str]] = [
                            (e[0], e[1]) for e in json.loads(row["entities"])
                        ]
                    except Exception:
                        entity_pairs = []
                    for ent_text, ent_type in entity_pairs:
                        await conn.execute(
                            """
                            INSERT INTO qortia_entities
                                (tenant_id, agent_id, entity_text, entity_type, linked_memory_ids)
                            VALUES ($1, $2, $3, $4, ARRAY[$5::uuid])
                            ON CONFLICT (tenant_id, agent_id, entity_text)
                                WHERE agent_id IS NOT NULL
                            DO UPDATE SET
                                entity_type = EXCLUDED.entity_type,
                                linked_memory_ids =
                                    array_append(qortia_entities.linked_memory_ids, $5),
                                updated_at = now()
                            WHERE NOT ($5 = ANY(qortia_entities.linked_memory_ids))
                        """,
                            row["tenant_id"],
                            row["agent_id"],
                            ent_text,
                            ent_type,
                            row["id"],
                        )
                        await _maybe_update_entity_summary(
                            conn=conn,
                            tenant_id=row["tenant_id"],
                            agent_id=row["agent_id"],
                            entity_text=ent_text,
                            memory_content=row.get("content", ""),
                            is_org=False,
                        )
                    await conn.execute(
                        "UPDATE hindsight_memories SET is_graphed = true WHERE id = $1",
                        row["id"],
                    )

            # 2. Org Memories (Shared)
            rows = await conn.fetch(
                """
                SELECT id, tenant_id, entities, content
                FROM org_memory
                WHERE is_graphed = false AND entities != '[]'
                LIMIT 50
                FOR UPDATE SKIP LOCKED
            """
            )
            for row in rows:
                async with conn.transaction():
                    try:
                        entity_pairs = [(e[0], e[1]) for e in json.loads(row["entities"])]
                    except Exception:
                        entity_pairs = []
                    for ent_text, ent_type in entity_pairs:
                        await conn.execute(
                            """
                            INSERT INTO qortia_entities
                                (tenant_id, agent_id, entity_text, entity_type, linked_memory_ids)
                            VALUES ($1, NULL, $2, $3, ARRAY[$4::uuid])
                            ON CONFLICT (tenant_id, entity_text) WHERE agent_id IS NULL
                            DO UPDATE SET
                                entity_type = EXCLUDED.entity_type,
                                linked_memory_ids =
                                    array_append(qortia_entities.linked_memory_ids, $4),
                                updated_at = now()
                            WHERE NOT ($4 = ANY(qortia_entities.linked_memory_ids))
                        """,
                            row["tenant_id"],
                            ent_text,
                            ent_type,
                            row["id"],
                        )
                        await _maybe_update_entity_summary(
                            conn=conn,
                            tenant_id=row["tenant_id"],
                            agent_id=None,
                            entity_text=ent_text,
                            memory_content=row.get("content", ""),
                            is_org=True,
                        )
                    await conn.execute(
                        "UPDATE org_memory SET is_graphed = true WHERE id = $1",
                        row["id"],
                    )
    except Exception as exc:
        logger.warning({"event": "graph_population_failed", "error": str(exc)})
