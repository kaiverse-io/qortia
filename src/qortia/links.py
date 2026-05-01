"""
Cross-memory linking — Part 16i.

Three public functions:
  _find_similar_memories  — query top-N similar hindsight_memories for a given embedding
  _upsert_memory_links    — write bidirectional link rows (fire-and-forget safe)
  _expand_with_links      — expand a recall result set with linked memories
"""

from __future__ import annotations

import logging
from uuid import UUID

from app.qortia.models import RecallResult
from app.db import get_main_pool, tenant_transaction

logger = logging.getLogger(__name__)

LINK_SIMILARITY_THRESHOLD = 0.70
LINK_TOP_N = 3
EXPAND_TOP_RESULTS = 5
EXPAND_LIMIT_PER_RESULT = 2


async def _find_similar_memories(
    memory_id: UUID,
    embedding: list[float],
    tenant_id: UUID,
    agent_id: UUID,
    threshold: float = LINK_SIMILARITY_THRESHOLD,
    top_n: int = LINK_TOP_N,
) -> list[dict]:  # type: ignore[type-arg]
    """
    Return up to `top_n` hindsight_memories rows with cosine similarity >= threshold
    to `embedding`, excluding `memory_id` itself and short_term memories.
    Returns list of dicts with keys: id, similarity.
    """
    try:
        async with tenant_transaction(get_main_pool(), tenant_id, agent_id) as conn:
            rows = await conn.fetch(
                """
                SELECT id, 1 - (embedding <=> $1::vector) AS similarity
                FROM hindsight_memories
                WHERE agent_id = $2
                  AND id != $3
                  AND embedding IS NOT NULL
                  AND type != 'short_term'
                  AND tier = 'active'
                  AND (expires_at IS NULL OR expires_at > now())
                  AND 1 - (embedding <=> $1::vector) >= $4
                ORDER BY embedding <=> $1::vector
                LIMIT $5
                """,
                str(embedding),
                agent_id,
                memory_id,
                threshold,
                top_n,
            )
        return [{"id": r["id"], "similarity": float(r["similarity"])} for r in rows]
    except Exception as exc:
        logger.warning({"event": "memory_link_find_failed", "error": str(exc)})
        return []


async def _upsert_memory_links(
    memory_id: UUID,
    similar_rows: list[dict],  # type: ignore[type-arg]
    tenant_id: UUID,
) -> None:
    """
    Insert bidirectional link rows for each (memory_id, similar_id) pair.
    Uses ON CONFLICT DO NOTHING — safe to call multiple times.
    Failure is non-fatal.
    """
    if not similar_rows:
        return
    try:
        async with get_main_pool().acquire() as conn:
            for row in similar_rows:
                target_id = row["id"]
                similarity = row["similarity"]
                await conn.execute(
                    """
                    INSERT INTO memory_links (tenant_id, source_id, target_id, similarity)
                    VALUES ($1, $2, $3, $4), ($1, $3, $2, $4)
                    ON CONFLICT (source_id, target_id) DO NOTHING
                    """,
                    tenant_id,
                    memory_id,
                    target_id,
                    similarity,
                )
    except Exception as exc:
        logger.warning({"event": "memory_link_upsert_failed", "error": str(exc)})


async def _expand_with_links(
    results: list[RecallResult],
    tenant_id: UUID,
    agent_id: UUID,
    limit_per_result: int = EXPAND_LIMIT_PER_RESULT,
) -> list[RecallResult]:
    """
    For each result in the top EXPAND_TOP_RESULTS, fetch its linked memory IDs
    and append any linked memories not already in the result set.
    Linked results carry `linked_via` set to the source result's ID.
    Returns the original results list extended with linked memories (deduplicated).
    """
    if not results:
        return results

    existing_ids = {r.id for r in results}
    candidates = results[:EXPAND_TOP_RESULTS]
    source_ids = [r.id for r in candidates]

    try:
        async with tenant_transaction(get_main_pool(), tenant_id, agent_id) as conn:
            link_rows = await conn.fetch(
                """
                SELECT source_id, target_id
                FROM memory_links
                WHERE source_id = ANY($1::uuid[])
                  AND tenant_id = $2
                ORDER BY similarity DESC
                """,
                source_ids,
                tenant_id,
            )

            # Collect target IDs not already in the result set, bounded per source
            per_source: dict[str, list[str]] = {}
            new_ids: list[str] = []
            for row in link_rows:
                src = str(row["source_id"])
                tgt = str(row["target_id"])
                if tgt in existing_ids:
                    continue
                bucket = per_source.setdefault(src, [])
                if len(bucket) >= limit_per_result:
                    continue
                bucket.append(tgt)
                if tgt not in new_ids:
                    new_ids.append(tgt)

            if not new_ids:
                return results

            mem_rows = await conn.fetch(
                """
                SELECT id, type, content, importance, created_at,
                       recall_count, last_recalled_at
                FROM hindsight_memories
                WHERE id = ANY($1::uuid[])
                  AND tier = 'active'
                  AND (expires_at IS NULL OR expires_at > now())
                """,
                new_ids,
            )

    except Exception as exc:
        logger.warning({"event": "memory_link_expand_failed", "error": str(exc)})
        return results

    # Build a reverse map: target_id → source_id (first match wins)
    target_to_source: dict[str, str] = {}
    for src, targets in per_source.items():
        for tgt in targets:
            target_to_source.setdefault(tgt, src)

    linked_results: list[RecallResult] = []
    for row in mem_rows:
        rid = str(row["id"])
        if rid in existing_ids:
            continue
        r = RecallResult(
            id=rid,
            type=row["type"],
            scope="private",
            content=row["content"],
            importance=row.get("importance"),
            created_at=row["created_at"].isoformat(),
            linked_via=target_to_source.get(rid),
        )
        r._recall_count = row.get("recall_count", 0) or 0
        r._last_recalled_at = row.get("last_recalled_at")
        linked_results.append(r)

    return results + linked_results
