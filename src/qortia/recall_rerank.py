"""LLM rerank and entity BFS traversal for the recall pipeline."""

from __future__ import annotations

import json
import logging
from uuid import UUID

from qortia import config
from qortia.auth import AgentIdentity, get_litellm_key
from qortia.chat import chat_completion
from qortia.db import get_main_pool, tenant_transaction
from qortia.models import RecallResult

logger = logging.getLogger(__name__)


async def _llm_rerank(
    query: str,
    results: list[RecallResult],
    agent: AgentIdentity,
) -> list[RecallResult]:
    if not results:
        return results
    model = config.settings.rerank_model
    if not model:
        # Not a failure — empty is the deliberate "no rerank model
        # configured" state (config.Settings.rerank_model's default),
        # distinct from a real call that fails. Returning here skips the
        # network round-trip that would otherwise always fail and, before
        # this guard, land in the broad except below as a misleading
        # rerank_failed warning for a state that isn't a failure at all.
        return results
    try:
        litellm_key = await get_litellm_key(str(agent.tenant_id))

        numbered = "\n".join(
            f"{i + 1}. [{r.type}] {r.content[:200]}" for i, r in enumerate(results)
        )
        prompt = (
            f"Query: {query}\n\nResults:\n{numbered}\n\n"
            f"Return a JSON array of result numbers in order of relevance. "
            f"Example: [3, 1, 2]. Include all {len(results)} numbers."
        )

        content = await chat_completion(
            model=model,
            prompt=prompt,
            litellm_key=litellm_key,
            timeout=30.0,
            json_mode=True,
            log_event="qortia_llm_rerank",
            tenant_id=str(agent.tenant_id),
        )
        order = json.loads(content)
        reranked = [results[i - 1] for i in order if 1 <= i <= len(results)]
        seen = {r.id for r in reranked}
        reranked += [r for r in results if r.id not in seen]
        return reranked
    except Exception as exc:
        logger.warning({"event": "rerank_failed", "error": str(exc)})
        return results


async def _bfs_entity_traversal(
    query_embedding: list[float],
    seed_entity_ids: list[UUID],
    tenant_id: UUID,
    agent_id: UUID,
    max_depth: int = 2,
    decay: float = 0.5,
) -> dict[str, float]:
    """
    BFS from seed entities via co-occurrence in linked_memory_ids.
    Returns {memory_id: boost_score} for memories reachable within max_depth hops.
    Each hop decays the boost score by `decay`.
    """
    if not seed_entity_ids:
        return {}

    visited: set[UUID] = set(seed_entity_ids)
    frontier = list(seed_entity_ids)
    boosts: dict[str, float] = {}
    current_decay = decay

    for _ in range(max_depth):
        if not frontier:
            break
        async with tenant_transaction(get_main_pool(), tenant_id, agent_id) as conn:
            next_entities = await conn.fetch(
                """
                SELECT e2.id, e2.linked_memory_ids,
                       1 - (e2.embedding <=> $1::vector) AS similarity
                FROM qortia_entities e1
                JOIN qortia_entities e2
                  ON e2.linked_memory_ids && e1.linked_memory_ids
                 AND e2.id != ALL($2::uuid[])
                WHERE e1.id = ANY($2::uuid[])
                  AND e1.tenant_id = $3
                  AND e2.tenant_id = $3
                  AND e2.embedding IS NOT NULL
                  AND 1 - (e2.embedding <=> $1::vector) >= 0.3
                LIMIT 10
                """,
                str(query_embedding),
                list(visited),
                tenant_id,
            )
        next_frontier: list[UUID] = []
        for row in next_entities:
            eid = row["id"]
            if eid not in visited:
                visited.add(eid)
                next_frontier.append(eid)
                sim = float(row["similarity"])
                n = len(row["linked_memory_ids"])
                hop_boost = sim * 0.5 / (1 + 0.001 * (n - 1) ** 2) * current_decay
                for mid in row["linked_memory_ids"]:
                    mid_str = str(mid)
                    boosts[mid_str] = max(boosts.get(mid_str, 0.0), hop_boost)
        frontier = next_frontier
        current_decay *= decay

    return boosts
