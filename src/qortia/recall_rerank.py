"""LLM rerank and entity BFS traversal for the recall pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import UUID

import yaml
from app.auth.models import AgentIdentity
from app.config import settings
from app.db import get_main_pool, tenant_transaction
from app.qortia.common import get_litellm_client
from app.qortia.models import RecallResult
from app.vault import get_litellm_key

logger = logging.getLogger(__name__)


async def _llm_rerank(
    query: str,
    results: list[RecallResult],
    agent: AgentIdentity,
) -> list[RecallResult]:
    if not results:
        return results
    try:
        async with get_main_pool().acquire() as conn:
            domain_md_raw = await conn.fetchval(
                "SELECT domain_md FROM auth.agents WHERE id = $1 AND tenant_id = $2",
                agent.agent_id,
                agent.tenant_id,
            )
        model = (yaml.safe_load(domain_md_raw or "{}") or {}).get("model") or settings.rerank_model
        litellm_key = await get_litellm_key(str(agent.tenant_id))

        numbered = "\n".join(f"{i+1}. [{r.type}] {r.content[:200]}" for i, r in enumerate(results))
        prompt = (
            f"Query: {query}\n\nResults:\n{numbered}\n\n"
            f"Return a JSON array of result numbers in order of relevance. "
            f"Example: [3, 1, 2]. Include all {len(results)} numbers."
        )

        async with asyncio.timeout(35.0):
            resp = await get_litellm_client().post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {litellm_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
                timeout=30.0,
            )
        raw_resp = resp.json()
        order = json.loads(raw_resp["choices"][0]["message"]["content"])
        usage = raw_resp.get("usage", {})
        logger.info(
            {
                "event": "qortia_llm_rerank",
                "the platform.tenant_id": str(agent.tenant_id),
                "model": model,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "result_count": len(results),
            }
        )
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
