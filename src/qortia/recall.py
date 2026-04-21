from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from uuid import UUID

import yaml
from fastapi import APIRouter, Depends, HTTPException

from app.auth.middleware import require_agent
from app.auth.models import AgentIdentity
from app.qortia.models import RecallRequest, RecallResponse, RecallResult
from app.qortia.reflect import EMBEDDING_MODEL, get_litellm_client
from app.db import get_main_pool, tenant_transaction
from app.vault import get_litellm_key

logger = logging.getLogger(__name__)
router = APIRouter()

RRF_K = 60


# ── Dynamic importance (Q95) ─────────────────────────────────

def dynamic_importance(
    base_importance: float,
    recall_count: int,
    last_recalled_at: datetime | None,
) -> float:
    frequency_boost = math.log1p(recall_count) / 10.0
    recency_boost = 0.0
    if last_recalled_at:
        days_since = (datetime.now(timezone.utc) - last_recalled_at).days
        recency_boost = max(0.0, 1.0 - (days_since / 30.0)) * 0.2
    return min(1.0, base_importance + frequency_boost + recency_boost)


# ── Entity filter helper ─────────────────────────────────────

def _entity_filter_clause(
    entities: list[str] | None,
    base_param: int,
) -> tuple[str, list]:  # type: ignore[type-arg]
    if not entities:
        return "", []
    return f"AND entities ?| ${base_param}::text[]", [entities]


# ── Result builder ───────────────────────────────────────────

def _to_result(row: dict, scope: str) -> RecallResult:  # type: ignore[type-arg]
    r = RecallResult(
        id=str(row["id"]),
        type=row.get("type", "knowledge"),
        scope=scope,  # type: ignore[arg-type]
        content=row["content"],
        importance=row.get("importance"),
        created_at=row["created_at"].isoformat(),
    )
    r._recall_count = row.get("recall_count", 0) or 0
    r._last_recalled_at = row.get("last_recalled_at")
    r._score = float(row.get("score") or row.get("rank") or 0.0)
    return r


# ── Embed query ──────────────────────────────────────────────

async def _embed_query(query: str, tenant_id: UUID) -> list[float] | None:
    try:
        litellm_key = await get_litellm_key(str(tenant_id))
        resp = await get_litellm_client().post(
            "/embeddings",
            headers={"Authorization": f"Bearer {litellm_key}"},
            json={"model": EMBEDDING_MODEL, "input": query},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as exc:
        logger.warning({"event": "recall_embed_failed", "error": str(exc)})
        return None


# ── Type-routed strategies ───────────────────────────────────

async def _recall_decisions(body: RecallRequest, agent: AgentIdentity) -> list[RecallResult]:
    if body.scope not in ("private", "all"):
        return []
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=3)
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        rows = await conn.fetch(f"""
            SELECT id, content, importance, created_at, recall_count, last_recalled_at,
                   ts_rank_cd(content_tsv, plainto_tsquery('english', $1)) AS rank
            FROM hindsight_memories
            WHERE agent_id = $2
              AND type = 'decision'
              AND content_tsv @@ plainto_tsquery('english', $1)
              {entity_clause}
            ORDER BY created_at DESC, rank DESC
            LIMIT 10
        """, body.query, agent.agent_id, *entity_params)
    return [_to_result(dict(r), "private") for r in rows]


async def _recall_lessons(body: RecallRequest, agent: AgentIdentity) -> list[RecallResult]:
    if body.scope not in ("private", "all"):
        return []
    query_embedding = await _embed_query(body.query, agent.tenant_id)
    if query_embedding is None:
        return []
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=3)
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        rows = await conn.fetch(f"""
            SELECT id, content, importance, created_at, recall_count, last_recalled_at,
                   1 - (embedding <=> $1::vector) AS score
            FROM hindsight_memories
            WHERE agent_id = $2
              AND type = 'lesson'
              AND embedding IS NOT NULL
              {entity_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT 10
        """, query_embedding, agent.agent_id, *entity_params)
    return [_to_result(dict(r), "private") for r in rows if (r.get("score") or 0) >= 0.35]


async def _recall_episodic(body: RecallRequest, agent: AgentIdentity) -> list[RecallResult]:
    if body.scope not in ("private", "all"):
        return []
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=3)
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        rows = await conn.fetch(f"""
            SELECT id, content, importance, created_at, recall_count, last_recalled_at,
                   ts_rank_cd(content_tsv, plainto_tsquery('english', $1)) AS rank
            FROM hindsight_memories
            WHERE agent_id = $2
              AND type = 'episodic'
              AND (
                content_tsv @@ plainto_tsquery('english', $1)
                OR created_at > now() - interval '7 days'
              )
              {entity_clause}
            ORDER BY created_at DESC, rank DESC
            LIMIT 10
        """, body.query, agent.agent_id, *entity_params)
    return [_to_result(dict(r), "private") for r in rows]


# ── Full hybrid pipeline ─────────────────────────────────────

async def _bm25_private(body: RecallRequest, agent: AgentIdentity) -> list[RecallResult]:
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=4)
    type_clause = f"AND type = '{body.type}'" if body.type else ""
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        rows = await conn.fetch(f"""
            SELECT id, type, content, importance, created_at, recall_count, last_recalled_at,
                   ts_rank_cd(content_tsv, plainto_tsquery('english', $1)) AS rank
            FROM hindsight_memories
            WHERE agent_id = $2
              AND content_tsv @@ plainto_tsquery('english', $1)
              {type_clause}
              {entity_clause}
            ORDER BY rank DESC LIMIT 20
        """, body.query, agent.agent_id, *entity_params)
    return [_to_result(dict(r), "private") for r in rows]


async def _vector_private(body: RecallRequest, agent: AgentIdentity, qe: list[float]) -> list[RecallResult]:
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=3)
    type_clause = f"AND type = '{body.type}'" if body.type else ""
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        rows = await conn.fetch(f"""
            SELECT id, type, content, importance, created_at, recall_count, last_recalled_at,
                   1 - (embedding <=> $1::vector) AS score
            FROM hindsight_memories
            WHERE agent_id = $2
              AND embedding IS NOT NULL
              {type_clause}
              {entity_clause}
            ORDER BY embedding <=> $1::vector LIMIT 20
        """, qe, agent.agent_id, *entity_params)
    return [_to_result(dict(r), "private") for r in rows]


async def _bm25_org(body: RecallRequest, agent: AgentIdentity) -> list[RecallResult]:
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=3)
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        rows = await conn.fetch(f"""
            SELECT id, type, content, NULL::float AS importance,
                   created_at, recall_count, last_recalled_at,
                   ts_rank_cd(content_tsv, plainto_tsquery('english', $1)) AS rank
            FROM org_memory
            WHERE tenant_id = $2
              AND content_tsv @@ plainto_tsquery('english', $1)
              {entity_clause}
            ORDER BY rank DESC LIMIT 10
        """, body.query, agent.tenant_id, *entity_params)
    return [_to_result(dict(r), "org") for r in rows]


async def _vector_org(body: RecallRequest, agent: AgentIdentity, qe: list[float]) -> list[RecallResult]:
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=3)
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        rows = await conn.fetch(f"""
            SELECT id, type, content, NULL::float AS importance,
                   created_at, recall_count, last_recalled_at,
                   1 - (embedding <=> $1::vector) AS score
            FROM org_memory
            WHERE tenant_id = $2
              AND embedding IS NOT NULL
              {entity_clause}
            ORDER BY embedding <=> $1::vector LIMIT 10
        """, qe, agent.tenant_id, *entity_params)
    return [_to_result(dict(r), "org") for r in rows]


async def _bm25_knowledge(body: RecallRequest, agent: AgentIdentity) -> list[RecallResult]:
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        rows = await conn.fetch("""
            SELECT id, 'knowledge' AS type, content, NULL::float AS importance,
                   created_at, recall_count, last_recalled_at,
                   ts_rank_cd(index_tsv, plainto_tsquery('english', $1)) AS rank
            FROM org_knowledge
            WHERE tenant_id = $2
              AND index_tsv @@ plainto_tsquery('english', $1)
            ORDER BY rank DESC LIMIT 16
        """, body.query, agent.tenant_id)
    return [_to_result(dict(r), "knowledge") for r in rows]


async def _vector_knowledge(body: RecallRequest, agent: AgentIdentity, qe: list[float]) -> list[RecallResult]:
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        rows = await conn.fetch("""
            SELECT id, 'knowledge' AS type, content, NULL::float AS importance,
                   created_at, recall_count, last_recalled_at,
                   1 - (embedding <=> $1::vector) AS score
            FROM org_knowledge
            WHERE tenant_id = $2
              AND embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector LIMIT 16
        """, qe, agent.tenant_id)
    results = []
    for r in rows:
        result = _to_result(dict(r), "knowledge")
        result._embedding = qe  # used by MMR
        results.append(result)
    return results


# ── RRF fusion ───────────────────────────────────────────────

def _rrf_fuse(results: list[RecallResult]) -> list[RecallResult]:
    if not results:
        return []
    scores: dict[str, float] = {}
    by_id: dict[str, RecallResult] = {}
    seen_positions: dict[str, int] = {}

    for result in results:
        key = result.id
        pos = seen_positions.get(key, 0) + 1
        seen_positions[key] = pos
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + pos)
        by_id[key] = result

    def final_score(rid: str) -> float:
        r = by_id[rid]
        imp = dynamic_importance(
            base_importance=r.importance if r.importance is not None else 0.5,
            recall_count=r._recall_count,
            last_recalled_at=r._last_recalled_at,
        )
        return scores[rid] * imp

    return [by_id[rid] for rid in sorted(scores.keys(), key=final_score, reverse=True)]


# ── MMR ──────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _mmr(
    query_embedding: list[float],
    candidates: list[RecallResult],
    lambda_: float = 0.5,
    dedup_threshold: float = 0.85,
    min_score: float = 0.35,
    k: int = 4,
) -> list[RecallResult]:
    eligible = [c for c in candidates if c._score >= min_score]
    if not eligible:
        return []

    selected: list[RecallResult] = []
    remaining = list(eligible)

    while remaining and len(selected) < k:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        best_candidate = None
        best_mmr_score = -1.0

        for candidate in remaining:
            if not candidate._embedding:
                continue
            relevance = _cosine(query_embedding, candidate._embedding)
            redundancy = max(
                _cosine(candidate._embedding, s._embedding)
                for s in selected if s._embedding
            ) if selected else 0.0
            if redundancy >= dedup_threshold:
                continue
            score = lambda_ * relevance - (1 - lambda_) * redundancy
            if score > best_mmr_score:
                best_mmr_score = score
                best_candidate = candidate

        if best_candidate is None:
            break
        remaining.remove(best_candidate)
        selected.append(best_candidate)

    return selected


# ── LLM re-rank ──────────────────────────────────────────────

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
                agent.agent_id, agent.tenant_id,
            )
        model = yaml.safe_load(domain_md_raw).get("model", "brain-sonnet")
        litellm_key = await get_litellm_key(str(agent.tenant_id))

        numbered = "\n".join(
            f"{i+1}. [{r.type}] {r.content[:200]}" for i, r in enumerate(results)
        )
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
        order = json.loads(resp.json()["choices"][0]["message"]["content"])
        reranked = [results[i - 1] for i in order if 1 <= i <= len(results)]
        seen = {r.id for r in reranked}
        reranked += [r for r in results if r.id not in seen]

        # Record cost — fire-and-forget
        usage = resp.json().get("usage", {})
        if usage:
            from app.qortia.reflect import _record_llm_cost
            asyncio.create_task(_record_llm_cost(
                agent_id=agent.agent_id,
                tenant_id=agent.tenant_id,
                model=model,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
            ))

        return reranked
    except Exception as exc:
        logger.warning({"event": "rerank_failed", "error": str(exc)})
        return results


# ── Access tracking ──────────────────────────────────────────

async def _record_recall_access(results: list[RecallResult]) -> None:
    by_table: dict[str, list[str]] = defaultdict(list)
    for r in results:
        table = (
            "hindsight_memories" if r.scope == "private"
            else "org_memory" if r.scope == "org"
            else "org_knowledge"
        )
        by_table[table].append(r.id)
    try:
        async with get_main_pool().acquire() as conn:
            for table, ids in by_table.items():
                await conn.execute(f"""
                    UPDATE {table}
                    SET recall_count = recall_count + 1,
                        last_recalled_at = now()
                    WHERE id = ANY($1::uuid[])
                """, ids)
    except Exception as exc:
        logger.warning({"event": "recall_access_tracking_failed", "error": str(exc)})


# ── POST /v1/recall ──────────────────────────────────────────

@router.post("/v1/recall", response_model=RecallResponse)
async def recall(
    body: RecallRequest,
    agent: AgentIdentity = Depends(require_agent),
) -> RecallResponse:
    from app.qortia.remember import assert_agent_active
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)

    results: list[RecallResult] = []

    if body.type == "decision":
        results = await _recall_decisions(body, agent)
    elif body.type == "lesson":
        results = await _recall_lessons(body, agent)
    elif body.type == "episodic":
        results = await _recall_episodic(body, agent)
    else:
        # Full hybrid pipeline
        query_embedding = await _embed_query(body.query, agent.tenant_id)
        tasks = []

        if body.scope in ("private", "all"):
            tasks.append(_bm25_private(body, agent))
            if query_embedding:
                tasks.append(_vector_private(body, agent, query_embedding))

        if body.scope in ("org", "all"):
            tasks.append(_bm25_org(body, agent))
            if query_embedding:
                tasks.append(_vector_org(body, agent, query_embedding))

        if body.scope in ("knowledge", "all"):
            tasks.append(_bm25_knowledge(body, agent))
            if query_embedding:
                tasks.append(_vector_knowledge(body, agent, query_embedding))

        result_sets = await asyncio.gather(*tasks, return_exceptions=True)

        memory_results: list[RecallResult] = []
        knowledge_candidates: list[RecallResult] = []

        for rs in result_sets:
            if isinstance(rs, Exception):
                logger.warning({"event": "recall_search_error", "error": str(rs)})
                continue
            for r in rs:
                if r.scope == "knowledge":
                    knowledge_candidates.append(r)
                else:
                    memory_results.append(r)

        fused_memory = _rrf_fuse(memory_results)

        if query_embedding and knowledge_candidates:
            knowledge_results = _mmr(
                query_embedding=query_embedding,
                candidates=knowledge_candidates,
            )
        else:
            knowledge_results = knowledge_candidates[:4]

        results = fused_memory + knowledge_results

    if body.rerank or len(results) < 3:
        results = await _llm_rerank(body.query, results, agent)

    asyncio.create_task(_record_recall_access(results))

    return RecallResponse(results=results)
