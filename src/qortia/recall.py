from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from uuid import UUID

from fastapi import APIRouter, Depends, Header

from app.auth.middleware import require_agent
from app.auth.models import AgentIdentity
from app.qortia.models import RecallRequest, RecallResponse, RecallResult
from app.qortia.common import EMBEDDING_MODEL, get_litellm_client
from app.qortia.embedding_cache import get_cached_embedding, put_cached_embedding
from app.db import get_main_pool, tenant_transaction
from app.vault import get_litellm_key
from app.qortia.remember import _fetch_agent_clearance
from app.qortia.recall_helpers import (
    SEARCH_FETCH_MULTIPLIER,
    PRIVATE_RESULT_LIMIT,
    ORG_RESULT_LIMIT,
    KNOWLEDGE_RESULT_LIMIT,
    _bm25_normalization,
    _entity_filter_clause,
    _type_filter_clause,
    _temporal_filter_clause,
    _lang_filter_clause,
    _to_result,
    _rrf_fuse,
    _keyword_boost,
    _mmr,
    _sort_by_importance,
)
from app.qortia.recall_rerank import _llm_rerank, _bfs_entity_traversal

logger = logging.getLogger(__name__)
router = APIRouter()


async def _embed_query(
    query: str, tenant_id: UUID, lang: str = "en"
) -> list[float] | None:
    tid = str(tenant_id)
    effective_lang = lang or "en"

    # Cache lookup — avoids redundant LiteLLM calls for repeated queries
    cached = get_cached_embedding(query, tid, effective_lang)
    if cached is not None:
        return cached

    try:
        litellm_key = await get_litellm_key(tid)
        resp = await get_litellm_client().post(
            "/embeddings",
            headers={"Authorization": f"Bearer {litellm_key}"},
            json={"model": EMBEDDING_MODEL, "input": query},
            timeout=10.0,
        )
        resp.raise_for_status()
        embedding: list[float] = resp.json()["data"][0]["embedding"]
        put_cached_embedding(query, tid, effective_lang, embedding)
        return embedding
    except Exception as exc:
        logger.warning({"event": "recall_embed_failed", "error": str(exc)})
        try:
            from app.telemetry.instruments import qortia_recall_degraded

            qortia_recall_degraded.add(
                1, {"reason": "embed_failed", "the platform.tenant_id": tid}
            )
        except Exception:
            pass
        return None


# ── Type-routed strategies ───────────────────────────────────


async def _recall_decisions(
    body: RecallRequest, agent: AgentIdentity
) -> list[RecallResult]:
    if body.scope not in ("private", "all", "archive"):
        return []
    tier_clause = (
        "AND tier = 'archive'" if body.scope == "archive" else "AND tier = 'active'"
    )
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=3)
    async with tenant_transaction(
        get_main_pool(), agent.tenant_id, agent.agent_id
    ) as conn:
        norm = _bm25_normalization(body.query)
        rows = await conn.fetch(
            f"""
            SELECT id, content, importance, created_at, recall_count, last_recalled_at,
                   confidence_multiplier,
                   ts_rank_cd(content_tsv, plainto_tsquery('simple', $1), {norm}) AS rank
            FROM hindsight_memories
            WHERE agent_id = $2
              AND type = 'decision'
              AND content_tsv @@ plainto_tsquery('simple', $1)
              {tier_clause}
              AND (expires_at IS NULL OR expires_at > now())
              AND (valid_until IS NULL OR valid_until > now())
              {entity_clause}
            ORDER BY rank DESC, created_at DESC
            LIMIT 10
        """,
            body.query,
            agent.agent_id,
            *entity_params,
        )
    return _sort_by_importance([_to_result(dict(r), "private") for r in rows])


async def _recall_lessons(
    body: RecallRequest, agent: AgentIdentity
) -> list[RecallResult]:
    if body.scope not in ("private", "all", "archive"):
        return []
    tier_clause = (
        "AND tier = 'archive'" if body.scope == "archive" else "AND tier = 'active'"
    )
    query_embedding = await _embed_query(
        body.query, agent.tenant_id, lang=body.lang or "en"
    )
    if query_embedding is None:
        return []
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=3)
    async with tenant_transaction(
        get_main_pool(), agent.tenant_id, agent.agent_id
    ) as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, content, importance, created_at, recall_count, last_recalled_at,
                   confidence_multiplier,
                   1 - (embedding <=> $1::vector) AS score
            FROM hindsight_memories
            WHERE agent_id = $2
              AND type = 'lesson'
              AND embedding IS NOT NULL
              {tier_clause}
              AND (expires_at IS NULL OR expires_at > now())
              AND (valid_until IS NULL OR valid_until > now())
              {entity_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT 10
        """,
            str(query_embedding),
            agent.agent_id,
            *entity_params,
        )
    return _sort_by_importance(
        [_to_result(dict(r), "private") for r in rows if (r.get("score") or 0) >= 0.35]
    )


async def _recall_episodic(
    body: RecallRequest, agent: AgentIdentity
) -> list[RecallResult]:
    if body.scope not in ("private", "all", "archive"):
        return []
    tier_clause = (
        "AND tier = 'archive'" if body.scope == "archive" else "AND tier = 'active'"
    )
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=3)
    async with tenant_transaction(
        get_main_pool(), agent.tenant_id, agent.agent_id
    ) as conn:
        norm = _bm25_normalization(body.query)
        rows = await conn.fetch(
            f"""
            SELECT id, content, importance, created_at, recall_count, last_recalled_at,
                   confidence_multiplier,
                   ts_rank_cd(content_tsv, plainto_tsquery('simple', $1), {norm}) AS rank
            FROM hindsight_memories
            WHERE agent_id = $2
              AND type = 'episodic'
              AND (
                content_tsv @@ plainto_tsquery('simple', $1)
                OR created_at > now() - interval '7 days'
              )
              {tier_clause}
              AND (expires_at IS NULL OR expires_at > now())
              AND (valid_until IS NULL OR valid_until > now())
              {entity_clause}
            ORDER BY
                CASE WHEN content_tsv @@ plainto_tsquery('simple', $1) THEN 0 ELSE 1 END,
                rank DESC,
                created_at DESC
            LIMIT 10
        """,
            body.query,
            agent.agent_id,
            *entity_params,
        )
    return _sort_by_importance([_to_result(dict(r), "private") for r in rows])


async def _recall_short_term(
    body: RecallRequest, agent: AgentIdentity
) -> list[RecallResult]:
    # short_term memories are always in active tier — archive scope is meaningless
    if body.scope == "archive":
        raise ValueError("scope='archive' is not supported for type='short_term'")
    if body.scope not in ("private", "all"):
        return []
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=3)
    async with tenant_transaction(
        get_main_pool(), agent.tenant_id, agent.agent_id
    ) as conn:
        norm = _bm25_normalization(body.query)
        rows = await conn.fetch(
            f"""
            SELECT id, content, importance, created_at, recall_count, last_recalled_at,
                   confidence_multiplier,
                   ts_rank_cd(content_tsv, plainto_tsquery('simple', $1), {norm}) AS rank
            FROM hindsight_memories
            WHERE agent_id = $2
              AND type = 'short_term'
              AND content_tsv @@ plainto_tsquery('simple', $1)
              AND tier = 'active'
              AND (expires_at IS NULL OR expires_at > now())
              AND (valid_until IS NULL OR valid_until > now())
              {entity_clause}
            ORDER BY rank DESC, created_at DESC
            LIMIT 10
        """,
            body.query,
            agent.agent_id,
            *entity_params,
        )
    return _sort_by_importance([_to_result(dict(r), "private") for r in rows])


# ── Full hybrid pipeline ─────────────────────────────────────


async def _bm25_private(
    body: RecallRequest, agent: AgentIdentity
) -> list[RecallResult]:
    tier_clause = (
        "AND tier = 'archive'" if body.scope == "archive" else "AND tier = 'active'"
    )
    # type param is $3; entity param shifts to $4 when type filter is active
    type_clause, type_params = _type_filter_clause(body.type, param=3)
    entity_clause, entity_params = _entity_filter_clause(
        body.entities, base_param=3 + len(type_params)
    )
    temporal_clause, temporal_params = _temporal_filter_clause(
        body.as_of, param=3 + len(type_params) + len(entity_params)
    )
    lang_clause, lang_params = _lang_filter_clause(
        body.lang,
        param=3 + len(type_params) + len(entity_params) + len(temporal_params),
    )
    async with tenant_transaction(
        get_main_pool(), agent.tenant_id, agent.agent_id
    ) as conn:
        norm = _bm25_normalization(body.query)
        rows = await conn.fetch(
            f"""
            SELECT id, type, content, importance, created_at, recall_count, last_recalled_at,
                   confidence_multiplier, valid_from, valid_until,
                   ts_rank_cd(content_tsv, plainto_tsquery('simple', $1), {norm}) AS rank
            FROM hindsight_memories
            WHERE agent_id = $2
              AND content_tsv @@ plainto_tsquery('simple', $1)
              AND type != 'short_term'
              {tier_clause}
              AND (expires_at IS NULL OR expires_at > now())
              AND (valid_until IS NULL OR valid_until > now())
              {type_clause}
              {entity_clause}
              {temporal_clause}
              {lang_clause}
            ORDER BY rank DESC LIMIT {PRIVATE_RESULT_LIMIT * SEARCH_FETCH_MULTIPLIER}
        """,
            body.query,
            agent.agent_id,
            *type_params,
            *entity_params,
            *temporal_params,
            *lang_params,
        )
    return [_to_result(dict(r), "private") for r in rows]


async def _vector_private(
    body: RecallRequest, agent: AgentIdentity, qe: list[float]
) -> list[RecallResult]:
    tier_clause = (
        "AND tier = 'archive'" if body.scope == "archive" else "AND tier = 'active'"
    )
    # type param is $3; entity param shifts to $4 when type filter is active
    type_clause, type_params = _type_filter_clause(body.type, param=3)
    entity_clause, entity_params = _entity_filter_clause(
        body.entities, base_param=3 + len(type_params)
    )
    temporal_clause, temporal_params = _temporal_filter_clause(
        body.as_of, param=3 + len(type_params) + len(entity_params)
    )
    lang_clause, lang_params = _lang_filter_clause(
        body.lang,
        param=3 + len(type_params) + len(entity_params) + len(temporal_params),
    )
    async with tenant_transaction(
        get_main_pool(), agent.tenant_id, agent.agent_id
    ) as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, type, content, importance, created_at, recall_count, last_recalled_at,
                   confidence_multiplier, valid_from, valid_until,
                   1 - (embedding <=> $1::vector) AS score
            FROM hindsight_memories
            WHERE agent_id = $2
              AND embedding IS NOT NULL
              AND type != 'short_term'
              {tier_clause}
              AND (expires_at IS NULL OR expires_at > now())
              AND (valid_until IS NULL OR valid_until > now())
              {type_clause}
              {entity_clause}
              {temporal_clause}
              {lang_clause}
            ORDER BY embedding <=> $1::vector LIMIT {PRIVATE_RESULT_LIMIT * SEARCH_FETCH_MULTIPLIER}
        """,
            str(qe),
            agent.agent_id,
            *type_params,
            *entity_params,
            *temporal_params,
            *lang_params,
        )
    return [_to_result(dict(r), "private") for r in rows]


async def _bm25_org(
    body: RecallRequest,
    agent: AgentIdentity,
    clearance_order: int,
    agent_division: str,
) -> list[RecallResult]:
    # Explicit clearance predicates: RLS tenant_visibility_read is bypassed by the
    # platform_write policy for qortia_platform (ADR-080 §0.8 fix).
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=5)
    lang_clause, lang_params = _lang_filter_clause(
        body.lang, param=5 + len(entity_params)
    )
    async with tenant_transaction(
        get_main_pool(),
        agent.tenant_id,
        agent.agent_id,
        memory_clearance_order=clearance_order,
        agent_division=agent_division,
    ) as conn:
        norm = _bm25_normalization(body.query)
        rows = await conn.fetch(
            f"""
            SELECT id, type, content, NULL::float AS importance,
                   created_at, recall_count, last_recalled_at,
                   confidence_multiplier,
                   ts_rank_cd(content_tsv, plainto_tsquery('simple', $1), {norm}) AS rank
            FROM org_memory
            WHERE tenant_id = $2
              AND content_tsv @@ plainto_tsquery('simple', $1)
              AND ($3 >= (SELECT level_order FROM tenant_clearance_levels
                           WHERE tenant_id = org_memory.tenant_id
                             AND level_name = org_memory.min_clearance)
                   OR NOT EXISTS (SELECT 1 FROM tenant_clearance_levels
                                  WHERE tenant_id = org_memory.tenant_id))
              AND ($4 = ANY(audience) OR 'all' = ANY(audience))
              AND (valid_until IS NULL OR valid_until > now())
              {entity_clause}
              {lang_clause}
            ORDER BY rank DESC LIMIT {ORG_RESULT_LIMIT * SEARCH_FETCH_MULTIPLIER}
        """,
            body.query,
            agent.tenant_id,
            clearance_order,
            agent_division,
            *entity_params,
            *lang_params,
        )
    return [_to_result(dict(r), "org") for r in rows]


async def _vector_org(
    body: RecallRequest,
    agent: AgentIdentity,
    qe: list[float],
    clearance_order: int,
    agent_division: str,
) -> list[RecallResult]:
    entity_clause, entity_params = _entity_filter_clause(body.entities, base_param=5)
    lang_clause, lang_params = _lang_filter_clause(
        body.lang, param=5 + len(entity_params)
    )
    async with tenant_transaction(
        get_main_pool(),
        agent.tenant_id,
        agent.agent_id,
        memory_clearance_order=clearance_order,
        agent_division=agent_division,
    ) as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, type, content, NULL::float AS importance,
                   created_at, recall_count, last_recalled_at,
                   confidence_multiplier,
                   1 - (embedding <=> $1::vector) AS score
            FROM org_memory
            WHERE tenant_id = $2
              AND embedding IS NOT NULL
              AND ($3 >= (SELECT level_order FROM tenant_clearance_levels
                           WHERE tenant_id = org_memory.tenant_id
                             AND level_name = org_memory.min_clearance)
                   OR NOT EXISTS (SELECT 1 FROM tenant_clearance_levels
                                  WHERE tenant_id = org_memory.tenant_id))
              AND ($4 = ANY(audience) OR 'all' = ANY(audience))
              AND (valid_until IS NULL OR valid_until > now())
              {entity_clause}
              {lang_clause}
            ORDER BY embedding <=> $1::vector LIMIT {ORG_RESULT_LIMIT * SEARCH_FETCH_MULTIPLIER}
        """,
            str(qe),
            agent.tenant_id,
            clearance_order,
            agent_division,
            *entity_params,
            *lang_params,
        )
    return [_to_result(dict(r), "org") for r in rows]


async def _bm25_knowledge(
    body: RecallRequest,
    agent: AgentIdentity,
    clearance_order: int,
    agent_division: str,
) -> list[RecallResult]:
    async with tenant_transaction(
        get_main_pool(),
        agent.tenant_id,
        agent.agent_id,
        memory_clearance_order=clearance_order,
        agent_division=agent_division,
    ) as conn:
        norm = _bm25_normalization(body.query)
        rows = await conn.fetch(
            f"""
            SELECT id, 'knowledge' AS type, content, NULL::float AS importance,
                   created_at, recall_count, last_recalled_at,
                   ts_rank_cd(index_tsv, plainto_tsquery('simple', $1), {norm}) AS rank
            FROM org_knowledge
            WHERE tenant_id = $2
              AND index_tsv @@ plainto_tsquery('simple', $1)
              AND ($3 >= (SELECT level_order FROM tenant_clearance_levels
                           WHERE tenant_id = org_knowledge.tenant_id
                             AND level_name = org_knowledge.min_clearance)
                   OR NOT EXISTS (SELECT 1 FROM tenant_clearance_levels
                                  WHERE tenant_id = org_knowledge.tenant_id))
              AND ($4 = ANY(audience) OR 'all' = ANY(audience))
            ORDER BY rank DESC LIMIT {KNOWLEDGE_RESULT_LIMIT * SEARCH_FETCH_MULTIPLIER}
        """,
            body.query,
            agent.tenant_id,
            clearance_order,
            agent_division,
        )
    return [_to_result(dict(r), "knowledge") for r in rows]


async def _vector_knowledge(
    body: RecallRequest,
    agent: AgentIdentity,
    qe: list[float],
    clearance_order: int,
    agent_division: str,
) -> list[RecallResult]:
    async with tenant_transaction(
        get_main_pool(),
        agent.tenant_id,
        agent.agent_id,
        memory_clearance_order=clearance_order,
        agent_division=agent_division,
    ) as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, 'knowledge' AS type, content, NULL::float AS importance,
                   created_at, recall_count, last_recalled_at,
                   1 - (embedding <=> $1::vector) AS score
            FROM org_knowledge
            WHERE tenant_id = $2
              AND embedding IS NOT NULL
              AND ($3 >= (SELECT level_order FROM tenant_clearance_levels
                           WHERE tenant_id = org_knowledge.tenant_id
                             AND level_name = org_knowledge.min_clearance)
                   OR NOT EXISTS (SELECT 1 FROM tenant_clearance_levels
                                  WHERE tenant_id = org_knowledge.tenant_id))
              AND ($4 = ANY(audience) OR 'all' = ANY(audience))
            ORDER BY embedding <=> $1::vector LIMIT {KNOWLEDGE_RESULT_LIMIT * SEARCH_FETCH_MULTIPLIER}
        """,
            str(qe),
            agent.tenant_id,
            clearance_order,
            agent_division,
        )
    results = []
    for r in rows:
        result = _to_result(dict(r), "knowledge")
        result._embedding = qe  # used by MMR
        results.append(result)
    return results


# ── RRF fusion ───────────────────────────────────────────────


# ── Access tracking ──────────────────────────────────────────


async def _record_recall_access(
    results: list[RecallResult],
    tenant_id: UUID,
    agent_id: UUID,
    memory_clearance_order: int = 2,
    agent_division: str = "all",
) -> None:
    by_table: dict[str, list[str]] = defaultdict(list)
    for r in results:
        table = (
            "hindsight_memories"
            if r.scope == "private"
            else "org_memory"
            if r.scope == "org"
            else "org_knowledge"
        )
        by_table[table].append(r.id)
    try:
        async with tenant_transaction(
            get_main_pool(),
            tenant_id,
            agent_id,
            memory_clearance_order=memory_clearance_order,
            agent_division=agent_division,
        ) as conn:
            for table, ids in by_table.items():
                await conn.execute(
                    f"""
                    UPDATE {table}
                    SET recall_count = recall_count + 1,
                        last_recalled_at = now()
                    WHERE id = ANY($1::uuid[])
                      AND tenant_id = $2
                    """,
                    ids,
                    tenant_id,
                )
    except Exception as exc:
        logger.warning({"event": "recall_access_tracking_failed", "error": str(exc)})
        try:
            from app.telemetry.instruments import qortia_recall_degraded

            qortia_recall_degraded.add(
                1, {"reason": "access_failed", "the platform.tenant_id": str(tenant_id)}
            )
        except Exception:
            pass


async def _record_work_order_outcome(
    work_order_id: UUID,
    tenant_id: UUID,
    agent_id: UUID,
    outcome: str,  # "SUCCESS" | "MINOR_FAILURE" | "CRITICAL_FAILURE"
    memory_clearance_order: int = 2,
    agent_division: str = "all",
) -> None:
    """Record WO outcome and decay confidence_multiplier on implicated memories (ADR-125 Phase 2)."""
    multipliers = {"SUCCESS": 1.05, "MINOR_FAILURE": 0.85, "CRITICAL_FAILURE": 0.60}
    multiplier = multipliers.get(outcome, 1.0)

    try:
        async with tenant_transaction(
            get_main_pool(),
            tenant_id,
            agent_id,
            memory_clearance_order=memory_clearance_order,
            agent_division=agent_division,
        ) as conn:
            # Fetch implicated memory IDs from session reads
            rows = await conn.fetch(
                "SELECT DISTINCT memory_id FROM qortia_session_reads WHERE work_order_id = $1 AND tenant_id = $2",
                work_order_id,
                tenant_id,
            )
            memory_ids = [str(r["memory_id"]) for r in rows]
            memory_count = len(memory_ids)

            # Update confidence_multiplier on each implicated memory (floor 0.10, cap 1.0)
            if memory_ids:
                await conn.execute(
                    """
                    UPDATE hindsight_memories
                    SET confidence_multiplier = GREATEST(0.10, LEAST(1.0, confidence_multiplier * $1))
                    WHERE id = ANY($2::uuid[]) AND tenant_id = $3
                    """,
                    multiplier,
                    memory_ids,
                    tenant_id,
                )

            # Insert outcome record (ON CONFLICT DO NOTHING — idempotent)
            await conn.execute(
                """
                INSERT INTO qortia_outcome_records
                    (tenant_id, agent_id, work_order_id, outcome, memory_count)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (work_order_id) DO NOTHING
                """,
                tenant_id,
                agent_id,
                work_order_id,
                outcome,
                memory_count,
            )
    except Exception as exc:
        logger.warning({"event": "outcome_record_failed", "error": str(exc)})


async def _log_session_reads(
    results: list[RecallResult],
    tenant_id: UUID,
    agent_id: UUID,
    work_order_id: UUID,
    memory_clearance_order: int = 2,
    agent_division: str = "all",
) -> None:
    """Fire-and-forget: log recalled private memory IDs against the work order (ADR-125 Phase 1)."""
    private_ids = [r.id for r in results if r.scope == "private"]
    if not private_ids:
        return
    try:
        async with tenant_transaction(
            get_main_pool(),
            tenant_id,
            agent_id,
            memory_clearance_order=memory_clearance_order,
            agent_division=agent_division,
        ) as conn:
            await conn.execute(
                """
                INSERT INTO qortia_session_reads
                    (tenant_id, agent_id, work_order_id, memory_id)
                SELECT $1, $2, $3, unnest($4::uuid[])
                """,
                tenant_id,
                agent_id,
                work_order_id,
                private_ids,
            )
    except Exception as exc:
        logger.warning({"event": "session_reads_log_failed", "error": str(exc)})


# ── POST /v1/recall ──────────────────────────────────────────


@router.post("/v1/recall", response_model=RecallResponse)
async def _hybrid_recall_pipeline(
    body: RecallRequest,
    agent: AgentIdentity,
    clearance_order: int,
    agent_division: str,
) -> list[RecallResult]:
    """Full hybrid search pipeline: BM25+vector across scopes, entity graph
    boost, BFS traversal, MMR/RRF fusion, and cross-memory link expansion."""
    query_embedding = await _embed_query(
        body.query, agent.tenant_id, lang=body.lang or "en"
    )
    tasks = []

    if body.scope in ("private", "all", "archive"):
        tasks.append(_bm25_private(body, agent))
        if query_embedding:
            tasks.append(_vector_private(body, agent, query_embedding))

    if body.scope in ("org", "all"):
        tasks.append(_bm25_org(body, agent, clearance_order, agent_division))
        if query_embedding:
            tasks.append(
                _vector_org(
                    body, agent, query_embedding, clearance_order, agent_division
                )
            )

    if body.scope in ("knowledge", "all"):
        tasks.append(_bm25_knowledge(body, agent, clearance_order, agent_division))
        if query_embedding:
            tasks.append(
                _vector_knowledge(
                    body, agent, query_embedding, clearance_order, agent_division
                )
            )

    result_sets = await asyncio.gather(*tasks, return_exceptions=True)

    memory_results: list[RecallResult] = []
    knowledge_candidates: list[RecallResult] = []

    for rs in result_sets:
        if isinstance(rs, Exception):
            logger.warning({"event": "recall_search_error", "error": str(rs)})
            try:
                from app.telemetry.instruments import (
                    qortia_recall_degraded,
                )

                qortia_recall_degraded.add(
                    1,
                    {
                        "reason": "search_error",
                        "the platform.tenant_id": str(agent.tenant_id),
                    },
                )
            except Exception:
                pass
            continue
        for r in list(rs):  # type: ignore[arg-type]
            if r.scope == "knowledge":
                knowledge_candidates.append(r)
            else:
                memory_results.append(r)

    # ── Entity Graph Boost (The Obsidian Layer) ──
    from app.qortia.knowledge import extract_entities

    try:
        query_entities = extract_entities(body.query)
    except Exception:
        query_entities = []
    entity_links: set[str] = set()
    top_entity_summary: str | None = None
    if query_entities:
        async with tenant_transaction(
            get_main_pool(),
            agent.tenant_id,
            agent.agent_id,
            memory_clearance_order=clearance_order,
            agent_division=agent_division,
        ) as conn:
            linked_rows = await conn.fetch(
                """
                SELECT unnest(linked_memory_ids) as mem_id, summary
                FROM qortia_entities
                WHERE tenant_id = $1
                  AND (agent_id IS NULL OR agent_id = $2)
                  AND entity_text = ANY($3::text[])
                  AND max_clearance_order <= $4
            """,
                agent.tenant_id,
                agent.agent_id,
                query_entities,
                clearance_order,
            )
            entity_links = {str(r["mem_id"]) for r in linked_rows}
            top_entity_summary = next(
                (r["summary"] for r in linked_rows if r["summary"]), None
            )

    fused_memory = _rrf_fuse(memory_results, entity_links=entity_links)

    # 2-hop BFS traversal — surfaces memories reachable via entity co-occurrence
    if entity_links and query_embedding:
        async with tenant_transaction(
            get_main_pool(), agent.tenant_id, agent.agent_id
        ) as conn:
            seed_rows = await conn.fetch(
                """
                SELECT id FROM qortia_entities
                WHERE tenant_id = $1
                  AND (agent_id IS NULL OR agent_id = $2)
                  AND entity_text = ANY($3::text[])
                  AND max_clearance_order <= $4
                """,
                agent.tenant_id,
                agent.agent_id,
                query_entities,
                clearance_order,
            )
        seed_ids = [r["id"] for r in seed_rows]
        bfs_boosts = await _bfs_entity_traversal(
            query_embedding, seed_ids, agent.tenant_id, agent.agent_id
        )
        if bfs_boosts:
            # Merge BFS boosts into entity_links for a second RRF pass
            combined_links = entity_links | set(bfs_boosts.keys())
            fused_memory = _rrf_fuse(memory_results, entity_links=combined_links)

    if query_embedding and knowledge_candidates:
        for kc in knowledge_candidates:
            boost = _keyword_boost(body.query, kc.content)
            kc._score = kc._score * (1.0 + boost)
        knowledge_results = _mmr(
            query_embedding=query_embedding,
            candidates=knowledge_candidates,
            min_score=0.30,
        )
    else:
        knowledge_results = knowledge_candidates[:4]

    # Cross-memory link expansion (16i) — expand top-5 fused results with linked memories
    if fused_memory and query_embedding:
        from app.qortia.links import _expand_with_links

        fused_memory = await _expand_with_links(
            fused_memory, agent.tenant_id, agent.agent_id
        )

    results = fused_memory + knowledge_results

    # Attach entity summary to the top result if available
    if top_entity_summary and results:
        results[0].entity_summary = top_entity_summary

    return results


async def recall(
    body: RecallRequest,
    agent: AgentIdentity = Depends(require_agent),
    x_work_order_id: str | None = Header(default=None, alias="X-Work-Order-Id"),
) -> RecallResponse:
    from app.qortia.common import assert_agent_active

    clearance_order, agent_division = await _fetch_agent_clearance(
        agent.agent_id, agent.tenant_id
    )

    async with tenant_transaction(
        get_main_pool(),
        agent.tenant_id,
        agent.agent_id,
        memory_clearance_order=clearance_order,
        agent_division=agent_division,
    ) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)

    results: list[RecallResult] = []

    if body.type == "decision":
        results = await _recall_decisions(body, agent)
    elif body.type == "lesson":
        results = await _recall_lessons(body, agent)
    elif body.type == "episodic":
        results = await _recall_episodic(body, agent)
    elif body.type == "short_term":
        results = await _recall_short_term(body, agent)
    else:
        results = await _hybrid_recall_pipeline(
            body, agent, clearance_order, agent_division
        )

    if body.rerank and len(results) >= 2:
        results = await _llm_rerank(body.query, results, agent)

    async def _safe_record_recall_access() -> None:
        try:
            await _record_recall_access(
                results,
                agent.tenant_id,
                agent.agent_id,
                memory_clearance_order=clearance_order,
                agent_division=agent_division,
            )
        except Exception as exc:
            logger.warning({"event": "recall_access_record_failed", "error": str(exc)})

    asyncio.create_task(_safe_record_recall_access())

    if isinstance(x_work_order_id, str) and x_work_order_id:
        try:
            wo_uuid = UUID(x_work_order_id)
        except ValueError:
            wo_uuid = None
        if wo_uuid is not None:

            async def _safe_log_session_reads() -> None:
                await _log_session_reads(
                    results,
                    agent.tenant_id,
                    agent.agent_id,
                    wo_uuid,
                    memory_clearance_order=clearance_order,
                    agent_division=agent_division,
                )

            asyncio.create_task(_safe_log_session_reads())

    logger.info(
        {
            "event": "recall_executed",
            "agent_id": str(agent.agent_id),
            "the platform.tenant_id": str(agent.tenant_id),
            "scope": body.scope,
            "result_count": len(results),
            "work_order_id": x_work_order_id
            if isinstance(x_work_order_id, str)
            else None,
        }
    )

    return RecallResponse(results=results)
