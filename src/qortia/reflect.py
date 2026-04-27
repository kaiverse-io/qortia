from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from uuid import UUID

import httpx
import yaml
from fastapi import APIRouter, Depends, HTTPException

from app.auth.middleware import require_agent
from app.auth.models import AgentIdentity
from app.config import settings
from app.qortia.models import ReflectResponse
from app.db import get_main_pool, tenant_transaction
from app.vault import get_litellm_key

logger = logging.getLogger(__name__)
router = APIRouter()

REFLECTION_THRESHOLD = 10
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 50
STABILITY_THRESHOLD = 0.95

_litellm_client: httpx.AsyncClient | None = None


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
    from app.qortia.recall import _cosine

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


def init_litellm_client() -> None:
    global _litellm_client
    _litellm_client = httpx.AsyncClient(base_url=settings.litellm_url, timeout=None)
    logger.info({"event": "litellm_client_initialized"})


async def close_litellm_client() -> None:
    if _litellm_client is not None:
        await _litellm_client.aclose()


def get_litellm_client() -> httpx.AsyncClient:
    assert _litellm_client is not None, "LiteLLM client not initialised"
    return _litellm_client


# ── Cost ledger ─────────────────────────────────────────────


async def _record_llm_cost(
    agent_id: UUID,
    tenant_id: UUID,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> None:
    """Fire-and-forget — failure is non-fatal."""
    # Approximate cost using OpenAI pricing tiers as a proxy.
    # LiteLLM returns cost in usage.cost when available; fall back to estimate.
    cost_usd = (tokens_in * 3.0 + tokens_out * 15.0) / 1_000_000
    try:
        async with get_main_pool().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_cost_ledger
                    (tenant_id, agent_id, model, tokens_in, tokens_out, cost_usd)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                tenant_id,
                agent_id,
                model,
                tokens_in,
                tokens_out,
                cost_usd,
            )
    except Exception as exc:
        logger.warning({"event": "cost_ledger_write_failed", "error": str(exc)})


# ── POST /v1/reflect ─────────────────────────────────────────


@router.post("/v1/reflect", response_model=ReflectResponse)
async def reflect(agent: AgentIdentity = Depends(require_agent)) -> ReflectResponse:
    from app.qortia.common import assert_agent_active

    # Fetch data outside the write transaction — no DB lock held during LLM call
    async with tenant_transaction(
        get_main_pool(), agent.tenant_id, agent.agent_id
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

        domain_md_raw = await conn.fetchval(
            "SELECT domain_md FROM auth.agents WHERE id = $1 AND tenant_id = $2",
            agent.agent_id,
            agent.tenant_id,
        )

    domain = yaml.safe_load(domain_md_raw)
    model = domain.get("model", "anthropic/claude-3-haiku-20240307")

    litellm_key = await get_litellm_key(str(agent.tenant_id))
    reflections = await _call_litellm_reflect(
        model=model,
        recent=[r["content"] for r in recent],
        existing=[
            {"id": str(r["id"]), "type": r["type"], "content": r["content"]}
            for r in existing
        ],
        litellm_key=litellm_key,
        agent_id=agent.agent_id,
        tenant_id=agent.tenant_id,
    )

    # Build existing embedding index keyed by id for stability computation
    existing_embeddings: dict[str, list[float] | None] = {
        str(r["id"]): list(r["embedding"]) if r.get("embedding") else None
        for r in existing
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
    async with tenant_transaction(
        get_main_pool(), agent.tenant_id, agent.agent_id
    ) as conn:
        active_ids = []
        for r in reflections:
            if r["action"] in ("UPDATE", "RETAIN"):
                active_ids.append(UUID(r["id"]))

        # 6a. PRUNE — anything not returned by LLM is marked non-consolidated
        if existing:
            await conn.execute(
                """
                UPDATE hindsight_memories
                SET is_consolidated = false
                WHERE agent_id = $1
                  AND type IN ('mental_model', 'lesson')
                  AND is_consolidated = true
                  AND id != ALL($2::uuid[])
            """,
                agent.agent_id,
                active_ids,
            )

        new_ids = []
        seen_hashes: set[str] = set()
        stable_count = 0
        unstable_count = 0
        for i, r in enumerate(reflections):
            if r["action"] == "RETAIN":
                continue

            # MD5 pre-filter: skip exact/near-exact duplicates within this batch
            content_hash = hashlib.md5(
                r["content"].lower().strip().encode()
            ).hexdigest()
            if content_hash in seen_hashes:
                logger.info({"event": "reflect_dedup_skipped", "hash": content_hash})
                continue
            seen_hashes.add(content_hash)

            # Compute stability score
            new_emb = new_embeddings.get(i)
            stability: float | None = None
            if r["action"] == "UPDATE" and r.get("id"):
                old_emb = existing_embeddings.get(r["id"])
                if new_emb and old_emb:
                    from app.qortia.recall import _cosine

                    stability = _cosine(new_emb, old_emb)
                    if stability >= STABILITY_THRESHOLD:
                        stable_count += 1
                    else:
                        unstable_count += 1
            elif r["action"] == "CREATE":
                unstable_count += 1

            # For UPDATE, deactivate the old row first
            if r["action"] == "UPDATE":
                await conn.execute(
                    """
                    UPDATE hindsight_memories
                    SET is_consolidated = false
                    WHERE id = $1 AND agent_id = $2
                """,
                    UUID(r["id"]),
                    agent.agent_id,
                )

            # Insert new version (for CREATE or UPDATE)
            try:
                from app.qortia.knowledge import extract_entities_with_types

                entities = extract_entities_with_types(r["content"])
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
                agent.tenant_id,
                agent.agent_id,
                r["type"],
                r["content"],
                float(r["importance"]),
                json.dumps(entities),
                str(new_emb) if new_emb else None,
                stability,
            )
            new_ids.append((row_id, r.get("id")))

        # 6c. Decrement reflection_counter atomically
        new_counter = await conn.fetchval(
            """
            UPDATE auth.agents
            SET reflection_counter = GREATEST(reflection_counter - $1, 0),
                updated_at = now()
            WHERE id = $2
            RETURNING reflection_counter
        """,
            REFLECTION_THRESHOLD,
            agent.agent_id,
        )

        # 6d. Audit trail (with lineage for updates)
        for row_id, parent_id in new_ids:
            metadata = {}
            if parent_id:
                metadata["parent_id"] = parent_id
            await conn.execute(
                """
                INSERT INTO memory_history
                    (tenant_id, agent_id, operation, target_table, target_id, metadata)
                VALUES ($1, $2, 'reflect', 'hindsight_memories', $3, $4)
            """,
                agent.tenant_id,
                agent.agent_id,
                row_id,
                json.dumps(metadata),
            )

    logger.info(
        {
            "event": "reflect_incremental",
            "agent_id": str(agent.agent_id),
            "memories_written": len(new_ids),
            "stable_updates": stable_count,
            "unstable_updates": unstable_count,
        }
    )

    return ReflectResponse(
        memories_written=len(new_ids), reflection_counter=new_counter
    )


async def _call_litellm_reflect(
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
        raw = resp.json()["choices"][0]["message"]["content"]
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
                if not isinstance(r.get("importance"), (int, float)):
                    raise ValueError("importance must be numeric")
                if not r.get("content"):
                    raise ValueError("content must not be empty")
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error({"event": "reflection_llm_malformed", "error": str(exc)})
        raise HTTPException(500, "Reflection failed: malformed LLM response")

    # Record cost — fire-and-forget, non-fatal
    usage = resp.json().get("usage", {})
    if usage:
        asyncio.create_task(
            _record_llm_cost(
                agent_id=agent_id,
                tenant_id=tenant_id,
                model=model,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
            )
        )

    return reflections  # type: ignore[no-any-return]


def _build_reflect_prompt(recent: list[str], existing: list[dict]) -> str:  # type: ignore[type-arg]
    recent_block = "\n".join(f"- {m}" for m in recent) or "(none)"
    existing_block = (
        "\n".join(f"- [{m['id']}] [{m['type']}] {m['content']}" for m in existing)
        or "(none)"
    )
    return f"""You are synthesising an agent's recent experiences into durable mental models and lessons.

Recent episodic and experiential memories (last 7 days):
{recent_block}

Existing consolidated knowledge (will be superseded by your output):
{existing_block}

Your task is to refine the agent's knowledge. You can CREATE new models, UPDATE existing ones with more detail, or RETAIN existing ones that are still accurate.

Return a JSON object with this exact schema:
{{
  "reflections": [
    {{"action": "CREATE", "type": "mental_model", "content": "...", "importance": 0.85}},
    {{"action": "UPDATE", "id": "uuid-here", "type": "lesson", "content": "updated content", "importance": 0.90}},
    {{"action": "RETAIN", "id": "uuid-here"}}
  ]
}}

Rules:
- action must be "CREATE", "UPDATE", or "RETAIN"
- type must be "mental_model" or "lesson" only
- importance is a float between 0.0 and 1.0
- content must be a non-empty string
- Return all knowledge items that should remain consolidated. Any existing ID NOT returned will be pruned (deleted).
- Synthesise patterns, not events
- Preserve named entities (people, systems, organisations) — they are recall anchors
- When a memory contains a temporal marker ("last quarter", "in March"), preserve it — do not strip temporal context
- Attribute observations to their source when relevant: "user reported X", "agent observed Y", "team decided Z"

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
    async with get_main_pool().acquire() as conn:
        async with conn.transaction():
            hindsight = await conn.fetch(
                """
                SELECT id, tenant_id, content AS text_to_embed, 'hindsight_memories' AS tbl
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
                SELECT id, tenant_id, content AS text_to_embed, 'org_memory' AS tbl
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
                SELECT id, tenant_id, index_summary AS text_to_embed, 'org_knowledge' AS tbl
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
                dict(r)
                for r in list(hindsight)
                + list(org_mem)
                + list(org_know)
                + list(entities)
            ]

    # Group by tenant_id to minimise Vault round-trips (one key fetch per tenant per batch)
    from collections import defaultdict

    rows_by_tenant = defaultdict(list)
    for row in rows:
        rows_by_tenant[str(row["tenant_id"])].append(row)

    for tenant_id_str, tenant_rows in rows_by_tenant.items():
        try:
            litellm_key = await get_litellm_key(tenant_id_str)
        except Exception as exc:
            logger.warning(
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
    try:
        embedding = await _get_embedding(row["text_to_embed"], litellm_key)
        async with get_main_pool().acquire() as conn:
            await conn.execute(
                f"UPDATE {row['tbl']} SET embedding = $1::vector WHERE id = $2",
                str(embedding),
                row["id"],
            )
    except Exception as exc:
        async with get_main_pool().acquire() as conn:
            await conn.execute(
                f"UPDATE {row['tbl']} SET embedding_attempts = embedding_attempts + 1 WHERE id = $1",
                row["id"],
            )
            attempts = await conn.fetchval(
                f"SELECT embedding_attempts FROM {row['tbl']} WHERE id = $1", row["id"]
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


async def _get_embedding(text: str, litellm_key: str) -> list[float]:
    resp = await get_litellm_client().post(
        "/embeddings",
        headers={"Authorization": f"Bearer {litellm_key}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]  # type: ignore[no-any-return]


async def validate_embedding_dimensions() -> None:
    resp = await get_litellm_client().post(
        "/embeddings",
        headers={"Authorization": f"Bearer {settings.litellm_master_key}"},
        json={"model": EMBEDDING_MODEL, "input": "dimension check"},
        timeout=10.0,
    )
    resp.raise_for_status()
    actual = len(resp.json()["data"][0]["embedding"])
    if actual != 768:
        raise RuntimeError(
            f"Embedding dimension mismatch: schema expects 768, got {actual}."
        )


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
                SELECT id, tenant_id, agent_id, entities
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
                            INSERT INTO qortia_entities (tenant_id, agent_id, entity_text, entity_type, linked_memory_ids)
                            VALUES ($1, $2, $3, $4, ARRAY[$5::uuid])
                            ON CONFLICT (tenant_id, agent_id, entity_text) WHERE agent_id IS NOT NULL
                            DO UPDATE SET
                                entity_type = EXCLUDED.entity_type,
                                linked_memory_ids = array_append(qortia_entities.linked_memory_ids, $5),
                                updated_at = now()
                            WHERE NOT ($5 = ANY(qortia_entities.linked_memory_ids))
                        """,
                            row["tenant_id"],
                            row["agent_id"],
                            ent_text,
                            ent_type,
                            row["id"],
                        )
                    await conn.execute(
                        "UPDATE hindsight_memories SET is_graphed = true WHERE id = $1",
                        row["id"],
                    )

            # 2. Org Memories (Shared)
            rows = await conn.fetch(
                """
                SELECT id, tenant_id, entities
                FROM org_memory
                WHERE is_graphed = false AND entities != '[]'
                LIMIT 50
                FOR UPDATE SKIP LOCKED
            """
            )
            for row in rows:
                async with conn.transaction():
                    try:
                        entity_pairs = [
                            (e[0], e[1]) for e in json.loads(row["entities"])
                        ]
                    except Exception:
                        entity_pairs = []
                    for ent_text, ent_type in entity_pairs:
                        await conn.execute(
                            """
                            INSERT INTO qortia_entities (tenant_id, agent_id, entity_text, entity_type, linked_memory_ids)
                            VALUES ($1, NULL, $2, $3, ARRAY[$4::uuid])
                            ON CONFLICT (tenant_id, entity_text) WHERE agent_id IS NULL
                            DO UPDATE SET
                                entity_type = EXCLUDED.entity_type,
                                linked_memory_ids = array_append(qortia_entities.linked_memory_ids, $4),
                                updated_at = now()
                            WHERE NOT ($4 = ANY(qortia_entities.linked_memory_ids))
                        """,
                            row["tenant_id"],
                            ent_text,
                            ent_type,
                            row["id"],
                        )
                    await conn.execute(
                        "UPDATE org_memory SET is_graphed = true WHERE id = $1",
                        row["id"],
                    )
    except Exception as exc:
        logger.warning({"event": "graph_population_failed", "error": str(exc)})
