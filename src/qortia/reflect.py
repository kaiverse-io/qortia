from __future__ import annotations

import asyncio
import json
import logging
from uuid import UUID
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

_litellm_client: httpx.AsyncClient | None = None


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
                tenant_id, agent_id, model, tokens_in, tokens_out, cost_usd,
            )
    except Exception as exc:
        logger.warning({"event": "cost_ledger_write_failed", "error": str(exc)})


# ── POST /v1/reflect ─────────────────────────────────────────

@router.post("/v1/reflect", response_model=ReflectResponse)
async def reflect(agent: AgentIdentity = Depends(require_agent)) -> ReflectResponse:
    from app.qortia.remember import assert_agent_active

    # Fetch data outside the write transaction — no DB lock held during LLM call
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)

        recent = await conn.fetch("""
            SELECT content FROM hindsight_memories
            WHERE agent_id = $1
              AND type IN ('episodic', 'experiential')
              AND created_at > now() - interval '7 days'
            ORDER BY created_at DESC LIMIT 30
        """, agent.agent_id)

        existing = await conn.fetch("""
            SELECT id, type, content FROM hindsight_memories
            WHERE agent_id = $1
              AND type IN ('mental_model', 'lesson')
              AND is_consolidated = true
            ORDER BY importance DESC
        """, agent.agent_id)

        domain_md_raw = await conn.fetchval(
            "SELECT domain_md FROM auth.agents WHERE id = $1 AND tenant_id = $2",
            agent.agent_id, agent.tenant_id,
        )

    domain = yaml.safe_load(domain_md_raw)
    model = domain.get("model", "anthropic/claude-3-haiku-20240307")

    litellm_key = await get_litellm_key(str(agent.tenant_id))
    new_memories = await _call_litellm_reflect(
        model=model,
        recent=[r["content"] for r in recent],
        existing=[{"type": r["type"], "content": r["content"]} for r in existing],
        litellm_key=litellm_key,
        agent_id=agent.agent_id,
        tenant_id=agent.tenant_id,
    )

    # Write transaction — supersede-first order (Q28, safety-critical)
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        # 6a. SUPERSEDE FIRST — crash here leaves old set intact
        if existing:
            await conn.execute("""
                UPDATE hindsight_memories
                SET is_consolidated = false
                WHERE agent_id = $1
                  AND type IN ('mental_model', 'lesson')
                  AND is_consolidated = true
            """, agent.agent_id)

        # 6b. Write new consolidated memories
        new_ids = []
        for mem in new_memories:
            try:
                from app.qortia.knowledge import extract_entities
                entities = extract_entities(mem["content"])
            except Exception:
                entities = []

            row_id = await conn.fetchval("""
                INSERT INTO hindsight_memories
                    (tenant_id, agent_id, type, content, importance, is_consolidated, entities)
                VALUES ($1, $2, $3, $4, $5, true, $6)
                RETURNING id
            """,
                agent.tenant_id, agent.agent_id,
                mem["type"], mem["content"], float(mem["importance"]),
                json.dumps(entities),
            )
            new_ids.append(row_id)

        # 6c. Decrement reflection_counter atomically
        new_counter = await conn.fetchval("""
            UPDATE auth.agents
            SET reflection_counter = GREATEST(reflection_counter - $1, 0),
                updated_at = now()
            WHERE id = $2
            RETURNING reflection_counter
        """, REFLECTION_THRESHOLD, agent.agent_id)

        # 6d. Audit trail
        for row_id in new_ids:
            await conn.execute("""
                INSERT INTO memory_history
                    (tenant_id, agent_id, operation, target_table, target_id, metadata)
                VALUES ($1, $2, 'reflect', 'hindsight_memories', $3, '{}')
            """, agent.tenant_id, agent.agent_id, row_id)

    return ReflectResponse(memories_written=len(new_ids), reflection_counter=new_counter)


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
        memories = parsed["memories"]
        if not memories:
            raise ValueError("empty memories array")
        for m in memories:
            if m["type"] not in ("mental_model", "lesson"):
                raise ValueError(f"invalid type: {m['type']}")
            if not isinstance(m.get("importance"), (int, float)):
                raise ValueError("importance must be numeric")
            if not m.get("content"):
                raise ValueError("content must not be empty")
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error({"event": "reflection_llm_malformed", "error": str(exc)})
        raise HTTPException(500, "Reflection failed: malformed LLM response")

    # Record cost — fire-and-forget, non-fatal
    usage = resp.json().get("usage", {})
    if usage:
        asyncio.create_task(_record_llm_cost(
            agent_id=agent_id,
            tenant_id=tenant_id,
            model=model,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        ))

    return memories


def _build_reflect_prompt(recent: list[str], existing: list[dict]) -> str:  # type: ignore[type-arg]
    recent_block = "\n".join(f"- {m}" for m in recent) or "(none)"
    existing_block = "\n".join(
        f"- [{m['type']}] {m['content']}" for m in existing
    ) or "(none)"
    return f"""You are synthesising an agent's recent experiences into durable mental models and lessons.

Recent episodic and experiential memories (last 7 days):
{recent_block}

Existing consolidated knowledge (will be superseded by your output):
{existing_block}

Return a JSON object with this exact schema:
{{
  "memories": [
    {{"type": "mental_model", "content": "...", "importance": 0.85}},
    {{"type": "lesson", "content": "...", "importance": 0.90}}
  ]
}}

Rules:
- type must be "mental_model" or "lesson" only
- importance is a float between 0.0 and 1.0
- content must be a non-empty string
- Return at least one memory
- Do not reference specific dates or ephemeral details
- Synthesise patterns, not events"""


# ── Embedding worker ─────────────────────────────────────────

async def run_embedding_worker() -> None:
    while True:
        await asyncio.sleep(10)
        await _process_embedding_batch()


async def _process_embedding_batch() -> None:
    # Step 1: claim rows outside a long-held transaction
    async with get_main_pool().acquire() as conn:
        async with conn.transaction():
            hindsight = await conn.fetch("""
                SELECT id, content AS text_to_embed, 'hindsight_memories' AS tbl
                FROM hindsight_memories
                WHERE embedding IS NULL AND embedding_attempts < 3
                ORDER BY tenant_id, created_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            """, EMBEDDING_BATCH_SIZE)

            org_mem = await conn.fetch("""
                SELECT id, content AS text_to_embed, 'org_memory' AS tbl
                FROM org_memory
                WHERE embedding IS NULL AND embedding_attempts < 3
                ORDER BY tenant_id, created_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            """, EMBEDDING_BATCH_SIZE)

            org_know = await conn.fetch("""
                SELECT id, index_summary AS text_to_embed, 'org_knowledge' AS tbl
                FROM org_knowledge
                WHERE embedding IS NULL AND embedding_attempts < 3
                ORDER BY tenant_id, created_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            """, EMBEDDING_BATCH_SIZE)

            rows = [dict(r) for r in list(hindsight) + list(org_mem) + list(org_know)]

    # Step 2: embed + write each row — no transaction held during HTTP calls
    for row in rows:
        await _embed_single_row(row)


async def _embed_single_row(row: dict) -> None:  # type: ignore[type-arg]
    if not row.get("text_to_embed"):
        return
    try:
        embedding = await _get_embedding(row["text_to_embed"])
        async with get_main_pool().acquire() as conn:
            await conn.execute(
                f"UPDATE {row['tbl']} SET embedding = $1 WHERE id = $2",
                embedding, row["id"],
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
            logger.warning({
                "event": "embedding_failed",
                "table": row["tbl"],
                "row_id": str(row["id"]),
                "attempts": attempts,
                "error": str(exc),
            })


async def _get_embedding(text: str) -> list[float]:
    resp = await get_litellm_client().post(
        "/embeddings",
        headers={"Authorization": f"Bearer {settings.litellm_master_key}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]
