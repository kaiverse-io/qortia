from __future__ import annotations

import datetime
import json
import logging
from uuid import UUID

from typing import Literal, Any
from fastapi import APIRouter, HTTPException

from app.auth.models import AgentIdentity
from app.qortia.knowledge import extract_entities_with_types
from app.db import get_main_pool
from app.vault import provision_eval_litellm_key
from pydantic import BaseModel
from app.config import settings

logger = logging.getLogger(__name__)
router: APIRouter = APIRouter(prefix="/v1/internal/eval")


# lint:allow-cross-tenant — ADR-073: eval endpoints seed test data across tenants by design;
# only reachable when settings.eval_mode is True (never enabled in production).
@router.post("/seed-agent")
async def seed_eval_agent(
    agent_id: UUID, tenant_id: UUID, name: str = "eval_agent", role: str = "custom"
) -> dict[str, Any]:
    """Seeds an agent for evaluation purposes. Only works in EVAL_MODE."""
    if not settings.eval_mode:
        raise HTTPException(404, "Not found")

    async with get_main_pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO auth.tenants (id, name) VALUES ($1, $2) ON CONFLICT (id) DO NOTHING",
            tenant_id,
            f"eval-tenant-{str(tenant_id)[:8]}",
        )
        await conn.execute(
            """
            INSERT INTO auth.agents (id, tenant_id, name, role, status, soul_md, domain_md)
            VALUES ($1, $2, $3, $4, 'active', 'eval soul', 'eval domain')
            ON CONFLICT (id) DO UPDATE SET status = 'active'
        """,
            agent_id,
            tenant_id,
            name,
            role,
        )

    await provision_eval_litellm_key(str(tenant_id))
    return {"status": "seeded", "agent_id": str(agent_id)}


class SeedMemoryRequest(BaseModel):
    agent_id: UUID
    tenant_id: UUID
    content: str
    mem_type: str = "episodic"
    scope: str = "private"
    lang: str = "en"
    importance: float = 0.5
    valid_from: str | None = None   # ISO-8601; None = always valid from creation
    valid_until: str | None = None  # ISO-8601; None = currently valid (no expiry)
    is_consolidated: bool = False   # For mental_model/lesson seeded directly


@router.post("/seed-memory")
async def seed_eval_memory(req: SeedMemoryRequest) -> dict[str, Any]:
    if not settings.eval_mode:
        raise HTTPException(404, "Not found")

    try:
        entities = extract_entities_with_types(req.content, lang=req.lang)
    except Exception:
        entities = []

    def _parse_dt(s: str | None) -> datetime.datetime | None:
        if s is None:
            return None
        try:
            dt = datetime.datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt
        except ValueError:
            return None

    valid_from_dt = _parse_dt(req.valid_from)
    valid_until_dt = _parse_dt(req.valid_until)

    # Build INSERT dynamically — only include temporal columns when explicitly set
    # to avoid overriding NOT NULL DEFAULT values with NULL.
    extra_cols = "is_consolidated"
    extra_vals = "$8"
    params: list[object] = [
        req.tenant_id, req.agent_id, req.mem_type, req.content,
        req.importance, json.dumps(entities), req.lang, req.is_consolidated,
    ]
    if valid_from_dt is not None:
        params.append(valid_from_dt)
        extra_cols += f", valid_from"
        extra_vals += f", ${len(params)}"
    if valid_until_dt is not None:
        params.append(valid_until_dt)
        extra_cols += f", valid_until"
        extra_vals += f", ${len(params)}"

    async with get_main_pool().acquire() as conn:
        row_id = await conn.fetchval(
            f"""
            INSERT INTO hindsight_memories
                (tenant_id, agent_id, type, content, importance, entities, lang,
                 {extra_cols})
            VALUES ($1, $2, $3, $4, $5, $6, $7, {extra_vals})
            RETURNING id
        """,
            *params,
        )

    return {"status": "seeded", "memory_id": str(row_id)}


@router.post("/recall")
async def eval_recall(
    query: str,
    tenant_id: UUID,
    agent_id: UUID,
    scope: Literal["private", "org", "knowledge", "all"] = "private",
    limit: int = 10,
) -> dict[str, Any]:
    if not settings.eval_mode:
        raise HTTPException(404, "Not found")

    from app.qortia.recall import recall
    from app.qortia.models import RecallRequest

    agent = AgentIdentity(agent_id=agent_id, tenant_id=tenant_id)
    body = RecallRequest(query=query, scope=scope)
    resp = await recall(body, agent, x_work_order_id=None)
    return resp.model_dump()


@router.post("/recall-full")
async def eval_recall_full(
    body: "RecallRequestFull",
    tenant_id: UUID,
    agent_id: UUID,
) -> dict[str, Any]:
    """Full RecallRequest eval endpoint — accepts complete recall body without JWT.

    Used by REH and ALB harnesses to call the production recall pipeline without
    requiring a signed JWT. Only reachable when EVAL_MODE=true.
    """
    if not settings.eval_mode:
        raise HTTPException(404, "Not found")

    from app.qortia.recall import recall
    from app.qortia.models import RecallRequest

    agent = AgentIdentity(agent_id=agent_id, tenant_id=tenant_id)
    req = RecallRequest(
        query=body.query,
        scope=body.scope,
        type=body.type,
        entities=body.entities,
        rerank=body.rerank,
        lang=body.lang,
        as_of=body.as_of,
    )
    resp = await recall(req, agent, x_work_order_id=None)
    return resp.model_dump()


@router.post("/reflect")
async def eval_reflect(
    tenant_id: UUID,
    agent_id: UUID,
) -> dict[str, Any]:
    """Trigger reflection for an eval agent without JWT.

    Used by ALB Task B to verify reflection consolidation behaviour.
    Only reachable when EVAL_MODE=true.
    """
    if not settings.eval_mode:
        raise HTTPException(404, "Not found")

    from app.qortia.reflect import reflect

    agent = AgentIdentity(agent_id=agent_id, tenant_id=tenant_id)
    resp = await reflect(agent)
    return resp.model_dump()


@router.post("/remember-org")
async def eval_remember_org(
    body: "RememberOrgRequestBody",
    tenant_id: UUID,
    agent_id: UUID,
) -> dict[str, Any]:
    """Seed org memory for eval without JWT. Only reachable when EVAL_MODE=true."""
    if not settings.eval_mode:
        raise HTTPException(404, "Not found")

    from app.qortia.remember import remember_org
    from app.qortia.models import RememberOrgRequest

    agent = AgentIdentity(agent_id=agent_id, tenant_id=tenant_id)
    req = RememberOrgRequest(
        type=body.type,
        title=body.title,
        content=body.content,
        lang=body.lang,
    )
    resp = await remember_org(req, agent)
    org_id = resp.id

    # Apply valid_until/valid_from if provided (eval-only — supports TEH temporal cases)
    if body.valid_until or body.valid_from:
        valid_until_dt = _parse_dt(body.valid_until)
        valid_from_dt = _parse_dt(body.valid_from)
        async with get_main_pool().acquire() as conn:
            if valid_until_dt:
                await conn.execute(
                    "UPDATE org_memory SET valid_until = $1 WHERE id = $2 AND tenant_id = $3",
                    valid_until_dt, UUID(org_id), tenant_id,
                )
            if valid_from_dt:
                await conn.execute(
                    "UPDATE org_memory SET valid_from = $1 WHERE id = $2 AND tenant_id = $3",
                    valid_from_dt, UUID(org_id), tenant_id,
                )

    return {"id": org_id}


# ── Request models for eval endpoints ─────────────────────────────────────


from pydantic import BaseModel as _BaseModel  # noqa: E402


class RecallRequestFull(_BaseModel):
    query: str
    scope: Literal["private", "org", "knowledge", "all"] = "private"
    type: str | None = None
    entities: list[str] | None = None
    rerank: bool = False
    lang: str | None = None
    as_of: str | None = None


class RememberOrgRequestBody(_BaseModel):
    type: str
    title: str
    content: str
    lang: str = "en"
    valid_from: str | None = None
    valid_until: str | None = None


@router.post("/knowledge")
async def eval_ingest_knowledge(
    body: "KnowledgeIngestBody",
    tenant_id: UUID,
    agent_id: UUID,
) -> dict[str, Any]:
    """Ingest knowledge for eval without JWT. Only reachable when EVAL_MODE=true."""
    if not settings.eval_mode:
        raise HTTPException(404, "Not found")

    from app.qortia.knowledge import ingest_knowledge
    from app.qortia.models import KnowledgeIngestRequest

    agent = AgentIdentity(agent_id=agent_id, tenant_id=tenant_id)
    req = KnowledgeIngestRequest(
        source_type=body.source_type,
        source_path=body.source_path,
        content=body.content,
        lang=body.lang,
    )
    return await ingest_knowledge(req, agent)


class KnowledgeIngestBody(_BaseModel):
    source_type: str
    source_path: str
    content: str
    lang: str = "en"
