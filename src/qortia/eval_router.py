from __future__ import annotations

import logging
from uuid import UUID

from typing import Literal, Any
from fastapi import APIRouter, HTTPException

from app.auth.models import AgentIdentity
from app.db import get_main_pool
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

    return {"status": "seeded", "agent_id": str(agent_id)}


class SeedMemoryRequest(BaseModel):
    agent_id: UUID
    tenant_id: UUID
    content: str
    mem_type: str = "episodic"
    scope: str = "private"


@router.post("/seed-memory")
async def seed_eval_memory(req: SeedMemoryRequest) -> dict[str, Any]:
    if not settings.eval_mode:
        raise HTTPException(404, "Not found")

    async with get_main_pool().acquire() as conn:
        row_id = await conn.fetchval(
            """
            INSERT INTO hindsight_memories (tenant_id, agent_id, type, content, importance)
            VALUES ($1, $2, $3, $4, 0.5)
            RETURNING id
        """,
            req.tenant_id,
            req.agent_id,
            req.mem_type,
            req.content,
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

    # We call the internal recall logic directly
    from app.qortia.recall import recall
    from app.qortia.models import RecallRequest

    # Mock agent identity
    agent = AgentIdentity(agent_id=agent_id, tenant_id=tenant_id)

    body = RecallRequest(query=query, scope=scope)
    resp = await recall(body, agent)
    return resp.model_dump()
