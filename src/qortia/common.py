from __future__ import annotations

import asyncpg
from uuid import UUID
from fastapi import HTTPException


async def assert_agent_active(
    agent_id: UUID, tenant_id: UUID, conn: asyncpg.Connection
) -> None:
    row = await conn.fetchrow(
        "SELECT status FROM auth.agents WHERE id = $1 AND tenant_id = $2",
        agent_id,
        tenant_id,
    )
    if row is None or row["status"] != "active":
        raise HTTPException(403, "Agent is not active")
