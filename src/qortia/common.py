from __future__ import annotations

from uuid import UUID

import asyncpg
import httpx
from fastapi import HTTPException

from qortia import config

# Back-compat alias — prefer qortia.embeddings.embedding_model().
EMBEDDING_MODEL = config.settings.embedding_model

_litellm_client: httpx.AsyncClient | None = None


def init_litellm_client() -> None:
    global _litellm_client
    _litellm_client = httpx.AsyncClient(base_url=config.settings.litellm_url, timeout=None)  # noqa: S113


async def close_litellm_client() -> None:
    if _litellm_client is not None:
        await _litellm_client.aclose()


def get_litellm_client() -> httpx.AsyncClient:
    assert _litellm_client is not None, "LiteLLM client not initialised"  # noqa: S101
    return _litellm_client


async def assert_agent_active(agent_id: UUID, tenant_id: UUID, conn: asyncpg.Connection) -> None:
    row = await conn.fetchrow(
        "SELECT status FROM qortia_agents WHERE id = $1 AND tenant_id = $2",
        agent_id,
        tenant_id,
    )
    if row is None or row["status"] != "active":
        raise HTTPException(403, "Agent is not active")
