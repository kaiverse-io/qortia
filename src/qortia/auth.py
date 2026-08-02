"""Standalone request authentication — API keys, no JWT/Vault/external IDP.

Replaces the old host-platform `app.auth.middleware` + `app.auth.models`.

An API key authenticates a *tenant*, not an individual agent — one key can
address any agent belonging to that tenant. The caller also sends an
`X-Agent-Id` header naming which agent it's acting as; `require_agent`
validates that agent actually belongs to the key's tenant (a cheap existence
check) before trusting it. This is what stops tenant A's key from being used
to address tenant B's agent.

Keys are stored as SHA-256 hashes only — the plaintext is returned once at
issuance (see qortia.provisioning) and never persisted. SHA-256 rather than a
slow password hash (bcrypt/argon2) is deliberate: these are high-entropy
random tokens, not user-chosen passwords, and this lookup runs on every
request, so a slow hash would add latency for no security benefit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from uuid import UUID

from fastapi import Header, HTTPException

from qortia import config
from qortia.db import get_main_pool


@dataclass(frozen=True)
class AgentIdentity:
    agent_id: UUID
    tenant_id: UUID
    clearance_order: int = 2
    division: str = "all"


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def require_agent(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None, alias="X-Agent-Id"),
) -> AgentIdentity:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header")
    api_key = authorization.removeprefix("Bearer ").strip()
    if not api_key:
        raise HTTPException(401, "Missing API key")

    if x_agent_id is None:
        raise HTTPException(401, "Missing X-Agent-Id header")
    try:
        agent_id = UUID(x_agent_id)
    except ValueError as exc:
        raise HTTPException(401, "X-Agent-Id is not a valid UUID") from exc

    key_hash = hash_api_key(api_key)
    async with get_main_pool().acquire() as conn:
        row = await conn.fetchrow(
            "SELECT tenant_id FROM qortia_api_keys WHERE key_hash = $1 AND revoked_at IS NULL",
            key_hash,
        )
        if row is None:
            raise HTTPException(401, "Invalid or revoked API key")
        tenant_id: UUID = row["tenant_id"]

        agent_row = await conn.fetchrow(
            """
            SELECT cl.level_order, a.division
            FROM qortia_agents a
            LEFT JOIN qortia_clearance_levels cl ON cl.level_name = a.clearance_level
            WHERE a.id = $1 AND a.tenant_id = $2
            """,
            agent_id,
            tenant_id,
        )
        if agent_row is None:
            raise HTTPException(403, "Agent does not belong to the authenticated tenant")

    clearance_order = agent_row["level_order"] if agent_row["level_order"] is not None else 2
    division = agent_row["division"] or "all"
    return AgentIdentity(
        agent_id=agent_id, tenant_id=tenant_id, clearance_order=clearance_order, division=division
    )


async def get_litellm_key(tenant_id: str) -> str:
    """Resolve the LiteLLM virtual key for a tenant (ADR-003).

    Looks up `QORTIA_LITELLM_TENANT_KEYS[tenant_id]` first, then falls back to
    the shared `QORTIA_LITELLM_API_KEY`. No Vault — operators mint virtual keys
    in the LiteLLM Admin UI/API and map them via env.
    """
    mapped = config.settings.litellm_tenant_keys.get(tenant_id)
    if mapped:
        return mapped
    return config.settings.litellm_api_key


def get_platform_embed_key() -> str:
    """Shared/master key for startup probes and non-tenant platform work."""
    return config.settings.litellm_api_key


async def provision_eval_litellm_key(tenant_id: str) -> None:
    """No-op standalone: map eval tenants in QORTIA_LITELLM_TENANT_KEYS if needed."""
    return None
