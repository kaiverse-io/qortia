"""Tenant/agent/API-key provisioning — CLI and direct function calls only.

There's deliberately no HTTP endpoint for any of this in v1: the very first
API key for a fresh tenant can't be created through the API (nothing to
authenticate that request with). The standard answer is an out-of-band
CLI/admin path with DB access, independent of the HTTP API entirely — that's
what `qortia-admin` is. A self-service HTTP endpoint for issuing *additional*
keys (once a tenant already holds one valid key) is a reasonable future
addition, not built here.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
from uuid import UUID, uuid4

import asyncpg

from qortia import config
from qortia.auth import hash_api_key


async def create_tenant(pool: asyncpg.Pool, name: str | None = None) -> UUID:
    tenant_id = uuid4()
    await pool.execute(
        "INSERT INTO qortia_tenants (id, name) VALUES ($1, $2)",
        tenant_id,
        name,
    )
    return tenant_id


async def create_agent(
    pool: asyncpg.Pool,
    tenant_id: UUID,
    *,
    clearance_level: str = "internal",
    division: str = "all",
) -> UUID:
    agent_id = uuid4()
    await pool.execute(
        """
        INSERT INTO qortia_agents (id, tenant_id, clearance_level, division)
        VALUES ($1, $2, $3, $4)
        """,
        agent_id,
        tenant_id,
        clearance_level,
        division,
    )
    return agent_id


async def issue_api_key(pool: asyncpg.Pool, tenant_id: UUID) -> str:
    """Generate a new API key for a tenant. Returns the plaintext key ONCE —
    only its hash is stored, matching every SaaS API-key pattern."""
    plaintext_key = f"qortia_sk_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(plaintext_key)
    await pool.execute(
        "INSERT INTO qortia_api_keys (id, tenant_id, key_hash) VALUES ($1, $2, $3)",
        uuid4(),
        tenant_id,
        key_hash,
    )
    return plaintext_key


async def revoke_api_key(pool: asyncpg.Pool, key_id: UUID) -> None:
    await pool.execute(
        "UPDATE qortia_api_keys SET revoked_at = now() WHERE id = $1",
        key_id,
    )


async def _cli_create_tenant(args: argparse.Namespace) -> None:
    pool = await asyncpg.create_pool(config.settings.database_url)
    try:
        tenant_id = await create_tenant(pool, name=args.name)
        print(f"tenant_id: {tenant_id}")
    finally:
        await pool.close()


async def _cli_create_agent(args: argparse.Namespace) -> None:
    pool = await asyncpg.create_pool(config.settings.database_url)
    try:
        agent_id = await create_agent(
            pool,
            UUID(args.tenant),
            clearance_level=args.clearance,
            division=args.division,
        )
        print(f"agent_id: {agent_id}")
    finally:
        await pool.close()


async def _cli_issue_key(args: argparse.Namespace) -> None:
    pool = await asyncpg.create_pool(config.settings.database_url)
    try:
        key = await issue_api_key(pool, UUID(args.tenant))
        print(f"api_key: {key}")
        print("Store this now — it will not be shown again.")
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="qortia-admin")
    sub = parser.add_subparsers(dest="command", required=True)

    p_tenant = sub.add_parser("create-tenant", help="Create a new tenant")
    p_tenant.add_argument("--name", default=None)
    p_tenant.set_defaults(func=_cli_create_tenant)

    p_agent = sub.add_parser("create-agent", help="Create a new agent under a tenant")
    p_agent.add_argument("--tenant", required=True, help="Tenant UUID")
    p_agent.add_argument("--clearance", default="internal")
    p_agent.add_argument("--division", default="all")
    p_agent.set_defaults(func=_cli_create_agent)

    p_key = sub.add_parser("issue-key", help="Issue a new API key for a tenant")
    p_key.add_argument("--tenant", required=True, help="Tenant UUID")
    p_key.set_defaults(func=_cli_issue_key)

    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
