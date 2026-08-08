"""Tenant/agent/API-key provisioning — core functions, called from two places.

The very first API key for a fresh tenant can't be created through a
*tenant-scoped* API that itself requires an existing tenant API key to
authenticate the request — that bootstrapping problem has no self-service
answer. The `qortia-admin` CLI (direct DB access, `main()` below) is the
original out-of-band answer to it, and stays the only path that needs
nothing but a Postgres connection string.

`qortia.admin_router` (ADR-004, docs/decisions/adrs/) is a second caller of
these same functions, for operators who need to provision over HTTP because
they have no shell into wherever Qortia runs — e.g. a separate control-plane
service in its own container. It does not reopen the bootstrapping problem
above: it's gated by a static platform-level `QORTIA_ADMIN_TOKEN` set
out-of-band by whoever deploys Qortia, not a per-tenant API key, so there is
still nothing self-service about it — see `qortia.auth.require_admin`.

Both callers share these functions rather than duplicating the SQL; neither
goes through `qortia.db.tenant_transaction` (RLS), because
`qortia_tenants`/`qortia_agents`/`qortia_api_keys` carry no RLS policies —
provisioning is inherently a cross-tenant, superuser-ish operation, same as
it always was for the CLI.
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
    name: str | None = None,
    clearance_level: str = "internal",
    division: str = "all",
) -> UUID:
    agent_id = uuid4()
    await pool.execute(
        """
        INSERT INTO qortia_agents (id, tenant_id, name, clearance_level, division)
        VALUES ($1, $2, $3, $4, $5)
        """,
        agent_id,
        tenant_id,
        name,
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
            name=args.name,
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
    p_agent.add_argument("--name", default=None)
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
