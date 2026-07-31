"""Standalone Postgres pool + tenant-scoped RLS transaction helper.

Replaces the old host-platform `app.db`. `tenant_transaction` implements the
RLS session-variable pattern documented in docs/01-design.md §4.4: every
memory operation runs inside a transaction with `app.tenant_id`/`app.agent_id`/
`app.memory_clearance_order`/`app.agent_division` set as Postgres session GUCs,
which the RLS policies on every tenant-scoped table read via `current_setting()`.

`set_config(name, value, is_local=true)` is used instead of raw `SET LOCAL
... = '...'` string interpolation so every value is passed as a bound
parameter — no f-string SQL construction, even for tenant-controlled fields
like agent_division.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg

from qortia import config

_pool: asyncpg.Pool | None = None


async def init_main_pool() -> None:
    global _pool
    _pool = await asyncpg.create_pool(config.settings.database_url)


async def close_main_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_main_pool() -> asyncpg.Pool:
    assert _pool is not None, "Postgres pool not initialised — call init_main_pool() at startup"  # noqa: S101
    return _pool


@asynccontextmanager
async def tenant_transaction(
    pool: asyncpg.Pool,
    tenant_id: UUID,
    agent_id: UUID | None = None,
    memory_clearance_order: int | None = None,
    agent_division: str | None = None,
) -> AsyncGenerator[asyncpg.Connection, None]:
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(tenant_id))
        if agent_id is not None:
            await conn.execute("SELECT set_config('app.agent_id', $1, true)", str(agent_id))
        order = memory_clearance_order if memory_clearance_order is not None else 2
        division = agent_division or "all"
        await conn.execute("SELECT set_config('app.memory_clearance_order', $1, true)", str(order))
        await conn.execute("SELECT set_config('app.agent_division', $1, true)", division)
        yield conn
