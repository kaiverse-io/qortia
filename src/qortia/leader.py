"""Postgres-advisory-lock leader election for singleton background jobs.

Replaces the old host-platform `app.background.leader`. Fully portable —
`pg_try_advisory_lock`/`pg_advisory_unlock` are plain Postgres built-ins, no
external coordination service needed. Whichever process (of possibly many
replicas) acquires the lock runs the job; everyone else skips this cycle.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg

LOCK_KEY_WEEKLY_SUMMARY = 7264_1001  # arbitrary stable int, unique per named job


@asynccontextmanager
async def try_acquire_leader(pool: asyncpg.Pool, lock_key: int) -> AsyncGenerator[bool, None]:
    async with pool.acquire() as conn:
        acquired: bool = await conn.fetchval("SELECT pg_try_advisory_lock($1)", lock_key)
        try:
            yield acquired
        finally:
            if acquired:
                await conn.execute("SELECT pg_advisory_unlock($1)", lock_key)
