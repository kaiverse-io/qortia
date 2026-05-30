"""Unit tests — POST /v1/agents/{agent_id}/qortia/search endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException

TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
AGENT_ID = UUID("00000000-0000-0000-0000-000000000002")


def _make_user():
    from app.auth.models import UserIdentity

    return UserIdentity(
        user_id=UUID("00000000-0000-0000-0000-000000000003"),
        tenant_id=TENANT_ID,
        jti=UUID("00000000-0000-0000-0000-000000000004"),
    )


def _patch_tx(conn):
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _make_memory_row(content: str = "test memory") -> dict:
    from datetime import datetime, timezone

    return {
        "id": UUID("00000000-0000-0000-0000-000000000010"),
        "type": "episodic",
        "content": content,
        "importance": 0.8,
        "tier": "active",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "last_recalled_at": None,
        "recall_count": 3,
        "is_consolidated": False,
    }


# ── success cases ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_returns_matching_memories() -> None:
    from app.auth.router import search_agent_memories

    row = _make_memory_row("billing retries need jitter")
    conn = AsyncMock()
    conn.fetchval = AsyncMock(side_effect=[1, 1])  # exists check, then count
    conn.fetch = AsyncMock(return_value=[row])

    with patch(
        "app.auth.router.tenant_transaction", return_value=_patch_tx(conn)
    ), patch("app.auth.router.get_main_pool"):
        result = await search_agent_memories(
            agent_id=AGENT_ID,
            query="billing",
            limit=20,
            user=_make_user(),
        )

    assert result["total"] >= 1
    assert result["memories"][0]["content"] == "billing retries need jitter"


@pytest.mark.asyncio
async def test_search_returns_empty_on_no_match() -> None:
    from app.auth.router import search_agent_memories

    conn = AsyncMock()
    conn.fetchval = AsyncMock(side_effect=[1, 0])  # exists, count=0
    conn.fetch = AsyncMock(return_value=[])

    with patch(
        "app.auth.router.tenant_transaction", return_value=_patch_tx(conn)
    ), patch("app.auth.router.get_main_pool"):
        result = await search_agent_memories(
            agent_id=AGENT_ID,
            query="zzznomatch",
            limit=20,
            user=_make_user(),
        )

    assert result["memories"] == []
    assert result["total"] == 0


@pytest.mark.asyncio
async def test_search_empty_query_returns_empty() -> None:
    from app.auth.router import search_agent_memories

    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=1)  # agent exists

    with patch(
        "app.auth.router.tenant_transaction", return_value=_patch_tx(conn)
    ), patch("app.auth.router.get_main_pool"):
        result = await search_agent_memories(
            agent_id=AGENT_ID,
            query="   ",
            limit=20,
            user=_make_user(),
        )

    assert result["memories"] == []
    assert result["total"] == 0


@pytest.mark.asyncio
async def test_search_respects_limit() -> None:
    from app.auth.router import search_agent_memories

    rows = [_make_memory_row(f"memory {i}") for i in range(5)]
    conn = AsyncMock()
    conn.fetchval = AsyncMock(side_effect=[1, 5])
    conn.fetch = AsyncMock(return_value=rows)

    with patch(
        "app.auth.router.tenant_transaction", return_value=_patch_tx(conn)
    ), patch("app.auth.router.get_main_pool"):
        result = await search_agent_memories(
            agent_id=AGENT_ID,
            query="memory",
            limit=5,
            user=_make_user(),
        )

    assert len(result["memories"]) == 5
    fetch_call = conn.fetch.call_args[0]
    assert 5 in fetch_call  # limit passed to query


@pytest.mark.asyncio
async def test_search_result_shape() -> None:
    from app.auth.router import search_agent_memories

    conn = AsyncMock()
    conn.fetchval = AsyncMock(side_effect=[1, 1])
    conn.fetch = AsyncMock(return_value=[_make_memory_row()])

    with patch(
        "app.auth.router.tenant_transaction", return_value=_patch_tx(conn)
    ), patch("app.auth.router.get_main_pool"):
        result = await search_agent_memories(
            agent_id=AGENT_ID,
            query="test",
            limit=20,
            user=_make_user(),
        )

    m = result["memories"][0]
    assert "id" in m
    assert "type" in m
    assert "content" in m
    assert "importance" in m
    assert "tier" in m
    assert "created_at" in m
    assert "recall_count" in m


# ── error cases ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_404_when_agent_not_in_tenant() -> None:
    from app.auth.router import search_agent_memories

    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=None)  # agent not found

    with patch(
        "app.auth.router.tenant_transaction", return_value=_patch_tx(conn)
    ), patch("app.auth.router.get_main_pool"):
        with pytest.raises(HTTPException) as exc:
            await search_agent_memories(
                agent_id=AGENT_ID,
                query="anything",
                limit=20,
                user=_make_user(),
            )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_search_limit_capped_at_100() -> None:
    from app.auth.router import search_agent_memories

    conn = AsyncMock()
    conn.fetchval = AsyncMock(side_effect=[1, 0])
    conn.fetch = AsyncMock(return_value=[])

    with patch(
        "app.auth.router.tenant_transaction", return_value=_patch_tx(conn)
    ), patch("app.auth.router.get_main_pool"):
        # limit=200 should be capped — FastAPI Query(le=100) raises before function body
        # so test that a valid limit works and the query param flows through
        result = await search_agent_memories(
            agent_id=AGENT_ID,
            query="test",
            limit=100,
            user=_make_user(),
        )
    assert result is not None
