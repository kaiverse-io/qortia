"""
Unit tests — qortia memory quality (Phase 3)
Covers: content floor validators, _maybe_dedup_memory, short_term skip, hash dedup.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from qortia.models import MemoryItem, RememberOrgRequest

# ── G2: Content length floor ─────────────────────────────────


def test_memory_item_4_words_raises() -> None:
    with pytest.raises(Exception, match="5 words"):
        MemoryItem(type="episodic", content="only four words here", ttl_seconds=None)


def test_memory_item_5_words_passes() -> None:
    m = MemoryItem(type="episodic", content="this has exactly five words")
    assert m.content == "this has exactly five words"


def test_memory_item_empty_raises() -> None:
    with pytest.raises(Exception, match="must not be empty"):
        MemoryItem(type="episodic", content="   ")


def test_remember_org_9_words_raises() -> None:
    with pytest.raises(Exception, match="10 words"):
        RememberOrgRequest(
            type="handoff",
            title="Test",
            content="only nine words in this content here now",
        )


def test_remember_org_10_words_passes() -> None:
    r = RememberOrgRequest(
        type="handoff",
        title="Test",
        content="this content has exactly ten words in it right now",
    )
    assert r.content.startswith("this content")


# ── G3: Post-embed dedup ──────────────────────────────────────


@pytest.mark.asyncio
async def test_maybe_dedup_archives_when_similarity_above_threshold() -> None:
    from qortia.reflect import _maybe_dedup_memory

    memory_id = uuid4()
    neighbour_id = uuid4()
    embedding = [0.1] * 1024

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"id": neighbour_id, "similarity": 0.96})
    mock_conn.execute = AsyncMock()

    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("qortia.entity_graph.get_main_pool", return_value=mock_pool):
        await _maybe_dedup_memory(memory_id, embedding, uuid4(), uuid4(), "episodic")

    mock_conn.execute.assert_called_once()
    call_args = mock_conn.execute.call_args[0]
    assert "tier = 'archive'" in call_args[0]
    metadata = json.loads(call_args[1])
    assert metadata["dedup_of"] == str(neighbour_id)


@pytest.mark.asyncio
async def test_maybe_dedup_no_archive_when_similarity_below_threshold() -> None:
    from qortia.reflect import _maybe_dedup_memory

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"id": uuid4(), "similarity": 0.93})
    mock_conn.execute = AsyncMock()

    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("qortia.entity_graph.get_main_pool", return_value=mock_pool):
        await _maybe_dedup_memory(uuid4(), [0.1] * 1024, uuid4(), uuid4(), "episodic")

    mock_conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_dedup_no_op_when_no_neighbour() -> None:
    from qortia.reflect import _maybe_dedup_memory

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock()

    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("qortia.entity_graph.get_main_pool", return_value=mock_pool):
        await _maybe_dedup_memory(uuid4(), [0.1] * 1024, uuid4(), uuid4(), "episodic")

    mock_conn.execute.assert_not_called()


# ── G5: short_term skip ───────────────────────────────────────


@pytest.mark.asyncio
async def test_short_term_skips_embedding() -> None:
    from qortia.reflect import _embed_single_row

    row = {
        "id": uuid4(),
        "tbl": "hindsight_memories",
        "type": "short_term",
        "text_to_embed": "this is a short term memory with enough words",
        "tenant_id": uuid4(),
        "agent_id": uuid4(),
    }

    with patch("qortia.reflect._get_embedding") as mock_embed:
        await _embed_single_row(row, "fake-key")
        mock_embed.assert_not_called()


@pytest.mark.asyncio
async def test_episodic_calls_maybe_dedup_after_embedding() -> None:
    from qortia.reflect import _embed_single_row

    row = {
        "id": uuid4(),
        "tbl": "hindsight_memories",
        "type": "episodic",
        "text_to_embed": "this is an episodic memory with enough words",
        "tenant_id": uuid4(),
        "agent_id": uuid4(),
    }
    fake_embedding = [0.1] * 1024

    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("qortia.reflect._get_embedding", return_value=fake_embedding),
        patch("qortia.reflect.get_main_pool", return_value=mock_pool),
        patch("qortia.links._find_similar_memories", return_value=[]),
        patch("qortia.links._upsert_memory_links"),
        patch("qortia.reflect._maybe_dedup_memory") as mock_dedup,
    ):
        await _embed_single_row(row, "fake-key")
        mock_dedup.assert_called_once()


@pytest.mark.asyncio
async def test_decision_does_not_call_maybe_dedup() -> None:
    from qortia.reflect import _embed_single_row

    row = {
        "id": uuid4(),
        "tbl": "hindsight_memories",
        "type": "decision",
        "text_to_embed": "this is a decision memory with enough words",
        "tenant_id": uuid4(),
        "agent_id": uuid4(),
    }
    fake_embedding = [0.1] * 1024

    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("qortia.reflect._get_embedding", return_value=fake_embedding),
        patch("qortia.reflect.get_main_pool", return_value=mock_pool),
        patch("qortia.links._find_similar_memories", return_value=[]),
        patch("qortia.links._upsert_memory_links"),
        patch("qortia.reflect._maybe_dedup_memory") as mock_dedup,
    ):
        await _embed_single_row(row, "fake-key")
        mock_dedup.assert_not_called()


# ── G4: Content hash dedup in remember() ─────────────────────


@pytest.mark.asyncio
async def test_content_hash_dedup_returns_existing_id() -> None:
    """Same content within 24h returns existing id, no new insert."""
    from qortia.auth import AgentIdentity
    from qortia.models import RememberRequest
    from qortia.remember import remember

    existing_id = uuid4()
    agent = AgentIdentity(
        agent_id=uuid4(),
        tenant_id=uuid4(),
    )
    body = RememberRequest(
        memories=[MemoryItem(type="episodic", content="this is a test episodic memory content")]
    )

    mock_conn = AsyncMock()
    # assert_agent_active check
    mock_conn.fetchrow = AsyncMock(return_value={"status": "active"})
    # content hash dedup returns existing
    mock_conn.fetchval = AsyncMock(return_value=existing_id)

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("qortia.remember.tenant_transaction", return_value=mock_ctx),
        patch("qortia.remember.get_main_pool"),
        patch("qortia.remember.extract_entities_with_types", return_value=[]),
    ):
        result = await remember(body, agent)

    assert str(existing_id) in result.ids
    # No INSERT should have been called (fetchval returned existing, no second fetchval for insert)


@pytest.mark.asyncio
async def test_content_hash_dedup_inserts_when_no_existing() -> None:
    """No existing hash → proceeds with insert."""
    from qortia.auth import AgentIdentity
    from qortia.models import RememberRequest
    from qortia.remember import remember

    new_id = uuid4()
    agent = AgentIdentity(
        agent_id=uuid4(),
        tenant_id=uuid4(),
    )
    body = RememberRequest(
        memories=[MemoryItem(type="episodic", content="this is a unique episodic memory content")]
    )

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"status": "active"})
    # First fetchval: dedup check returns None; second: INSERT returns new_id
    mock_conn.fetchval = AsyncMock(side_effect=[None, new_id])
    mock_conn.execute = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("qortia.remember.tenant_transaction", return_value=mock_ctx),
        patch("qortia.remember.get_main_pool"),
        patch("qortia.remember.extract_entities_with_types", return_value=[]),
    ):
        result = await remember(body, agent)

    assert str(new_id) in result.ids
