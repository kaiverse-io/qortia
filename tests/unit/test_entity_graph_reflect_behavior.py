"""Unit tests for entity graph summary maintenance and reflect background tasks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

TENANT_ID = uuid4()
AGENT_ID = uuid4()


class _AcquireContext:
    def __init__(self, conn: MagicMock) -> None:
        self.conn = conn

    async def __aenter__(self) -> MagicMock:
        return self.conn

    async def __aexit__(self, *_exc: object) -> None:
        return None


@pytest.mark.asyncio
async def test_update_entity_summary_bootstraps_without_llm() -> None:
    from qortia.entity_graph import _update_entity_summary

    summary = await _update_entity_summary(
        None, "Qortia chose PostgreSQL for durable storage", "unused-key"
    )
    assert summary == "Qortia chose PostgreSQL for durable storage"


@pytest.mark.asyncio
async def test_update_entity_summary_merges_with_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia.entity_graph import _update_entity_summary

    response = MagicMock()
    response.json.return_value = {"choices": [{"message": {"content": "Updated Qortia summary"}}]}
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("qortia.entity_graph.get_litellm_client", lambda: client)

    summary = await _update_entity_summary(
        "Old summary about Qortia",
        "New PostgreSQL deployment details for Qortia",
        "litellm-key",
    )
    assert summary == "Updated Qortia summary"
    client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_entity_summary_returns_existing_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.entity_graph import _update_entity_summary

    client = MagicMock()
    client.post = AsyncMock(side_effect=RuntimeError("down"))
    monkeypatch.setattr("qortia.entity_graph.get_litellm_client", lambda: client)

    existing = "Stable Qortia summary"
    summary = await _update_entity_summary(existing, "New info", "key")
    assert summary == existing


@pytest.mark.asyncio
async def test_maybe_update_entity_summary_bootstraps_on_first_link() -> None:
    from qortia.entity_graph import _maybe_update_entity_summary

    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value={"link_count": 1, "summary": None})
    conn.execute = AsyncMock()

    await _maybe_update_entity_summary(
        conn, TENANT_ID, AGENT_ID, "Qortia", "PostgreSQL graph memory content", is_org=False
    )
    conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_update_entity_summary_llm_refresh_every_third_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.entity_graph import _maybe_update_entity_summary

    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value={"link_count": 3, "summary": "Old"})
    conn.execute = AsyncMock()
    monkeypatch.setattr(
        "qortia.entity_graph._update_entity_summary",
        AsyncMock(return_value="Refreshed summary"),
    )
    monkeypatch.setattr("qortia.entity_graph.get_litellm_key", AsyncMock(return_value="tenant-key"))

    await _maybe_update_entity_summary(
        conn, TENANT_ID, AGENT_ID, "Qortia", "New PostgreSQL memory content", is_org=False
    )
    conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_populate_graph_batch_skips_invalid_entity_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.entity_graph import _populate_graph_batch

    memory_id = uuid4()
    conn = MagicMock()
    conn.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "id": memory_id,
                    "tenant_id": TENANT_ID,
                    "agent_id": AGENT_ID,
                    "entities": "not-json",
                    "content": "bad entities payload",
                }
            ],
            [],
        ]
    )
    conn.transaction.return_value = _AcquireContext(conn)
    conn.execute = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value = _AcquireContext(conn)
    monkeypatch.setattr("qortia.entity_graph.get_main_pool", lambda: pool)
    monkeypatch.setattr("qortia.entity_graph._maybe_update_entity_summary", AsyncMock())

    await _populate_graph_batch()
    assert conn.execute.await_count >= 1


@pytest.mark.asyncio
async def test_archive_old_episodic_memories_logs_when_rows_archived() -> None:
    from qortia.reflect import _archive_old_episodic_memories

    conn = MagicMock()
    conn.execute = AsyncMock(return_value="UPDATE 2")
    pool = MagicMock()
    pool.acquire.return_value = _AcquireContext(conn)

    with patch("qortia.reflect.get_main_pool", return_value=pool):
        await _archive_old_episodic_memories()


@pytest.mark.asyncio
async def test_purge_expired_short_term_memories_deletes_rows() -> None:
    from qortia.reflect import _purge_expired_short_term_memories

    conn = MagicMock()
    conn.execute = AsyncMock(return_value="DELETE 1")
    pool = MagicMock()
    pool.acquire.return_value = _AcquireContext(conn)

    with patch("qortia.reflect.get_main_pool", return_value=pool):
        await _purge_expired_short_term_memories()


@pytest.mark.asyncio
async def test_validate_embedding_dimensions_accepts_1024(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qortia.config as config
    from qortia.reflect import validate_embedding_dimensions

    monkeypatch.setattr(config.settings, "litellm_api_key", "embed-key")
    monkeypatch.setattr(config.settings, "embedding_dimension", 1024)
    response = MagicMock()
    response.json.return_value = {"data": [{"embedding": [0.1] * 1024}]}
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("qortia.embeddings.get_litellm_client", lambda: client)
    monkeypatch.setattr("qortia.auth.get_platform_embed_key", lambda: "embed-key")

    await validate_embedding_dimensions()


@pytest.mark.asyncio
async def test_validate_embedding_dimensions_raises_on_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qortia.config as config
    from qortia.reflect import validate_embedding_dimensions

    monkeypatch.setattr(config.settings, "litellm_api_key", "embed-key")
    monkeypatch.setattr(config.settings, "embedding_dimension", 1024)
    response = MagicMock()
    response.json.return_value = {"data": [{"embedding": [0.1] * 512}]}
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("qortia.embeddings.get_litellm_client", lambda: client)
    monkeypatch.setattr("qortia.auth.get_platform_embed_key", lambda: "embed-key")

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        await validate_embedding_dimensions()


@pytest.mark.asyncio
async def test_trigger_idle_reflections_invokes_reflect_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.reflect import _trigger_idle_reflections

    agent_id = uuid4()
    tenant_id = uuid4()
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[{"agent_id": agent_id, "tenant_id": tenant_id}])
    pool = MagicMock()
    pool.acquire.return_value = _AcquireContext(conn)
    reflect_agent = AsyncMock()
    monkeypatch.setattr("qortia.reflect.get_main_pool", lambda: pool)
    monkeypatch.setattr("qortia.reflect._reflect_agent", reflect_agent)

    await _trigger_idle_reflections()
    reflect_agent.assert_awaited_once_with(agent_id, tenant_id)
