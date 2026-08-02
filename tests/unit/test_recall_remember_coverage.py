"""Unit tests for recall, remember forget paths, and links error handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from qortia.auth import AgentIdentity
from qortia.models import ForgetRequest, RecallRequest, RecallResult

TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
AGENT_ID = UUID("00000000-0000-0000-0000-000000000002")


class _AcquireContext:
    def __init__(self, conn: MagicMock) -> None:
        self.conn = conn

    async def __aenter__(self) -> MagicMock:
        return self.conn

    async def __aexit__(self, *_exc: object) -> None:
        return None


def _tenant_tx(conn: MagicMock) -> _AcquireContext:
    return _AcquireContext(conn)


def _result(rid: str, scope: str = "private") -> RecallResult:
    return RecallResult(
        id=rid,
        type="episodic",
        scope=scope,
        content="Qortia PostgreSQL retrieval memory content",
        importance=0.5,
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_embed_query_returns_cached_embedding_without_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.recall import _embed_query

    cached = [0.2] * 1024
    monkeypatch.setattr("qortia.embeddings.get_cached_embedding", lambda *_a, **_k: cached)
    post = AsyncMock()
    monkeypatch.setattr("qortia.embeddings.get_litellm_client", lambda: MagicMock(post=post))

    embedding = await _embed_query("database", TENANT_ID)
    assert embedding == cached
    post.assert_not_called()


@pytest.mark.asyncio
async def test_embed_query_returns_none_when_litellm_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.recall import _embed_query

    monkeypatch.setattr("qortia.embeddings.get_cached_embedding", lambda *_a, **_k: None)
    client = MagicMock()
    client.post = AsyncMock(side_effect=RuntimeError("down"))
    monkeypatch.setattr("qortia.embeddings.get_litellm_client", lambda: client)
    monkeypatch.setattr("qortia.embeddings.get_litellm_key", AsyncMock(return_value="key"))
    monkeypatch.setattr("qortia.embeddings.put_cached_embedding", lambda *_a, **_k: None)

    assert await _embed_query("database", TENANT_ID) is None


@pytest.mark.parametrize(
    ("memory_type", "scope", "expect_empty"),
    [
        ("decision", "org", True),
        ("lesson", "org", True),
        ("episodic", "org", True),
        ("short_term", "archive", True),
    ],
)
@pytest.mark.asyncio
async def test_type_routed_recall_returns_empty_for_invalid_scope(
    memory_type: str, scope: str, expect_empty: bool
) -> None:
    from qortia import recall as recall_mod

    body = RecallRequest(query="database", scope=scope, type=memory_type)  # type: ignore[arg-type]
    agent = AgentIdentity(agent_id=AGENT_ID, tenant_id=TENANT_ID)
    fn = {
        "decision": recall_mod._recall_decisions,
        "lesson": recall_mod._recall_lessons,
        "episodic": recall_mod._recall_episodic,
        "short_term": recall_mod._recall_short_term,
    }[memory_type]
    if memory_type == "short_term" and scope == "archive":
        with pytest.raises(ValueError, match="not supported"):
            await fn(body, agent)
        return
    results = await fn(body, agent)
    assert (len(results) == 0) is expect_empty


@pytest.mark.asyncio
async def test_record_work_order_outcome_updates_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.recall import _record_work_order_outcome

    memory_id = str(uuid4())
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[{"memory_id": memory_id}])
    conn.execute = AsyncMock()
    monkeypatch.setattr("qortia.recall.get_main_pool", lambda: MagicMock())
    monkeypatch.setattr("qortia.recall.tenant_transaction", lambda *_a, **_k: _tenant_tx(conn))

    work_order_id = uuid4()
    await _record_work_order_outcome(
        work_order_id,
        TENANT_ID,
        AGENT_ID,
        "SUCCESS",
        memory_clearance_order=2,
        agent_division="all",
    )
    assert conn.execute.await_count == 2


@pytest.mark.asyncio
async def test_log_session_reads_inserts_private_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia.recall import _log_session_reads

    conn = MagicMock()
    conn.execute = AsyncMock()
    monkeypatch.setattr("qortia.recall.get_main_pool", lambda: MagicMock())
    monkeypatch.setattr("qortia.recall.tenant_transaction", lambda *_a, **_k: _tenant_tx(conn))

    await _log_session_reads(
        [_result("one"), _result("two", scope="org")],
        TENANT_ID,
        AGENT_ID,
        uuid4(),
    )
    conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_recall_endpoint_logs_invalid_work_order_without_session_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.recall import recall

    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value={"status": "active"})
    monkeypatch.setattr("qortia.recall.tenant_transaction", lambda *_a, **_k: _tenant_tx(conn))
    monkeypatch.setattr("qortia.recall.get_main_pool", lambda: object())
    monkeypatch.setattr("qortia.common.assert_agent_active", AsyncMock())
    monkeypatch.setattr(
        "qortia.recall._hybrid_recall_pipeline", AsyncMock(return_value=[_result("one")])
    )
    monkeypatch.setattr("qortia.recall._record_recall_access", AsyncMock())
    log_reads = AsyncMock()
    monkeypatch.setattr("qortia.recall._log_session_reads", log_reads)

    agent = AgentIdentity(agent_id=AGENT_ID, tenant_id=TENANT_ID)
    await recall(RecallRequest(query="database"), agent, x_work_order_id="not-a-uuid")

    log_reads.assert_not_called()


@pytest.mark.asyncio
async def test_forget_org_process_requires_chief(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia.remember import forget

    conn = MagicMock()
    conn.fetchrow = AsyncMock(
        side_effect=[
            None,
            {
                "id": uuid4(),
                "author_id": AGENT_ID,
                "type": "process",
                "content": "process content with enough words here",
            },
        ]
    )
    conn.fetchval = AsyncMock(return_value="engineer")
    monkeypatch.setattr("qortia.remember.get_main_pool", lambda: MagicMock())
    monkeypatch.setattr("qortia.remember.tenant_transaction", lambda *_a, **_k: _tenant_tx(conn))
    monkeypatch.setattr("qortia.remember.assert_agent_active", AsyncMock())

    agent = AgentIdentity(agent_id=AGENT_ID, tenant_id=TENANT_ID)
    with pytest.raises(HTTPException) as exc:
        await forget(ForgetRequest(id=str(uuid4())), agent)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_find_similar_memories_returns_empty_on_db_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.links import _find_similar_memories

    monkeypatch.setattr(
        "qortia.links.tenant_transaction",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("db down")),
    )
    # tenant_transaction is context manager - need proper mock
    broken = MagicMock()
    broken.__aenter__ = AsyncMock(side_effect=RuntimeError("db down"))
    broken.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("qortia.links.tenant_transaction", lambda *_a, **_k: broken)

    similar = await _find_similar_memories(uuid4(), [0.1] * 1024, TENANT_ID, AGENT_ID)
    assert similar == []


@pytest.mark.asyncio
async def test_expand_with_links_returns_original_on_db_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.links import _expand_with_links

    original = [_result("only")]
    broken = MagicMock()
    broken.__aenter__ = AsyncMock(side_effect=RuntimeError("db down"))
    broken.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("qortia.links.tenant_transaction", lambda *_a, **_k: broken)

    expanded = await _expand_with_links(original, TENANT_ID, AGENT_ID)
    assert expanded == original
