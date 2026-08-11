from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from qortia import config
from qortia.auth import AgentIdentity
from qortia.models import RecallResult

TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
AGENT_ID = UUID("00000000-0000-0000-0000-000000000002")


def _recall_result(rid: str, content: str) -> RecallResult:
    return RecallResult(
        id=rid,
        type="episodic",
        scope="private",
        content=content,
        importance=0.5,
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_embedding_cache_is_tenant_and_language_scoped(caplog: pytest.LogCaptureFixture) -> None:
    from qortia.embedding_cache import (
        clear_all_caches,
        get_cache_stats,
        get_cached_embedding,
        put_cached_embedding,
    )

    clear_all_caches()
    put_cached_embedding("Same Query", "tenant-a", "en", [0.1, 0.2])

    with caplog.at_level("DEBUG"):
        assert get_cached_embedding(" same query ", "tenant-a", "en") == [0.1, 0.2]

    assert get_cached_embedding("same query", "tenant-a", "hi") is None
    assert get_cached_embedding("same query", "tenant-b", "en") is None
    # Misses do not allocate tenant caches — only the put + hit tenant exists,
    # plus any tenant that received a lookup that created an empty cache entry.
    stats = get_cache_stats()
    assert stats["total_entries"] == 1
    assert stats["tenant_count"] >= 1
    assert any(record.msg.get("event") == "embedding_cache_hit" for record in caplog.records)

    clear_all_caches()
    assert get_cache_stats() == {"tenant_count": 0, "total_entries": 0}


class _AcquireContext:
    def __init__(self, conn: MagicMock) -> None:
        self.conn = conn

    async def __aenter__(self) -> MagicMock:
        return self.conn

    async def __aexit__(self, *_exc: object) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("acquired", [True, False])
async def test_try_acquire_leader_yields_lock_state_and_only_unlocks_when_acquired(
    acquired: bool,
) -> None:
    from qortia.leader import try_acquire_leader

    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=acquired)
    conn.execute = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value = _AcquireContext(conn)

    async with try_acquire_leader(pool, 1234) as is_leader:
        assert is_leader is acquired

    conn.fetchval.assert_awaited_once_with("SELECT pg_try_advisory_lock($1)", 1234)
    if acquired:
        conn.execute.assert_awaited_once_with("SELECT pg_advisory_unlock($1)", 1234)
    else:
        conn.execute.assert_not_called()


def test_telemetry_make_counter_falls_back_to_noop_when_optional_dependency_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia import telemetry

    monkeypatch.setitem(sys.modules, "opentelemetry", None)

    counter = telemetry._make_counter("qortia.test.noop", "no-op fallback")

    assert counter.add(1, {"reason": "missing"}) is None


def test_telemetry_make_counter_uses_opentelemetry_meter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia import telemetry

    created = MagicMock()
    meter = MagicMock()
    meter.create_counter.return_value = created
    metrics_mod = types.SimpleNamespace(get_meter=MagicMock(return_value=meter))
    otel_mod = types.SimpleNamespace(metrics=metrics_mod)
    monkeypatch.setitem(sys.modules, "opentelemetry", otel_mod)

    counter = telemetry._make_counter("qortia.test.counter", "real counter")

    assert counter is created
    metrics_mod.get_meter.assert_called_once_with("qortia")
    meter.create_counter.assert_called_once_with("qortia.test.counter", description="real counter")


@pytest.mark.asyncio
async def test_llm_rerank_empty_results_returns_without_litellm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.recall_rerank import _llm_rerank

    get_key = AsyncMock()
    monkeypatch.setattr("qortia.recall_rerank.get_litellm_key", get_key)

    result = await _llm_rerank("query", [], AgentIdentity(AGENT_ID, TENANT_ID))

    assert result == []
    get_key.assert_not_called()


@pytest.mark.asyncio
async def test_llm_rerank_uses_model_order_and_appends_omitted_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.recall_rerank import _llm_rerank

    # rerank_model defaults to "" (not configured, skip) — this test is
    # specifically about the reranking-happens path, so it must configure
    # one explicitly rather than rely on a default that used to be non-empty.
    monkeypatch.setattr(config.settings, "rerank_model", "test-rerank-model")
    first = _recall_result("first", "first memory")
    second = _recall_result("second", "second memory")
    third = _recall_result("third", "third memory")

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"message": {"content": "[3, 1]"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("qortia.chat.get_litellm_client", lambda: client)
    monkeypatch.setattr("qortia.recall_rerank.get_litellm_key", AsyncMock(return_value="key"))

    reranked = await _llm_rerank(
        "query", [first, second, third], AgentIdentity(AGENT_ID, TENANT_ID)
    )

    assert [r.id for r in reranked] == ["third", "first", "second"]
    payload = client.post.await_args.kwargs["json"]
    assert payload["model"]
    assert "Include all 3 numbers" in payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_llm_rerank_skips_the_network_call_when_no_model_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rerank_model="" (the default — see config.Settings) means 'not
    configured', not 'call litellm with an empty model string and let it
    fail'. Before this guard, every rerank=True call paid a network
    round-trip guaranteed to fail and logged a misleading rerank_failed
    warning for a state that is deliberate, not a failure."""
    from qortia.recall_rerank import _llm_rerank

    monkeypatch.setattr(config.settings, "rerank_model", "")
    results = [_recall_result("one", "one"), _recall_result("two", "two")]
    client = MagicMock()
    client.post = AsyncMock()
    monkeypatch.setattr("qortia.chat.get_litellm_client", lambda: client)

    out = await _llm_rerank("query", results, AgentIdentity(AGENT_ID, TENANT_ID))

    assert out == results
    client.post.assert_not_called()


@pytest.mark.asyncio
async def test_llm_rerank_returns_original_order_on_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.recall_rerank import _llm_rerank

    # Must configure a model, or the empty-model guard returns early and this
    # test would pass vacuously without ever reaching the malformed-response
    # handling it's meant to exercise.
    monkeypatch.setattr(config.settings, "rerank_model", "test-rerank-model")
    results = [_recall_result("one", "one"), _recall_result("two", "two")]
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"choices": [{"message": {"content": "not json"}}]}
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("qortia.chat.get_litellm_client", lambda: client)
    monkeypatch.setattr("qortia.recall_rerank.get_litellm_key", AsyncMock(return_value="key"))

    assert await _llm_rerank("query", results, AgentIdentity(AGENT_ID, TENANT_ID)) == results


class _TenantTx:
    def __init__(self, rows_by_call: list[list[dict[str, object]]]) -> None:
        self.rows_by_call = rows_by_call
        self.call_count = 0

    def __call__(self, *_args: object, **_kwargs: object) -> _AcquireContext:
        rows = self.rows_by_call[self.call_count]
        self.call_count += 1
        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=rows)
        return _AcquireContext(conn)


@pytest.mark.asyncio
async def test_bfs_entity_traversal_walks_cooccurring_entities_with_decay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.recall_rerank import _bfs_entity_traversal

    seed = UUID("00000000-0000-0000-0000-000000000010")
    next_entity = UUID("00000000-0000-0000-0000-000000000011")
    linked_memory = UUID("00000000-0000-0000-0000-000000000012")
    tx = _TenantTx(
        [
            [{"id": next_entity, "linked_memory_ids": [linked_memory], "similarity": 0.8}],
            [],
        ]
    )
    monkeypatch.setattr("qortia.recall_rerank.tenant_transaction", tx)
    monkeypatch.setattr("qortia.recall_rerank.get_main_pool", lambda: object())

    boosts = await _bfs_entity_traversal([0.1] * 1024, [seed], TENANT_ID, AGENT_ID)

    assert boosts == {str(linked_memory): pytest.approx(0.2)}


@pytest.mark.asyncio
async def test_bfs_entity_traversal_empty_seed_returns_no_boosts() -> None:
    from qortia.recall_rerank import _bfs_entity_traversal

    assert await _bfs_entity_traversal([0.1] * 1024, [], TENANT_ID, AGENT_ID) == {}
