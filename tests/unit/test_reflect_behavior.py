from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
AGENT_ID = UUID("00000000-0000-0000-0000-000000000002")


class _AsyncContext:
    def __init__(self, value: object = None) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *_exc: object) -> None:
        return None


def _tenant_tx(conn: MagicMock) -> _AsyncContext:
    return _AsyncContext(conn)


def _pool(conn: MagicMock) -> MagicMock:
    pool = MagicMock()
    pool.acquire.return_value = _AsyncContext(conn)
    return pool


def _litellm_response(content: str, *, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "choices": [{"message": {"content": content}}],
    }
    return response


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"reflections": []}, "empty reflections"),
        ({"reflections": [{"action": "DELETE"}]}, "invalid action"),
        ({"reflections": [{"action": "UPDATE", "type": "lesson"}]}, "id required"),
        (
            {
                "reflections": [
                    {"action": "CREATE", "type": "episodic", "content": "x", "importance": 0.4}
                ]
            },
            "invalid type",
        ),
        (
            {
                "reflections": [
                    {
                        "action": "CREATE",
                        "type": "lesson",
                        "content": "x",
                        "importance": "high",
                    }
                ]
            },
            "importance",
        ),
        (
            {
                "reflections": [
                    {"action": "CREATE", "type": "lesson", "content": "", "importance": 0.4}
                ]
            },
            "content",
        ),
    ],
)
@pytest.mark.asyncio
async def test_call_litellm_reflect_rejects_malformed_reflection_contract(
    payload: dict[str, object],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from qortia.reflect import _call_litellm_reflect

    client = MagicMock()
    client.post = AsyncMock(return_value=_litellm_response(json.dumps(payload)))
    monkeypatch.setattr("qortia.reflect.get_litellm_client", lambda: client)

    with caplog.at_level("ERROR"), pytest.raises(HTTPException) as exc:
        await _call_litellm_reflect(
            "model",
            ["recent memory"],
            [],
            "key",
            AGENT_ID,
            TENANT_ID,
        )

    assert exc.value.status_code == 500
    assert any(message in str(record.msg) for record in caplog.records)


@pytest.mark.asyncio
async def test_call_litellm_reflect_returns_valid_create_update_retain_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.reflect import _call_litellm_reflect

    existing_id = str(uuid4())
    payload = {
        "reflections": [
            {
                "action": "CREATE",
                "type": "lesson",
                "content": "Keep migrations append only",
                "importance": 0.9,
            },
            {
                "action": "UPDATE",
                "id": existing_id,
                "type": "mental_model",
                "content": "Qortia stores portable agent memory",
                "importance": 0.8,
            },
            {"action": "RETAIN", "id": existing_id},
        ]
    }
    client = MagicMock()
    client.post = AsyncMock(return_value=_litellm_response(json.dumps(payload)))
    monkeypatch.setattr("qortia.reflect.get_litellm_client", lambda: client)

    reflections = await _call_litellm_reflect("model", ["recent"], [], "key", AGENT_ID, TENANT_ID)

    assert [r["action"] for r in reflections] == ["CREATE", "UPDATE", "RETAIN"]
    assert client.post.await_args.kwargs["json"]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_call_litellm_reflect_raises_on_non_200_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.reflect import _call_litellm_reflect

    client = MagicMock()
    client.post = AsyncMock(return_value=_litellm_response("{}", status_code=503))
    monkeypatch.setattr("qortia.reflect.get_litellm_client", lambda: client)

    with pytest.raises(HTTPException) as exc:
        await _call_litellm_reflect("model", ["recent"], [], "key", AGENT_ID, TENANT_ID)

    assert exc.value.status_code == 500


def test_compute_stability_scores_only_scores_updates_with_both_embeddings() -> None:
    from qortia.reflect import _compute_stability_scores

    existing_id = str(uuid4())

    scores = _compute_stability_scores(
        [
            {"action": "CREATE", "embedding": [1.0, 0.0]},
            {"action": "UPDATE", "id": existing_id, "embedding": [1.0, 0.0]},
            {"action": "UPDATE", "id": "missing", "embedding": [1.0, 0.0]},
        ],
        {existing_id: [1.0, 0.0]},
    )

    assert scores == [None, pytest.approx(1.0), None]


@pytest.mark.asyncio
async def test_write_reflections_refuses_to_prune_every_existing_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.reflect import _write_reflections

    conn = MagicMock()
    # Enough fetchval returns for prune-abort path + later inserts.
    conn.fetchval = AsyncMock(side_effect=[2, 0, uuid4(), 0, uuid4(), 0])
    conn.execute = AsyncMock()
    monkeypatch.setattr("qortia.reflect.tenant_transaction", lambda *_a, **_kw: _tenant_tx(conn))
    monkeypatch.setattr("qortia.reflect.get_main_pool", lambda: object())
    monkeypatch.setattr("qortia.knowledge.extract_entities_with_types", lambda *_a, **_kw: [])

    written, counter = await _write_reflections(
        agent_id=AGENT_ID,
        tenant_id=TENANT_ID,
        reflections=[
            {"action": "CREATE", "type": "lesson", "content": "New lesson", "importance": 0.7}
        ],
        new_embeddings={0: [0.1] * 8},
        existing_embeddings={},
        existing=[
            {"id": str(uuid4()), "type": "lesson", "content": "one"},
            {"id": str(uuid4()), "type": "mental_model", "content": "two"},
        ],
        clearance_order=2,
        agent_division="all",
    )

    assert written >= 0
    assert not any(
        "id != ALL" in call.args[0]
        for call in conn.execute.await_args_list
        if call.args and isinstance(call.args[0], str)
    )


@pytest.mark.asyncio
async def test_write_reflections_skips_duplicate_reflection_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.reflect import _write_reflections

    conn = MagicMock()
    conn.fetchval = AsyncMock(side_effect=[uuid4(), 0])
    conn.execute = AsyncMock()
    monkeypatch.setattr("qortia.reflect.tenant_transaction", lambda *_a, **_kw: _tenant_tx(conn))
    monkeypatch.setattr("qortia.reflect.get_main_pool", lambda: object())
    monkeypatch.setattr("qortia.knowledge.extract_entities_with_types", lambda *_a, **_kw: [])

    written, _counter = await _write_reflections(
        agent_id=AGENT_ID,
        tenant_id=TENANT_ID,
        reflections=[
            {
                "action": "CREATE",
                "type": "lesson",
                "content": "Same durable lesson",
                "importance": 0.7,
            },
            {
                "action": "CREATE",
                "type": "lesson",
                "content": " same durable lesson ",
                "importance": 0.8,
            },
        ],
        new_embeddings={0: [0.1] * 1024, 1: [0.2] * 1024},
        existing_embeddings={},
        existing=[],
        clearance_order=2,
        agent_division="all",
    )

    assert written == 1


@pytest.mark.asyncio
async def test_process_embedding_batch_groups_rows_by_tenant_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.reflect import _process_embedding_batch

    rows = [
        {
            "id": uuid4(),
            "tenant_id": TENANT_ID,
            "agent_id": AGENT_ID,
            "text_to_embed": "private memory",
            "lang": "en",
            "tbl": "hindsight_memories",
            "type": "episodic",
        },
        {
            "id": uuid4(),
            "tenant_id": TENANT_ID,
            "text_to_embed": "org memory",
            "lang": "en",
            "tbl": "org_memory",
        },
    ]
    conn = MagicMock()
    conn.transaction.return_value = _AsyncContext()
    conn.fetch = AsyncMock(side_effect=[rows[:1], rows[1:], [], []])
    monkeypatch.setattr("qortia.reflect.get_main_pool", lambda: _pool(conn))
    get_key = AsyncMock(return_value="tenant-key")
    embed_row = AsyncMock()
    monkeypatch.setattr("qortia.reflect.get_litellm_key", get_key)
    monkeypatch.setattr("qortia.reflect._embed_single_row", embed_row)

    await _process_embedding_batch()

    get_key.assert_awaited_once_with(str(TENANT_ID))
    assert embed_row.await_count == 2


@pytest.mark.asyncio
async def test_embed_single_row_updates_embeddings_and_links_private_memories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.reflect import _embed_single_row

    conn = MagicMock()
    conn.execute = AsyncMock()
    monkeypatch.setattr("qortia.reflect.get_main_pool", lambda: _pool(conn))
    monkeypatch.setattr("qortia.reflect._get_embedding", AsyncMock(return_value=[0.1] * 1024))
    find_links = AsyncMock(return_value=[{"id": uuid4(), "similarity": 0.9}])
    upsert_links = AsyncMock()
    dedup = AsyncMock()
    monkeypatch.setattr("qortia.links._find_similar_memories", find_links)
    monkeypatch.setattr("qortia.links._upsert_memory_links", upsert_links)
    monkeypatch.setattr("qortia.reflect._maybe_dedup_memory", dedup)

    memory_id = uuid4()
    await _embed_single_row(
        {
            "id": memory_id,
            "tenant_id": TENANT_ID,
            "agent_id": AGENT_ID,
            "text_to_embed": "memory text",
            "tbl": "hindsight_memories",
            "type": "episodic",
        },
        "key",
    )

    conn.execute.assert_awaited_once()
    find_links.assert_awaited_once()
    upsert_links.assert_awaited_once()
    dedup.assert_awaited_once()


@pytest.mark.asyncio
async def test_embed_single_row_skips_short_term_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.reflect import _embed_single_row

    get_embedding = AsyncMock()
    monkeypatch.setattr("qortia.reflect._get_embedding", get_embedding)

    await _embed_single_row(
        {
            "id": uuid4(),
            "tenant_id": TENANT_ID,
            "agent_id": AGENT_ID,
            "text_to_embed": "temporary note",
            "tbl": "hindsight_memories",
            "type": "short_term",
        },
        "key",
    )

    get_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_embed_single_row_records_failed_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.reflect import _embed_single_row

    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.fetchval = AsyncMock(return_value=3)
    monkeypatch.setattr("qortia.reflect.get_main_pool", lambda: _pool(conn))
    monkeypatch.setattr(
        "qortia.reflect._get_embedding", AsyncMock(side_effect=RuntimeError("down"))
    )

    await _embed_single_row(
        {
            "id": uuid4(),
            "tenant_id": TENANT_ID,
            "text_to_embed": "knowledge summary",
            "tbl": "org_knowledge",
        },
        "key",
    )

    assert conn.execute.await_count == 1
    assert conn.fetchval.await_count == 1
