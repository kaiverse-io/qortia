"""Integration tests for memory links and entity graph population."""

from __future__ import annotations

from uuid import UUID

import pytest

from tests.integration.conftest import (
    MOCK_EMBEDDING,
    _call,
    create_active_agent,
    make_agent_headers,
)
from tests.integration.helpers import VECTOR_LITERAL, memory_payload, patch_entity_extraction


def test_memory_links_find_upsert_and_expand_real_db(
    app_client, _session_loop, committed_conn, tenant_id: str
) -> None:
    agent_id = create_active_agent(committed_conn, tenant_id)
    headers = make_agent_headers(agent_id, tenant_id)

    response = _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    memory_payload(
                        "decision",
                        "Qortia stores durable retrieval decisions in PostgreSQL today",
                    ),
                    memory_payload(
                        "lesson",
                        "PostgreSQL retrieval decisions should remain close to Qortia links",
                    ),
                ]
            },
            headers=headers,
        ),
    )
    assert response.status_code == 200
    first_id, second_id = response.json()["ids"]
    committed_conn.track("hindsight_memories", first_id)
    committed_conn.track("hindsight_memories", second_id)
    committed_conn.execute(
        "UPDATE hindsight_memories SET embedding = $1::vector WHERE id = ANY($2::uuid[])",
        VECTOR_LITERAL,
        [first_id, second_id],
    )

    async def scenario() -> list[str | None]:
        from qortia.links import _expand_with_links, _find_similar_memories, _upsert_memory_links
        from qortia.models import RecallResult

        similar = await _find_similar_memories(
            UUID(first_id),
            MOCK_EMBEDDING,
            UUID(tenant_id),
            UUID(agent_id),
            threshold=0.99,
            top_n=2,
        )
        assert similar == [{"id": UUID(second_id), "similarity": pytest.approx(1.0)}]

        await _upsert_memory_links(UUID(first_id), similar, UUID(tenant_id))
        await _upsert_memory_links(UUID(first_id), [], UUID(tenant_id))

        original = RecallResult(
            id=first_id,
            type="decision",
            scope="private",
            content="Qortia stores durable retrieval decisions in PostgreSQL today",
            importance=0.9,
            created_at="2026-08-02T00:00:00+00:00",
        )
        expanded = await _expand_with_links([original], UUID(tenant_id), UUID(agent_id))
        assert [r.id for r in expanded] == [first_id, second_id]
        return [r.linked_via for r in expanded]

    linked_via = _call(_session_loop, scenario())
    assert linked_via == [None, first_id]


def test_entity_graph_population_from_remembered_entities(
    app_client, _session_loop, committed_conn, tenant_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_entity_extraction(monkeypatch)
    agent_id = create_active_agent(committed_conn, tenant_id)
    headers = make_agent_headers(agent_id, tenant_id)

    response = _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    memory_payload(
                        "decision",
                        "Qortia chose PostgreSQL as the durable graph backing store",
                    )
                ]
            },
            headers=headers,
        ),
    )
    assert response.status_code == 200
    memory_id = response.json()["ids"][0]
    committed_conn.track("hindsight_memories", memory_id)

    org_response = _call(
        _session_loop,
        app_client.post(
            "/v1/remember-org",
            json={
                "type": "handoff",
                "title": "Qortia graph handoff",
                "content": (
                    "Qortia and PostgreSQL graph population handoff covers entity links "
                    "for the whole organisation today"
                ),
            },
            headers=headers,
        ),
    )
    assert org_response.status_code == 200
    org_id = org_response.json()["id"]
    committed_conn.track("org_memory", org_id)

    async def graph_scenario() -> tuple[bool, list[str], list[str], str | None]:
        from qortia.db import get_main_pool
        from qortia.entity_graph import _populate_graph_batch

        await _populate_graph_batch()
        async with get_main_pool().acquire() as conn:
            private_row = await conn.fetchrow(
                """
                SELECT linked_memory_ids, summary
                FROM qortia_entities
                WHERE tenant_id = $1 AND agent_id = $2 AND entity_text = 'Qortia'
                """,
                UUID(tenant_id),
                UUID(agent_id),
            )
            org_row = await conn.fetchrow(
                """
                SELECT linked_memory_ids
                FROM qortia_entities
                WHERE tenant_id = $1 AND agent_id IS NULL AND entity_text = 'Qortia'
                """,
                UUID(tenant_id),
            )
            graphed = await conn.fetchval(
                "SELECT is_graphed FROM hindsight_memories WHERE id = $1",
                UUID(memory_id),
            )
        assert private_row is not None
        assert org_row is not None
        return (
            bool(graphed),
            [str(v) for v in private_row["linked_memory_ids"]],
            [str(v) for v in org_row["linked_memory_ids"]],
            private_row["summary"],
        )

    graphed, private_links, org_links, summary = _call(_session_loop, graph_scenario())
    assert graphed is True
    assert memory_id in private_links
    assert org_id in org_links
    assert summary is not None
