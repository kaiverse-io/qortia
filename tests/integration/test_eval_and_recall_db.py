"""Integration tests for eval router endpoints and recall DB paths."""

from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tests.integration.conftest import (
    _call,
    create_active_agent,
    make_agent_headers,
)
from tests.integration.helpers import (
    VECTOR_LITERAL,
    memory_payload,
    patch_entity_extraction,
    patch_knowledge_index,
)


async def _delete_eval_tenant(tenant_id: object, superuser_url: str) -> None:
    su = await asyncpg.connect(superuser_url)
    try:
        await su.execute("DELETE FROM memory_links WHERE tenant_id = $1", tenant_id)
        await su.execute("DELETE FROM qortia_entities WHERE tenant_id = $1", tenant_id)
        await su.execute("DELETE FROM qortia_session_reads WHERE tenant_id = $1", tenant_id)
        await su.execute("DELETE FROM qortia_outcome_records WHERE tenant_id = $1", tenant_id)
        await su.execute("DELETE FROM org_knowledge WHERE tenant_id = $1", tenant_id)
        await su.execute("DELETE FROM org_memory WHERE tenant_id = $1", tenant_id)
        await su.execute("DELETE FROM hindsight_memories WHERE tenant_id = $1", tenant_id)
        await su.execute("DELETE FROM qortia_api_keys WHERE tenant_id = $1", tenant_id)
        await su.execute("DELETE FROM qortia_agents WHERE tenant_id = $1", tenant_id)
        await su.execute("DELETE FROM qortia_tenants WHERE id = $1", tenant_id)
    finally:
        await su.close()


def test_eval_router_endpoints_with_eval_mode_real_db(
    _session_loop,
    pg_superuser_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_entity_extraction(monkeypatch)
    patch_knowledge_index(monkeypatch)
    tenant_id = uuid4()
    agent_id = uuid4()

    async def scenario() -> dict[str, object]:
        from qortia import config
        from qortia.eval_router import router as eval_router

        old_eval_mode = config.settings.eval_mode
        config.settings.eval_mode = True
        app = FastAPI()
        app.include_router(eval_router)
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://eval",
            ) as client:
                seeded = await client.post(
                    "/v1/internal/eval/seed-agent",
                    params={
                        "tenant_id": str(tenant_id),
                        "agent_id": str(agent_id),
                        "name": "eval-chief",
                        "role": "chief",
                    },
                )
                assert seeded.status_code == 200

                private_memory = await client.post(
                    "/v1/internal/eval/seed-memory",
                    json={
                        "tenant_id": str(tenant_id),
                        "agent_id": str(agent_id),
                        "content": "Qortia eval memory recalls PostgreSQL durable retrieval",
                        "mem_type": "decision",
                        "importance": 0.9,
                        "is_consolidated": True,
                        "valid_from": "2026-01-01T00:00:00+00:00",
                        "valid_until": "2027-01-01T00:00:00+00:00",
                    },
                )
                assert private_memory.status_code == 200
                memory_id = private_memory.json()["memory_id"]

                org_memory = await client.post(
                    "/v1/internal/eval/remember-org",
                    params={"tenant_id": str(tenant_id), "agent_id": str(agent_id)},
                    json={
                        "type": "process",
                        "title": "Eval process",
                        "content": (
                            "Qortia eval process recalls PostgreSQL durable retrieval "
                            "for every teammate today"
                        ),
                        "valid_from": "2026-01-01T00:00:00+00:00",
                        "valid_until": "2027-01-01T00:00:00+00:00",
                    },
                )
                assert org_memory.status_code == 200

                knowledge = await client.post(
                    "/v1/internal/eval/knowledge",
                    params={"tenant_id": str(tenant_id), "agent_id": str(agent_id)},
                    json={
                        "source_type": "note",
                        "source_path": "eval/qortia.md",
                        "content": " ".join(["Qortia PostgreSQL retrieval knowledge"] * 20),
                    },
                )
                assert knowledge.status_code == 200

                recall = await client.post(
                    "/v1/internal/eval/recall",
                    params={
                        "tenant_id": str(tenant_id),
                        "agent_id": str(agent_id),
                        "query": "Qortia PostgreSQL retrieval",
                        "scope": "all",
                    },
                )
                assert recall.status_code == 200

                recall_full = await client.post(
                    "/v1/internal/eval/recall-full",
                    params={"tenant_id": str(tenant_id), "agent_id": str(agent_id)},
                    json={
                        "query": "Qortia PostgreSQL retrieval",
                        "scope": "private",
                        "type": "decision",
                        "rerank": False,
                    },
                )
                assert recall_full.status_code == 200

                reflect = await client.post(
                    "/v1/internal/eval/reflect",
                    params={"tenant_id": str(tenant_id), "agent_id": str(agent_id)},
                )
                assert reflect.status_code == 200

                return {
                    "memory_id": memory_id,
                    "recall_count": len(recall.json()["results"]),
                    "full_count": len(recall_full.json()["results"]),
                }
        finally:
            config.settings.eval_mode = old_eval_mode
            await _delete_eval_tenant(tenant_id, pg_superuser_url)

    result = _call(_session_loop, scenario())
    assert result["memory_id"]
    assert result["recall_count"] >= 1
    assert result["full_count"] >= 1


def test_recall_type_org_knowledge_archive_and_session_paths(
    app_client,
    _session_loop,
    committed_conn,
    tenant_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_entity_extraction(monkeypatch)
    patch_knowledge_index(monkeypatch)
    agent_id = create_active_agent(committed_conn, tenant_id, role="chief")
    headers = make_agent_headers(agent_id, tenant_id)
    work_order_id = str(uuid4())

    remember_response = _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    memory_payload(
                        "decision",
                        "Qortia archive strategy keeps PostgreSQL decisions searchable",
                    ),
                    memory_payload(
                        "lesson",
                        "Qortia vector lesson says PostgreSQL retrieval needs embeddings",
                    ),
                    memory_payload(
                        "episodic",
                        "Qortia episodic recall captured PostgreSQL archive work today",
                    ),
                    memory_payload(
                        "short_term",
                        "Qortia short term recall should expire after the test window",
                        ttl_seconds=3600,
                    ),
                ]
            },
            headers=headers,
        ),
    )
    assert remember_response.status_code == 200
    decision_id, lesson_id, episodic_id, short_term_id = remember_response.json()["ids"]
    for memory_id in (decision_id, lesson_id, episodic_id, short_term_id):
        committed_conn.track("hindsight_memories", memory_id)
    committed_conn.execute(
        "UPDATE hindsight_memories SET embedding = $1::vector WHERE id = ANY($2::uuid[])",
        VECTOR_LITERAL,
        [decision_id, lesson_id, episodic_id, short_term_id],
    )
    committed_conn.execute(
        "UPDATE hindsight_memories SET tier = 'archive' WHERE id = $1", decision_id
    )

    org_response = _call(
        _session_loop,
        app_client.post(
            "/v1/remember-org",
            json={
                "type": "process",
                "title": "Qortia retrieval process",
                "content": (
                    "Qortia process memory says PostgreSQL retrieval knowledge is shared "
                    "with all teams every week"
                ),
            },
            headers=headers,
        ),
    )
    assert org_response.status_code == 200
    committed_conn.track("org_memory", org_response.json()["id"])

    knowledge_response = _call(
        _session_loop,
        app_client.post(
            "/v1/knowledge",
            json={
                "source_type": "note",
                "source_path": f"docs/qortia-{tenant_id}.md",
                "content": " ".join(["Qortia PostgreSQL knowledge retrieval handbook"] * 20),
            },
            headers=headers,
        ),
    )
    assert knowledge_response.status_code == 200
    knowledge_ids = committed_conn.fetch(
        "SELECT id FROM org_knowledge WHERE tenant_id = $1",
        tenant_id,
    )
    for row in knowledge_ids:
        committed_conn.track("org_knowledge", str(row["id"]))
    committed_conn.execute(
        "UPDATE org_knowledge SET embedding = $1::vector WHERE tenant_id = $2",
        VECTOR_LITERAL,
        tenant_id,
    )

    def recall(body: dict[str, object], extra_headers: dict[str, str] | None = None) -> list[dict]:
        merged_headers = dict(headers)
        if extra_headers:
            merged_headers.update(extra_headers)
        response = _call(
            _session_loop,
            app_client.post("/v1/recall", json=body, headers=merged_headers),
        )
        assert response.status_code == 200, response.text
        return response.json()["results"]

    archive_results = recall({"query": "archive strategy PostgreSQL", "scope": "archive"})
    assert decision_id in {result["id"] for result in archive_results}

    decision_results = recall(
        {"query": "archive strategy PostgreSQL", "scope": "archive", "type": "decision"}
    )
    assert decision_results[0]["id"] == decision_id

    lesson_results = recall(
        {"query": "vector lesson embeddings", "scope": "private", "type": "lesson"}
    )
    assert lesson_id in {result["id"] for result in lesson_results}

    episodic_results = recall(
        {"query": "episodic recall PostgreSQL", "scope": "private", "type": "episodic"}
    )
    assert episodic_id in {result["id"] for result in episodic_results}

    short_term_results = recall(
        {"query": "short term expire", "scope": "private", "type": "short_term"},
        {"X-Work-Order-Id": work_order_id},
    )
    assert short_term_id in {result["id"] for result in short_term_results}

    org_results = recall({"query": "process memory PostgreSQL", "scope": "org"})
    assert any(result["scope"] == "org" for result in org_results)

    knowledge_results = recall({"query": "knowledge retrieval handbook", "scope": "knowledge"})
    assert any(result["scope"] == "knowledge" for result in knowledge_results)

    all_results = recall({"query": "Qortia PostgreSQL retrieval", "scope": "all"})
    assert any(result["scope"] in {"org", "knowledge", "private"} for result in all_results)

    # Knowledge delete + dedup re-ingest
    delete_response = _call(
        _session_loop,
        app_client.delete(
            f"/v1/knowledge/docs/qortia-{tenant_id}.md",
            headers=headers,
        ),
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["chunks_deleted"] >= 1

    dedup_response = _call(
        _session_loop,
        app_client.post(
            "/v1/knowledge",
            json={
                "source_type": "note",
                "source_path": f"docs/qortia-{tenant_id}.md",
                "content": " ".join(["Qortia PostgreSQL knowledge retrieval handbook"] * 20),
            },
            headers=headers,
        ),
    )
    assert dedup_response.status_code == 200
    for row in committed_conn.fetch(
        "SELECT id FROM org_knowledge WHERE tenant_id = $1",
        tenant_id,
    ):
        committed_conn.track("org_knowledge", str(row["id"]))

    dedup_again = _call(
        _session_loop,
        app_client.post(
            "/v1/knowledge",
            json={
                "source_type": "note",
                "source_path": f"docs/qortia-{tenant_id}.md",
                "content": " ".join(["Qortia PostgreSQL knowledge retrieval handbook"] * 20),
            },
            headers=headers,
        ),
    )
    assert dedup_again.status_code == 200
    assert dedup_again.json()["sections_created"] == 0
