"""Qortia API integration tests."""

from __future__ import annotations

import json
from datetime import UTC
from uuid import uuid4

from tests.integration.conftest import (
    _call,
    create_active_agent,
    fresh_agent_headers,
    make_agent_headers,
)


def _active_agent(loop, conn, tenant_id: str) -> str:
    return create_active_agent(conn, tenant_id)


# ── remember ──────────────────────────────────────────────────────────────────


def test_remember_requires_auth(app_client, _session_loop) -> None:
    r = _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "episodic",
                        "content": "something happened today in the system",
                    }
                ]
            },
        ),
    )
    assert r.status_code == 401


def test_remember_inactive_agent_403(app_client, _session_loop) -> None:
    r = _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "episodic",
                        "content": "something happened today in the system",
                    }
                ]
            },
            headers=fresh_agent_headers(),
        ),
    )
    assert r.status_code == 403


def test_remember_empty_batch_422(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    r = _call(
        _session_loop,
        app_client.post(
            "/v1/remember", json={"memories": []}, headers=make_agent_headers(aid, tenant_id)
        ),
    )
    assert r.status_code == 422


def test_remember_invalid_type_422(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    r = _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "org_chart",
                        "content": "something happened today in the system",
                    }
                ]
            },
            headers=make_agent_headers(aid, tenant_id),
        ),
    )
    assert r.status_code == 422


def test_remember_batch_atomicity(app_client, _session_loop, committed_conn, tenant_id) -> None:
    """All inserts + counter increment in one transaction."""
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = make_agent_headers(aid, tenant_id)

    r = _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "episodic",
                        "content": "event one happened today in the system",
                    },
                    {
                        "type": "episodic",
                        "content": "event two happened today in the system",
                    },
                    {
                        "type": "lesson",
                        "content": "learned something important about the system today",
                    },
                ]
            },
            headers=headers,
        ),
    )
    assert r.status_code == 200
    assert len(r.json()["ids"]) == 3

    count = committed_conn.fetchval(
        "SELECT COUNT(*) FROM hindsight_memories WHERE agent_id = $1", aid
    )
    assert count == 3

    counter = committed_conn.fetchval(
        "SELECT reflection_counter FROM qortia_agents WHERE id = $1", aid
    )
    assert counter == 2  # only episodic memories increment counter


def test_remember_entities_extracted(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    r = _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "decision",
                        "content": "chose PostgreSQL for the database storage layer",
                    }
                ]
            },
            headers=make_agent_headers(aid, tenant_id),
        ),
    )
    assert r.status_code == 200

    row = committed_conn.fetchval(
        "SELECT entities FROM hindsight_memories WHERE agent_id = $1", aid
    )
    assert row is not None
    assert isinstance(json.loads(row), list)


# ── remember-org ──────────────────────────────────────────────────────────────


def test_remember_org_handoff(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    r = _call(
        _session_loop,
        app_client.post(
            "/v1/remember-org",
            json={
                "type": "handoff",
                "title": "Completed auth",
                "content": "Done with the task successfully today for the entire team",
            },
            headers=make_agent_headers(aid, tenant_id),
        ),
    )
    assert r.status_code == 200
    assert "id" in r.json()


def test_remember_org_process_requires_chief(
    app_client, _session_loop, committed_conn, tenant_id
) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)  # engineer
    r = _call(
        _session_loop,
        app_client.post(
            "/v1/remember-org",
            json={
                "type": "process",
                "title": "Deploy",
                "content": "How we deploy the application to production servers every time",
            },
            headers=make_agent_headers(aid, tenant_id),
        ),
    )
    assert r.status_code == 403


def test_remember_org_invalid_type_422(
    app_client, _session_loop, committed_conn, tenant_id
) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    r = _call(
        _session_loop,
        app_client.post(
            "/v1/remember-org",
            json={
                "type": "org_chart",
                "title": "t",
                "content": "content for the org memory handoff here now",
            },
            headers=make_agent_headers(aid, tenant_id),
        ),
    )
    assert r.status_code == 422


# ── recall ────────────────────────────────────────────────────────────────────


def test_recall_requires_auth(app_client, _session_loop) -> None:
    r = _call(_session_loop, app_client.post("/v1/recall", json={"query": "test"}))
    assert r.status_code == 401


def test_recall_empty_query_422(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    r = _call(
        _session_loop,
        app_client.post(
            "/v1/recall", json={"query": ""}, headers=make_agent_headers(aid, tenant_id)
        ),
    )
    assert r.status_code == 422


def test_recall_invalid_scope_422(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    r = _call(
        _session_loop,
        app_client.post(
            "/v1/recall",
            json={"query": "test", "scope": "invalid"},
            headers=make_agent_headers(aid, tenant_id),
        ),
    )
    assert r.status_code == 422


def test_recall_returns_results(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = make_agent_headers(aid, tenant_id)

    _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "decision",
                        "content": "chose PostgreSQL for the database storage layer",
                    }
                ]
            },
            headers=headers,
        ),
    )

    r = _call(
        _session_loop,
        app_client.post(
            "/v1/recall",
            json={
                "query": "database",
                "scope": "private",
            },
            headers=headers,
        ),
    )
    assert r.status_code == 200
    assert "results" in r.json()


def test_recall_entities_filter(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = make_agent_headers(aid, tenant_id)

    _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "decision",
                        "content": "chose PostgreSQL for the database storage layer",
                    },
                ]
            },
            headers=headers,
        ),
    )

    r = _call(
        _session_loop,
        app_client.post(
            "/v1/recall",
            json={
                "query": "database",
                "scope": "private",
                "entities": ["PostgreSQL"],
            },
            headers=headers,
        ),
    )
    assert r.status_code == 200


# ── context ───────────────────────────────────────────────────────────────────


def test_context_requires_agent_auth(app_client, _session_loop) -> None:
    r = _call(_session_loop, app_client.get("/v1/context"))
    assert r.status_code == 401


def test_context_structure(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    r = _call(
        _session_loop,
        app_client.get("/v1/context", headers=make_agent_headers(aid, tenant_id)),
    )
    assert r.status_code == 200
    data = r.json()
    for key in ("org_chart", "processes", "handoffs", "memories"):
        assert key in data
    for key in ("mental_models", "decisions", "lessons"):
        assert key in data["memories"]


# ── forget ────────────────────────────────────────────────────────────────────


def test_forget_own_memory(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = make_agent_headers(aid, tenant_id)

    r = _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "episodic",
                        "content": "to be forgotten after this test runs today",
                    }
                ]
            },
            headers=headers,
        ),
    )
    mem_id = r.json()["ids"][0]

    r = _call(
        _session_loop,
        app_client.post("/v1/forget", json={"id": mem_id}, headers=headers),
    )
    assert r.status_code == 200

    count = committed_conn.fetchval("SELECT COUNT(*) FROM hindsight_memories WHERE id = $1", mem_id)
    assert count == 0


# ── reflect ───────────────────────────────────────────────────────────────────


def test_reflect_supersede_first(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = make_agent_headers(aid, tenant_id)

    # Existing consolidated memory
    old_id = str(uuid4())
    committed_conn.execute(
        """
        INSERT INTO hindsight_memories
            (id, tenant_id, agent_id, type, content, importance, is_consolidated)
        VALUES ($1, $2, $3, $4, $5, $6, true)
    """,
        old_id,
        tenant_id,
        aid,
        "mental_model",
        "old model",
        0.8,
    )
    committed_conn.track("hindsight_memories", old_id)

    # Write episodic memories
    _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "episodic",
                        "content": f"event {i} happened today in the system",
                    }
                    for i in range(3)
                ]
            },
            headers=headers,
        ),
    )

    r = _call(_session_loop, app_client.post("/v1/reflect", json={}, headers=headers))
    assert r.status_code == 200
    assert "memories_written" in r.json()

    # Old memory must be superseded
    is_consolidated = committed_conn.fetchval(
        "SELECT is_consolidated FROM hindsight_memories WHERE id = $1", old_id
    )
    assert is_consolidated is False


def test_idle_reflection_trigger_gates_on_reflection_counter(
    app_client, _session_loop, committed_conn, tenant_id
) -> None:
    """Background idle-reflect trigger must skip agents that haven't accumulated
    reflection_threshold new episodics, even if they're idle — and must reflect
    (and decrement the counter for) agents that have, once idle. Regression test
    for the gate that was dropped between design (qortia-background-reflection-
    trigger.md) and the shipped _trigger_idle_reflections query."""
    from qortia.reflect import _trigger_idle_reflections

    below_id = _active_agent(_session_loop, committed_conn, tenant_id)
    at_id = _active_agent(_session_loop, committed_conn, tenant_id)

    def _remember_episodics(aid: str, n: int) -> None:
        _call(
            _session_loop,
            app_client.post(
                "/v1/remember",
                json={
                    "memories": [
                        {"type": "episodic", "content": f"idle trigger event {i} today"}
                        for i in range(n)
                    ]
                },
                headers=make_agent_headers(aid, tenant_id),
            ),
        )

    _remember_episodics(below_id, 5)  # below reflection_threshold (default 10)
    _remember_episodics(at_id, 10)  # at reflection_threshold (default 10)

    # Both idle well past idle_reflection_window_h (default 1h)
    for aid in (below_id, at_id):
        committed_conn.execute(
            "UPDATE qortia_agents SET updated_at = now() - interval '2 hours' WHERE id = $1",
            aid,
        )

    _call(_session_loop, _trigger_idle_reflections())

    below_counter = committed_conn.fetchval(
        "SELECT reflection_counter FROM qortia_agents WHERE id = $1", below_id
    )
    at_counter = committed_conn.fetchval(
        "SELECT reflection_counter FROM qortia_agents WHERE id = $1", at_id
    )
    assert below_counter == 5, "below-threshold idle agent must not be reflected"
    assert at_counter == 0, "at-threshold idle agent must be reflected, counter decremented"

    at_consolidated = committed_conn.fetchval(
        """
        SELECT count(*) FROM hindsight_memories
        WHERE agent_id = $1 AND is_consolidated = true
        """,
        at_id,
    )
    assert at_consolidated > 0


# ── memory_links (16i) ────────────────────────────────────────────────────────


def test_forget_cleans_up_memory_links(
    app_client, _session_loop, committed_conn, tenant_id
) -> None:
    """Forgetting a memory removes its memory_links rows."""
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = make_agent_headers(aid, tenant_id)

    r = _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "episodic",
                        "content": "to be forgotten after this test runs today",
                    }
                ]
            },
            headers=headers,
        ),
    )
    mem_id = r.json()["ids"][0]

    # Manually insert a link row so we can verify cleanup without running the worker
    committed_conn.execute(
        """
        INSERT INTO memory_links (tenant_id, source_id, target_id, similarity)
        VALUES ($1, $2, $3, 0.80)
        """,
        tenant_id,
        mem_id,
        mem_id,  # self-link is nonsensical but sufficient to test cleanup
    )

    r = _call(
        _session_loop,
        app_client.post("/v1/forget", json={"id": mem_id}, headers=headers),
    )
    assert r.status_code == 200

    count = committed_conn.fetchval(
        "SELECT COUNT(*) FROM memory_links WHERE source_id = $1 OR target_id = $1",
        mem_id,
    )
    assert count == 0


def test_recall_linked_via_field_present(
    app_client, _session_loop, committed_conn, tenant_id
) -> None:
    """RecallResult schema includes linked_via field (null when not a linked result)."""
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = make_agent_headers(aid, tenant_id)

    _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "decision",
                        "content": "chose PostgreSQL for the database storage layer",
                    }
                ]
            },
            headers=headers,
        ),
    )

    r = _call(
        _session_loop,
        app_client.post(
            "/v1/recall",
            json={"query": "database", "scope": "private"},
            headers=headers,
        ),
    )
    assert r.status_code == 200
    results = r.json()["results"]
    # Every result must have the linked_via key (null for non-linked results)
    for result in results:
        assert "linked_via" in result


def test_memory_links_rls_tenant_isolation(
    app_client, _session_loop, committed_conn, tenant_id
) -> None:
    """Agent from tenant A cannot see memory_links belonging to tenant B."""
    # Tenant A
    aid_a = _active_agent(_session_loop, committed_conn, tenant_id)

    # Tenant B — separate tenant
    tid_b = str(uuid4())
    committed_conn.execute(
        "INSERT INTO qortia_tenants (id, name) VALUES ($1, $2)",
        tid_b,
        f"tenant-b-{tid_b[:8]}",
    )
    committed_conn.track("qortia_tenants", tid_b)
    aid_b = _active_agent(_session_loop, committed_conn, tid_b)

    # Write a memory for tenant B and manually insert a link
    r = _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "episodic",
                        "content": "tenant B secret data should not be visible",
                    }
                ]
            },
            headers=make_agent_headers(aid_b, tid_b),
        ),
    )
    mem_b_id = r.json()["ids"][0]
    committed_conn.execute(
        """
        INSERT INTO memory_links (tenant_id, source_id, target_id, similarity)
        VALUES ($1, $2, $3, 0.90)
        """,
        tid_b,
        mem_b_id,
        mem_b_id,
    )

    # Tenant A recall must not surface tenant B's link
    r = _call(
        _session_loop,
        app_client.post(
            "/v1/recall",
            json={"query": "tenant B secret", "scope": "private"},
            headers=make_agent_headers(aid_a, tenant_id),
        ),
    )
    assert r.status_code == 200
    result_ids = {res["id"] for res in r.json()["results"]}
    assert mem_b_id not in result_ids


# ── temporal fact bounds (16j) ────────────────────────────────────────────────


def test_valid_until_set_on_supersede(app_client, _session_loop, committed_conn, tenant_id) -> None:
    """After reflect(), superseded consolidated memories have valid_until set."""
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = make_agent_headers(aid, tenant_id)

    old_id = str(uuid4())
    committed_conn.execute(
        """
        INSERT INTO hindsight_memories
            (id, tenant_id, agent_id, type, content, importance, is_consolidated)
        VALUES ($1, $2, $3, $4, $5, $6, true)
        """,
        old_id,
        tenant_id,
        aid,
        "mental_model",
        "old model to supersede",
        0.8,
    )
    committed_conn.track("hindsight_memories", old_id)

    _call(
        _session_loop,
        app_client.post(
            "/v1/remember",
            json={
                "memories": [
                    {
                        "type": "episodic",
                        "content": f"event {i} happened today in the system",
                    }
                    for i in range(3)
                ]
            },
            headers=headers,
        ),
    )

    r = _call(_session_loop, app_client.post("/v1/reflect", json={}, headers=headers))
    assert r.status_code == 200

    valid_until = committed_conn.fetchval(
        "SELECT valid_until FROM hindsight_memories WHERE id = $1", old_id
    )
    assert valid_until is not None, "Superseded memory must have valid_until set"


def test_superseded_memory_excluded_from_default_recall(
    app_client, _session_loop, committed_conn, tenant_id
) -> None:
    """A memory with valid_until set must not appear in default recall."""
    from datetime import datetime, timedelta

    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = make_agent_headers(aid, tenant_id)

    mem_id = str(uuid4())
    committed_conn.execute(
        """
        INSERT INTO hindsight_memories
            (id, tenant_id, agent_id, type, content, importance,
             valid_from, valid_until)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        mem_id,
        tenant_id,
        aid,
        "mental_model",
        "superseded unique content xq9z",
        0.8,
        datetime.now(UTC) - timedelta(days=10),
        datetime.now(UTC) - timedelta(days=1),  # superseded yesterday
    )
    committed_conn.track("hindsight_memories", mem_id)

    r = _call(
        _session_loop,
        app_client.post(
            "/v1/recall",
            json={"query": "superseded unique content xq9z", "scope": "private"},
            headers=headers,
        ),
    )
    assert r.status_code == 200
    result_ids = {res["id"] for res in r.json()["results"]}
    assert mem_id not in result_ids


def test_as_of_returns_superseded_memory(
    app_client, _session_loop, committed_conn, tenant_id
) -> None:
    """as_of set to when the memory was valid must return it."""
    from datetime import datetime, timedelta

    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = make_agent_headers(aid, tenant_id)

    valid_start = datetime.now(UTC) - timedelta(days=30)
    valid_end = datetime.now(UTC) - timedelta(days=1)
    as_of = datetime.now(UTC) - timedelta(days=15)  # within the valid window

    mem_id = str(uuid4())
    committed_conn.execute(
        """
        INSERT INTO hindsight_memories
            (id, tenant_id, agent_id, type, content, importance,
             valid_from, valid_until)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        mem_id,
        tenant_id,
        aid,
        "mental_model",
        "historical fact aof7k",
        0.8,
        valid_start,
        valid_end,
    )
    committed_conn.track("hindsight_memories", mem_id)

    r = _call(
        _session_loop,
        app_client.post(
            "/v1/recall",
            json={
                "query": "historical fact aof7k",
                "scope": "private",
                "as_of": as_of.isoformat(),
            },
            headers=headers,
        ),
    )
    assert r.status_code == 200
    result_ids = {res["id"] for res in r.json()["results"]}
    assert mem_id in result_ids


def test_as_of_excludes_memory_not_yet_written(
    app_client, _session_loop, committed_conn, tenant_id
) -> None:
    """as_of set before valid_from must not return the memory."""
    from datetime import datetime, timedelta

    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = make_agent_headers(aid, tenant_id)

    valid_start = datetime.now(UTC) - timedelta(days=1)
    as_of = datetime.now(UTC) - timedelta(days=10)  # before valid_from

    mem_id = str(uuid4())
    committed_conn.execute(
        """
        INSERT INTO hindsight_memories
            (id, tenant_id, agent_id, type, content, importance, valid_from)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        mem_id,
        tenant_id,
        aid,
        "mental_model",
        "future fact bz3m",
        0.8,
        valid_start,
    )
    committed_conn.track("hindsight_memories", mem_id)

    r = _call(
        _session_loop,
        app_client.post(
            "/v1/recall",
            json={
                "query": "future fact bz3m",
                "scope": "private",
                "as_of": as_of.isoformat(),
            },
            headers=headers,
        ),
    )
    assert r.status_code == 200
    result_ids = {res["id"] for res in r.json()["results"]}
    assert mem_id not in result_ids
