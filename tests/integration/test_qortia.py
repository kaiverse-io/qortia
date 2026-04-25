"""Qortia API integration tests."""
from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import asyncpg
import pytest
from httpx import AsyncClient

from tests.integration.conftest import fresh_agent_headers, VAULT_TOKEN, create_active_agent


def _call(loop: asyncio.AbstractEventLoop, coro):  # type: ignore[return]
    return loop.run_until_complete(coro)


def _active_agent(loop, conn, tenant_id: str) -> str:
    return create_active_agent(conn, tenant_id)


# ── remember ──────────────────────────────────────────────────────────────────

def test_remember_requires_auth(app_client, _session_loop) -> None:
    r = _call(_session_loop, app_client.post("/v1/remember", json={
        "memories": [{"type": "episodic", "content": "test"}]
    }))
    assert r.status_code == 401


def test_remember_inactive_agent_403(app_client, _session_loop) -> None:
    r = _call(_session_loop, app_client.post("/v1/remember", json={
        "memories": [{"type": "episodic", "content": "test"}]
    }, headers=fresh_agent_headers()))
    assert r.status_code == 403


def test_remember_empty_batch_422(app_client, _session_loop) -> None:
    r = _call(_session_loop, app_client.post("/v1/remember",
        json={"memories": []}, headers=fresh_agent_headers()))
    assert r.status_code == 422


def test_remember_invalid_type_422(app_client, _session_loop) -> None:
    r = _call(_session_loop, app_client.post("/v1/remember", json={
        "memories": [{"type": "org_chart", "content": "test"}]
    }, headers=fresh_agent_headers()))
    assert r.status_code == 422


def test_remember_batch_atomicity(app_client, _session_loop, committed_conn, tenant_id) -> None:
    """All inserts + counter increment in one transaction."""
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = {"X-Agent-ID": aid, "X-Tenant-ID": tenant_id}

    r = _call(_session_loop, app_client.post("/v1/remember", json={"memories": [
        {"type": "episodic", "content": "event one"},
        {"type": "episodic", "content": "event two"},
        {"type": "lesson", "content": "learned something"},
    ]}, headers=headers))
    assert r.status_code == 200
    assert len(r.json()["ids"]) == 3

    count = committed_conn.fetchval(
        "SELECT COUNT(*) FROM hindsight_memories WHERE agent_id = $1", aid
    )
    assert count == 3

    counter = committed_conn.fetchval(
        "SELECT reflection_counter FROM auth.agents WHERE id = $1", aid
    )
    assert counter == 2  # only episodic memories increment counter


def test_remember_entities_extracted(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    r = _call(_session_loop, app_client.post("/v1/remember", json={"memories": [
        {"type": "decision", "content": "chose PostgreSQL for the database"}
    ]}, headers={"X-Agent-ID": aid, "X-Tenant-ID": tenant_id}))
    assert r.status_code == 200

    row = committed_conn.fetchval(
        "SELECT entities FROM hindsight_memories WHERE agent_id = $1", aid
    )
    assert row is not None
    assert isinstance(json.loads(row), list)


# ── remember-org ──────────────────────────────────────────────────────────────

def test_remember_org_handoff(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    r = _call(_session_loop, app_client.post("/v1/remember-org", json={
        "type": "handoff", "title": "Completed auth", "content": "Done",
    }, headers={"X-Agent-ID": aid, "X-Tenant-ID": tenant_id}))
    assert r.status_code == 200
    assert "id" in r.json()


def test_remember_org_process_requires_chief(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)  # engineer
    r = _call(_session_loop, app_client.post("/v1/remember-org", json={
        "type": "process", "title": "Deploy", "content": "How we deploy",
    }, headers={"X-Agent-ID": aid, "X-Tenant-ID": tenant_id}))
    assert r.status_code == 403


def test_remember_org_invalid_type_422(app_client, _session_loop) -> None:
    r = _call(_session_loop, app_client.post("/v1/remember-org", json={
        "type": "org_chart", "title": "t", "content": "c",
    }, headers=fresh_agent_headers()))
    assert r.status_code == 422


# ── recall ────────────────────────────────────────────────────────────────────

def test_recall_requires_auth(app_client, _session_loop) -> None:
    r = _call(_session_loop, app_client.post("/v1/recall", json={"query": "test"}))
    assert r.status_code == 401


def test_recall_empty_query_422(app_client, _session_loop) -> None:
    r = _call(_session_loop, app_client.post("/v1/recall",
        json={"query": ""}, headers=fresh_agent_headers()))
    assert r.status_code == 422


def test_recall_invalid_scope_422(app_client, _session_loop) -> None:
    r = _call(_session_loop, app_client.post("/v1/recall",
        json={"query": "test", "scope": "invalid"}, headers=fresh_agent_headers()))
    assert r.status_code == 422


def test_recall_returns_results(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = {"X-Agent-ID": aid, "X-Tenant-ID": tenant_id}

    _call(_session_loop, app_client.post("/v1/remember", json={"memories": [
        {"type": "decision", "content": "chose PostgreSQL for the database"}
    ]}, headers=headers))

    r = _call(_session_loop, app_client.post("/v1/recall", json={
        "query": "database", "scope": "private",
    }, headers=headers))
    assert r.status_code == 200
    assert "results" in r.json()


def test_recall_entities_filter(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = {"X-Agent-ID": aid, "X-Tenant-ID": tenant_id}

    _call(_session_loop, app_client.post("/v1/remember", json={"memories": [
        {"type": "decision", "content": "chose PostgreSQL"},
    ]}, headers=headers))

    r = _call(_session_loop, app_client.post("/v1/recall", json={
        "query": "database", "scope": "private", "entities": ["PostgreSQL"],
    }, headers=headers))
    assert r.status_code == 200


# ── context ───────────────────────────────────────────────────────────────────

def test_context_requires_agent_auth(app_client, _session_loop) -> None:
    r = _call(_session_loop, app_client.get("/v1/context"))
    assert r.status_code == 401


def test_context_rejects_user_auth(app_client, _session_loop, user_headers) -> None:
    r = _call(_session_loop, app_client.get("/v1/context", headers=user_headers))
    assert r.status_code == 403


def test_context_structure(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    r = _call(_session_loop, app_client.get("/v1/context",
        headers={"X-Agent-ID": aid, "X-Tenant-ID": tenant_id}))
    assert r.status_code == 200
    data = r.json()
    for key in ("org_chart", "processes", "handoffs", "memories"):
        assert key in data
    for key in ("mental_models", "decisions", "lessons"):
        assert key in data["memories"]


# ── forget ────────────────────────────────────────────────────────────────────

def test_forget_own_memory(app_client, _session_loop, committed_conn, tenant_id) -> None:
    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = {"X-Agent-ID": aid, "X-Tenant-ID": tenant_id}

    r = _call(_session_loop, app_client.post("/v1/remember", json={"memories": [
        {"type": "episodic", "content": "to be forgotten"}
    ]}, headers=headers))
    mem_id = r.json()["ids"][0]

    r = _call(_session_loop, app_client.post("/v1/forget",
        json={"id": mem_id}, headers=headers))
    assert r.status_code == 200

    count = committed_conn.fetchval(
        "SELECT COUNT(*) FROM hindsight_memories WHERE id = $1", mem_id
    )
    assert count == 0


# ── reflect ───────────────────────────────────────────────────────────────────

def test_reflect_supersede_first(app_client, _session_loop, committed_conn, tenant_id, _vault_addr) -> None:
    # Seed LiteLLM key for this tenant in Vault so reflect can proceed
    import hvac
    vault = hvac.Client(url=_vault_addr, token=VAULT_TOKEN)
    vault.secrets.kv.v2.create_or_update_secret(
        path=f"{tenant_id}/litellm_key", secret={"key": "sk-test-master"}
    )

    aid = _active_agent(_session_loop, committed_conn, tenant_id)
    headers = {"X-Agent-ID": aid, "X-Tenant-ID": tenant_id}

    # Existing consolidated memory
    old_id = str(uuid4())
    committed_conn.execute("""
        INSERT INTO hindsight_memories
            (id, tenant_id, agent_id, type, content, importance, is_consolidated)
        VALUES ($1, $2, $3, $4, $5, $6, true)
    """, old_id, tenant_id, aid, "mental_model", "old model", 0.8)
    committed_conn.track("hindsight_memories", old_id)

    # Write episodic memories
    _call(_session_loop, app_client.post("/v1/remember", json={"memories": [
        {"type": "episodic", "content": f"event {i}"} for i in range(3)
    ]}, headers=headers))

    r = _call(_session_loop, app_client.post("/v1/reflect", json={}, headers=headers))
    assert r.status_code == 200
    assert "memories_written" in r.json()

    # Old memory must be superseded
    is_consolidated = committed_conn.fetchval(
        "SELECT is_consolidated FROM hindsight_memories WHERE id = $1", old_id
    )
    assert is_consolidated is False
