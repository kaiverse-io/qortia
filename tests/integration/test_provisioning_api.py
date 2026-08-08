"""Integration tests for tenant/agent/API-key provisioning and auth.

Covers both provisioning callers against a real DB: the qortia.provisioning
functions used directly (as qortia-admin CLI does) and the same functions
reached over HTTP via qortia.admin_router (ADR-004).
"""

from __future__ import annotations

from uuid import UUID, uuid4

import asyncpg
import pytest
from fastapi import HTTPException

from tests.integration.conftest import ADMIN_TOKEN, _call


def test_provisioning_api_keys_and_require_agent_real_db(
    _session_loop, pg_url: str, pg_superuser_url: str
) -> None:
    async def scenario() -> None:
        from qortia.auth import hash_api_key, require_agent
        from qortia.provisioning import create_agent, create_tenant, issue_api_key, revoke_api_key

        pool = await asyncpg.create_pool(pg_url)
        try:
            tenant_id = await create_tenant(pool, name="Provisioning Co")
            agent_id = await create_agent(
                pool,
                tenant_id,
                clearance_level="restricted",
                division="research",
            )
            api_key = await issue_api_key(pool, tenant_id)

            identity = await require_agent(f"Bearer {api_key}", str(agent_id))
            assert identity.tenant_id == tenant_id
            assert identity.agent_id == agent_id
            assert identity.clearance_order == 3
            assert identity.division == "research"

            wrong_agent = uuid4()
            with pytest.raises(HTTPException) as wrong_agent_error:
                await require_agent(f"Bearer {api_key}", str(wrong_agent))
            assert wrong_agent_error.value.status_code == 403

            with pytest.raises(HTTPException) as malformed_agent_error:
                await require_agent(f"Bearer {api_key}", "not-a-uuid")
            assert malformed_agent_error.value.status_code == 401

            key_id = await pool.fetchval(
                "SELECT id FROM qortia_api_keys WHERE key_hash = $1",
                hash_api_key(api_key),
            )
            assert key_id is not None
            await revoke_api_key(pool, key_id)

            with pytest.raises(HTTPException) as revoked_key_error:
                await require_agent(f"Bearer {api_key}", str(agent_id))
            assert revoked_key_error.value.status_code == 401

            with pytest.raises(HTTPException) as blank_key_error:
                await require_agent("Bearer   ", str(agent_id))
            assert blank_key_error.value.status_code == 401

            with pytest.raises(HTTPException) as missing_agent_error:
                await require_agent(f"Bearer {api_key}", None)
            assert missing_agent_error.value.status_code == 401
        finally:
            if "tenant_id" in locals():
                su = await asyncpg.connect(pg_superuser_url)
                try:
                    await su.execute("DELETE FROM qortia_tenants WHERE id = $1", tenant_id)
                finally:
                    await su.close()
            await pool.close()

    _call(_session_loop, scenario())


# ── /v1/admin/* over real HTTP (ADR-004) ────────────────────────────────────
# The shared app_client is built with QORTIA_ADMIN_TOKEN=ADMIN_TOKEN (see
# tests/integration/conftest.py::_app_and_loop), so qortia.admin_router is
# mounted for the whole session here. The "unmounted/404 when
# QORTIA_ADMIN_TOKEN is unset" behavior is covered instead by
# tests/unit/test_admin_router.py against an isolated throwaway app — doing
# that here would mean reloading the qortia.app singleton this whole session
# shares, which risks disturbing every other integration test.


def test_admin_provisioning_http_round_trip_real_db(
    _session_loop, app_client, pg_superuser_url: str
) -> None:
    """POST /v1/admin/tenants -> /agents -> /keys against a real DB, then
    proves the freshly issued key/agent pair actually authenticates."""

    async def scenario() -> None:
        headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
        tenant_id = ""
        try:
            tenant_resp = await app_client.post(
                "/v1/admin/tenants", json={"name": "Admin HTTP Co"}, headers=headers
            )
            assert tenant_resp.status_code == 200
            tenant_id = tenant_resp.json()["tenant_id"]

            agent_resp = await app_client.post(
                "/v1/admin/agents",
                json={
                    "tenant_id": tenant_id,
                    "name": "bot-1",
                    "clearance_level": "restricted",
                    "division": "research",
                },
                headers=headers,
            )
            assert agent_resp.status_code == 200
            agent_id = agent_resp.json()["agent_id"]

            key_resp = await app_client.post(
                "/v1/admin/keys", json={"tenant_id": tenant_id}, headers=headers
            )
            assert key_resp.status_code == 200
            api_key = key_resp.json()["api_key"]
            assert api_key.startswith("qortia_sk_")

            # The row admin_create_agent wrote actually has the fields we sent.
            su = await asyncpg.connect(pg_superuser_url)
            try:
                row = await su.fetchrow(
                    "SELECT name, clearance_level, division FROM qortia_agents WHERE id = $1",
                    UUID(agent_id),
                )
            finally:
                await su.close()
            assert row is not None
            assert row["name"] == "bot-1"
            assert row["clearance_level"] == "restricted"
            assert row["division"] == "research"

            # The issued key + agent pair round-trips through the normal
            # tenant-scoped auth path (qortia.auth.require_agent), same as
            # any key minted by qortia-admin or qortia.provisioning directly.
            ctx_resp = await app_client.get(
                "/v1/context",
                headers={"Authorization": f"Bearer {api_key}", "X-Agent-Id": agent_id},
            )
            assert ctx_resp.status_code == 200
        finally:
            if tenant_id:
                su = await asyncpg.connect(pg_superuser_url)
                try:
                    await su.execute("DELETE FROM qortia_tenants WHERE id = $1", UUID(tenant_id))
                finally:
                    await su.close()

    _call(_session_loop, scenario())


def test_admin_provisioning_http_rejects_missing_or_wrong_token(_session_loop, app_client) -> None:
    async def scenario() -> None:
        no_auth = await app_client.post("/v1/admin/tenants", json={"name": "x"})
        assert no_auth.status_code == 401

        wrong_token = await app_client.post(
            "/v1/admin/tenants",
            json={"name": "x"},
            headers={"Authorization": "Bearer definitely-not-the-admin-token"},
        )
        assert wrong_token.status_code == 401

    _call(_session_loop, scenario())


def test_admin_agent_and_key_404_for_unknown_tenant_real_db(_session_loop, app_client) -> None:
    async def scenario() -> None:
        headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
        missing_tenant = str(uuid4())

        agent_resp = await app_client.post(
            "/v1/admin/agents", json={"tenant_id": missing_tenant}, headers=headers
        )
        assert agent_resp.status_code == 404

        key_resp = await app_client.post(
            "/v1/admin/keys", json={"tenant_id": missing_tenant}, headers=headers
        )
        assert key_resp.status_code == 404

    _call(_session_loop, scenario())
