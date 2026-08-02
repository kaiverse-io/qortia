"""Integration tests for tenant/agent/API-key provisioning and auth."""

from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest
from fastapi import HTTPException

from tests.integration.conftest import _call


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
