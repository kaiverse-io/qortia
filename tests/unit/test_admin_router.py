"""Unit tests for qortia.auth.require_admin and qortia.admin_router (ADR-004).

Mirrors the existing split for the tenant-auth path: qortia.auth.require_agent
and the provisioning CLI both get direct, mocked-DB unit coverage
(tests/unit/test_provisioning_eval_behavior.py) plus a real-DB integration
test (tests/integration/test_provisioning_api.py). This file is the
require_admin/admin_router half of that same split; the HTTP-level tests here
mount qortia.admin_router onto a throwaway FastAPI() instance rather than the
shared qortia.app singleton, so toggling config.settings.qortia_admin_token
per-test can't leak into the integration suite's session-scoped app.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

# ── require_admin (direct, function-level) ──────────────────────────────────


@pytest.mark.asyncio
async def test_require_admin_404_when_token_unset() -> None:
    from qortia import config
    from qortia.auth import require_admin

    old = config.settings.qortia_admin_token
    config.settings.qortia_admin_token = ""
    try:
        with pytest.raises(HTTPException) as exc:
            await require_admin("Bearer anything")
        assert exc.value.status_code == 404
    finally:
        config.settings.qortia_admin_token = old


@pytest.mark.asyncio
async def test_require_admin_401_when_header_missing() -> None:
    from qortia import config
    from qortia.auth import require_admin

    old = config.settings.qortia_admin_token
    config.settings.qortia_admin_token = "secret-token"
    try:
        with pytest.raises(HTTPException) as exc:
            await require_admin(None)
        assert exc.value.status_code == 401
    finally:
        config.settings.qortia_admin_token = old


@pytest.mark.asyncio
async def test_require_admin_401_when_header_malformed() -> None:
    from qortia import config
    from qortia.auth import require_admin

    old = config.settings.qortia_admin_token
    config.settings.qortia_admin_token = "secret-token"
    try:
        with pytest.raises(HTTPException) as exc:
            await require_admin("Basic secret-token")
        assert exc.value.status_code == 401
    finally:
        config.settings.qortia_admin_token = old


@pytest.mark.asyncio
async def test_require_admin_401_when_token_blank() -> None:
    from qortia import config
    from qortia.auth import require_admin

    old = config.settings.qortia_admin_token
    config.settings.qortia_admin_token = "secret-token"
    try:
        with pytest.raises(HTTPException) as exc:
            await require_admin("Bearer   ")
        assert exc.value.status_code == 401
    finally:
        config.settings.qortia_admin_token = old


@pytest.mark.asyncio
async def test_require_admin_401_when_token_wrong() -> None:
    from qortia import config
    from qortia.auth import require_admin

    old = config.settings.qortia_admin_token
    config.settings.qortia_admin_token = "secret-token"
    try:
        with pytest.raises(HTTPException) as exc:
            await require_admin("Bearer wrong-token")
        assert exc.value.status_code == 401
    finally:
        config.settings.qortia_admin_token = old


@pytest.mark.asyncio
async def test_require_admin_passes_when_token_correct() -> None:
    from qortia import config
    from qortia.auth import require_admin

    old = config.settings.qortia_admin_token
    config.settings.qortia_admin_token = "secret-token"
    try:
        assert await require_admin("Bearer secret-token") is None
    finally:
        config.settings.qortia_admin_token = old


# ── route handlers (direct, function-level, mocked pool/provisioning) ──────


@pytest.mark.asyncio
async def test_admin_create_tenant_returns_tenant_id() -> None:
    from qortia.admin_router import TenantCreateRequest, admin_create_tenant

    tenant_id = uuid4()
    with patch("qortia.admin_router.create_tenant", AsyncMock(return_value=tenant_id)):
        resp = await admin_create_tenant(TenantCreateRequest(name="Acme"))
    assert resp.tenant_id == str(tenant_id)


@pytest.mark.asyncio
async def test_admin_create_tenant_name_is_optional() -> None:
    from qortia.admin_router import TenantCreateRequest, admin_create_tenant

    tenant_id = uuid4()
    create_mock = AsyncMock(return_value=tenant_id)
    with patch("qortia.admin_router.create_tenant", create_mock):
        await admin_create_tenant(TenantCreateRequest())
    assert create_mock.call_args.kwargs["name"] is None


@pytest.mark.asyncio
async def test_admin_create_agent_returns_agent_id() -> None:
    from qortia.admin_router import AgentCreateRequest, admin_create_agent

    tenant_id = uuid4()
    agent_id = uuid4()
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    with (
        patch("qortia.admin_router.get_main_pool", return_value=pool),
        patch("qortia.admin_router.create_agent", AsyncMock(return_value=agent_id)) as create_mock,
    ):
        resp = await admin_create_agent(
            AgentCreateRequest(
                tenant_id=tenant_id, name="bot", clearance_level="restricted", division="research"
            )
        )
    assert resp.agent_id == str(agent_id)
    create_mock.assert_awaited_once_with(
        pool, tenant_id, name="bot", clearance_level="restricted", division="research"
    )


@pytest.mark.asyncio
async def test_admin_create_agent_defaults_match_provisioning_defaults() -> None:
    """clearance_level/division defaults on the HTTP body must mirror
    qortia.provisioning.create_agent's own defaults exactly."""
    from qortia.admin_router import AgentCreateRequest

    body = AgentCreateRequest(tenant_id=uuid4())
    assert body.clearance_level == "internal"
    assert body.division == "all"
    assert body.name is None


@pytest.mark.asyncio
async def test_admin_create_agent_404_when_tenant_missing() -> None:
    from qortia.admin_router import AgentCreateRequest, admin_create_agent

    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=None)
    with (
        patch("qortia.admin_router.get_main_pool", return_value=pool),
        patch("qortia.admin_router.create_agent", AsyncMock()) as create_mock,
    ):
        with pytest.raises(HTTPException) as exc:
            await admin_create_agent(AgentCreateRequest(tenant_id=uuid4()))
    assert exc.value.status_code == 404
    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_issue_api_key_returns_key() -> None:
    from qortia.admin_router import ApiKeyIssueRequest, admin_issue_api_key

    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    with (
        patch("qortia.admin_router.get_main_pool", return_value=pool),
        patch("qortia.admin_router.issue_api_key", AsyncMock(return_value="qortia_sk_test")),
    ):
        resp = await admin_issue_api_key(ApiKeyIssueRequest(tenant_id=uuid4()))
    assert resp.api_key == "qortia_sk_test"


@pytest.mark.asyncio
async def test_admin_issue_api_key_404_when_tenant_missing() -> None:
    from qortia.admin_router import ApiKeyIssueRequest, admin_issue_api_key

    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=None)
    with (
        patch("qortia.admin_router.get_main_pool", return_value=pool),
        patch("qortia.admin_router.issue_api_key", AsyncMock()) as issue_mock,
    ):
        with pytest.raises(HTTPException) as exc:
            await admin_issue_api_key(ApiKeyIssueRequest(tenant_id=uuid4()))
    assert exc.value.status_code == 404
    issue_mock.assert_not_awaited()


# ── HTTP-level, isolated throwaway app (real ASGI dispatch, mocked DB) ─────
# A fresh FastAPI() instance mounting only qortia.admin_router — never the
# qortia.app singleton other tests share — so toggling
# config.settings.qortia_admin_token here can't affect the integration
# suite's session-scoped app (see tests/integration/test_provisioning_api.py
# for the real-DB round trip through that shared app).


def _throwaway_admin_app() -> FastAPI:
    from qortia.admin_router import router as admin_router

    app = FastAPI()
    app.include_router(admin_router)
    return app


@pytest.mark.asyncio
async def test_admin_route_404_over_http_when_token_unset() -> None:
    from qortia import config

    old = config.settings.qortia_admin_token
    config.settings.qortia_admin_token = ""
    try:
        app = _throwaway_admin_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/admin/tenants", json={"name": "Acme"})
        assert resp.status_code == 404
    finally:
        config.settings.qortia_admin_token = old


@pytest.mark.asyncio
async def test_admin_route_401_over_http_when_token_missing() -> None:
    from qortia import config

    old = config.settings.qortia_admin_token
    config.settings.qortia_admin_token = "secret-token"
    try:
        app = _throwaway_admin_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/admin/tenants", json={"name": "Acme"})
        assert resp.status_code == 401
    finally:
        config.settings.qortia_admin_token = old


@pytest.mark.asyncio
async def test_admin_route_401_over_http_when_token_wrong() -> None:
    from qortia import config

    old = config.settings.qortia_admin_token
    config.settings.qortia_admin_token = "secret-token"
    try:
        app = _throwaway_admin_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/admin/tenants",
                json={"name": "Acme"},
                headers={"Authorization": "Bearer nope"},
            )
        assert resp.status_code == 401
    finally:
        config.settings.qortia_admin_token = old


@pytest.mark.asyncio
async def test_admin_route_200_over_http_when_token_correct() -> None:
    from qortia import config

    old = config.settings.qortia_admin_token
    config.settings.qortia_admin_token = "secret-token"
    tenant_id = uuid4()
    try:
        app = _throwaway_admin_app()
        transport = ASGITransport(app=app)
        with patch("qortia.admin_router.create_tenant", AsyncMock(return_value=tenant_id)):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/admin/tenants",
                    json={"name": "Acme"},
                    headers={"Authorization": "Bearer secret-token"},
                )
        assert resp.status_code == 200
        assert resp.json() == {"tenant_id": str(tenant_id)}
    finally:
        config.settings.qortia_admin_token = old
