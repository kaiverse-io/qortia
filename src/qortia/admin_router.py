"""Platform-admin HTTP provisioning — `/v1/admin/*` (ADR-004).

Reaches the same `qortia.provisioning` functions the `qortia-admin` CLI
already calls, but over HTTP — for operators whose caller has no shell into
wherever Qortia runs (e.g. a separate control-plane service in its own
container). Gated end-to-end by a static `QORTIA_ADMIN_TOKEN` bearer token
(`qortia.auth.require_admin`), not a per-tenant API key, so this does not
reopen the "first key for a fresh tenant" bootstrapping problem documented in
`qortia.provisioning`.

Inert by default, the same belt-and-suspenders pattern as
`qortia.eval_router`: `qortia.app` only mounts this router when
`QORTIA_ADMIN_TOKEN` is set, and `require_admin` — applied to every route via
`APIRouter(dependencies=...)` rather than repeated per-handler, since it has
no identity payload worth injecting — 404s outright when the token is unset,
so a deployment that never opted in doesn't leak that the surface exists.

These handlers talk to `qortia.db.get_main_pool()` directly, not
`qortia.db.tenant_transaction`: `qortia_tenants`/`qortia_agents`/
`qortia_api_keys` carry no RLS policies (identity/provisioning predates any
tenant to scope a transaction to), so there is no RLS convention to route
through here — see `qortia.provisioning`'s module docstring.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from qortia.auth import require_admin
from qortia.db import get_main_pool
from qortia.provisioning import create_agent, create_tenant, issue_api_key

router: APIRouter = APIRouter(prefix="/v1/admin", dependencies=[Depends(require_admin)])


async def _assert_tenant_exists(tenant_id: UUID) -> None:
    exists = await get_main_pool().fetchval("SELECT 1 FROM qortia_tenants WHERE id = $1", tenant_id)
    if not exists:
        raise HTTPException(404, "Tenant not found")


# ── POST /v1/admin/tenants ──────────────────────────────────────────────────


class TenantCreateRequest(BaseModel):
    name: str | None = None


class TenantCreateResponse(BaseModel):
    tenant_id: str


@router.post("/tenants", response_model=TenantCreateResponse)
async def admin_create_tenant(body: TenantCreateRequest) -> TenantCreateResponse:
    tenant_id = await create_tenant(get_main_pool(), name=body.name)
    return TenantCreateResponse(tenant_id=str(tenant_id))


# ── POST /v1/admin/agents ───────────────────────────────────────────────────


class AgentCreateRequest(BaseModel):
    tenant_id: UUID
    name: str | None = None
    clearance_level: str = "internal"
    division: str = "all"


class AgentCreateResponse(BaseModel):
    agent_id: str


@router.post("/agents", response_model=AgentCreateResponse)
async def admin_create_agent(body: AgentCreateRequest) -> AgentCreateResponse:
    await _assert_tenant_exists(body.tenant_id)
    agent_id = await create_agent(
        get_main_pool(),
        body.tenant_id,
        name=body.name,
        clearance_level=body.clearance_level,
        division=body.division,
    )
    return AgentCreateResponse(agent_id=str(agent_id))


# ── POST /v1/admin/keys ──────────────────────────────────────────────────────


class ApiKeyIssueRequest(BaseModel):
    tenant_id: UUID


class ApiKeyIssueResponse(BaseModel):
    api_key: str


@router.post("/keys", response_model=ApiKeyIssueResponse)
async def admin_issue_api_key(body: ApiKeyIssueRequest) -> ApiKeyIssueResponse:
    await _assert_tenant_exists(body.tenant_id)
    api_key = await issue_api_key(get_main_pool(), body.tenant_id)
    return ApiKeyIssueResponse(api_key=api_key)
