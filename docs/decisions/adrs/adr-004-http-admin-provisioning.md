---
kind: adr
status: accepted
owner: platform
last_reviewed: 2026-08-08
---

# ADR-004 — Platform-admin HTTP provisioning (`/v1/admin/*`)

- **Status:** Accepted (2026-08-08)
- **Deciders:** founder
- **Related:** ADR-001 (standalone extraction — auth model), `qortia.provisioning`
  module docstring, `SECURITY.md` (already scoped "RLS / auth bypass on the
  HTTP admin or memory APIs" before this existed)

## Context

`qortia.provisioning` (`create_tenant`, `create_agent`, `issue_api_key`) has been
CLI-only (`qortia-admin`) and direct-function-call-only since the standalone
extraction (ADR-001), by design: the very first API key for a fresh tenant
can't be minted through a *tenant-scoped* API that itself requires an existing
tenant API key to authenticate the request. That's a real bootstrapping
problem, and the CLI (direct DB access, no HTTP at all) is a correct answer
to it.

It is not, however, the only bootstrapping problem in play. A caller that
needs to provision Qortia tenants programmatically — e.g. a separate
control-plane service creating a Qortia tenant/agent/key per one of *its own*
tenants — may have no shell into wherever Qortia runs at all: a different
container, a different host, possibly a different operator entirely. For
that caller, "just run `qortia-admin`" isn't an out-of-band path, it's not a
path. Qortia already accepts an analogous distinction elsewhere:
`config.settings` is "a plain environment variable with a sane local
default" (`qortia.config` docstring), and `qortia.eval_router` already
establishes the pattern of an HTTP surface gated by a static env flag —
unmounted by `qortia.app` and 404ing per-handler unless that flag is
deliberately set. The gap this ADR closes is the same shape, with the gate
being a secret bearer token instead of a boolean.

## Decision

1. **New router** `qortia.admin_router`, mounted at `/v1/admin/*`, calling
   `qortia.provisioning.create_tenant` / `create_agent` / `issue_api_key`
   directly — no reimplementation of that logic, and no `tenant_transaction`/
   RLS involved, because `qortia_tenants`/`qortia_agents`/`qortia_api_keys`
   carry no RLS policies in `migrations/V1__initial_schema.sql` (identity
   substrate, not tenant-scoped memory data). That was already true of the
   CLI path and stays true here — provisioning is inherently a cross-tenant,
   superuser-ish operation.
2. **New static credential** `QORTIA_ADMIN_TOKEN` (env var, `qortia.config`),
   distinct from per-tenant API keys: one platform-level bearer token set
   out-of-band by whoever deploys Qortia, checked with `secrets.compare_digest`
   (`qortia.auth.require_admin`) rather than the SHA-256-hash-and-DB-lookup
   path `require_agent` uses for tenant keys — there is nothing to look up,
   it's a single process-wide secret compared in memory.
3. **Off by default, on two levels** (mirrors `qortia.eval_router`):
   `qortia.app` only mounts `admin_router` when `QORTIA_ADMIN_TOKEN` is set,
   and `require_admin` independently 404s (not 401) when it's unset — so a
   deployment that never opted in doesn't even reveal the surface exists.
4. **Exact contract** (the calling side's starting proposal, landed unchanged
   except as noted in point 5):
   - `POST /v1/admin/tenants` `{"name": string | null}` → `200
     {"tenant_id": string}`
   - `POST /v1/admin/agents` `{"tenant_id": string, "name": string | null,
     "clearance_level": string, "division": string}` (`clearance_level`/
     `division` default to `"internal"`/`"all"`, matching `create_agent`'s
     own defaults) → `200 {"agent_id": string}`, or `404` if `tenant_id`
     doesn't exist
   - `POST /v1/admin/keys` `{"tenant_id": string}` → `200 {"api_key":
     string}`, or `404` if `tenant_id` doesn't exist
   - Auth: `Authorization: Bearer <QORTIA_ADMIN_TOKEN>` on every route, applied
     once via `APIRouter(dependencies=[Depends(require_admin)])` rather than
     per-handler — there's no admin identity payload worth injecting into each
     handler, unlike `AgentIdentity` for `require_agent`.
   - No `X-Agent-Id`: admin routes are platform-level, not tenant/agent-scoped.
5. **One deliberate deviation from the starting proposal:** `create_agent`
   gained an optional `name: str | None = None` parameter (here, and in the
   `qortia-admin create-agent --name` CLI for parity) instead of the HTTP
   layer silently accepting and dropping a `name` field the underlying
   function had no way to persist. `qortia_agents.name` already existed
   (nullable column, read by `qortia.knowledge`'s weekly-summary agent
   attribution) and simply wasn't wired up to either caller before now.

## Consequences

**Good:** a caller across a container boundary (no shell into Qortia) can
provision tenants/agents/keys the same way the CLI always could, through the
same functions, with the same off-by-default posture as every other opt-in
surface in this codebase. `SECURITY.md` already scoped "RLS / auth bypass on
the HTTP admin ... API" before this ADR — this fills in what that sentence
was pointing at.

**Bad / risks:** a second secret to manage (`QORTIA_ADMIN_TOKEN`) alongside
`QORTIA_LITELLM_API_KEY`/the DB URL — operators who don't need HTTP
provisioning simply never set it and get the pre-ADR-004 posture exactly
(unmounted router, CLI/direct-call only). A leaked admin token grants
tenant/agent/key creation across every tenant on the instance — the same
blast radius CLI access to the DB credential already had — but it does not
grant recall/remember/reflect access to any tenant's memory data, since
`admin_router` never touches `tenant_transaction` or the memory tables at
all. Rotation today is "change the env var and restart the process," the
same granularity as the CLI's DB credential; a revocable, DB-backed
admin-token scheme is a reasonable future increment if multiple distinct
admin callers ever need independent revocation — not needed for a single
caller and not built here.
