---
kind: adr
status: accepted
owner: platform
last_reviewed: 2026-08-02
---

# ADR-001 — Standalone extraction: identity, secrets, and workers

- **Status:** Accepted (2026-08-02)
- **Deciders:** founder
- **Supersedes (for standalone Qortia):** host-era Vault/JWT/worker assumptions in
  [`docs/archive/extraction/legacy-adrs.md`](../../archive/extraction/legacy-adrs.md)
  (notably ADR-014/016 process model, ADR-076 embed-key Vault path, ADR-080
  tenant-scoped divisions as hosted)

## Context

Qortia was extracted from an internal host platform into this repository so it can
run as a portable memory layer (any Postgres+pgvector, any LiteLLM-compatible
gateway). The host coupled auth, secrets, and background workers to platform
tables and Vault. Those couplings are documented in
[`docs/explanation/extraction-requirements.md`](../../explanation/extraction-requirements.md).
Host-era ADRs and design docs remain under
[`docs/archive/extraction/`](../../archive/extraction/) for provenance only.

## Decision

1. **Architecture source of truth** for current behavior is
   [`ARCHITECTURE.md`](../../../ARCHITECTURE.md), not archived design docs.
2. **Auth:** per-tenant SHA-256-hashed API keys + `X-Agent-Id` belonging to that
   tenant — no JWT/JWKS, no host IDP, no Vault (`qortia.auth`).
3. **Secrets / model keys:** a single configured LiteLLM key from environment
   (`get_litellm_key` returns `config.settings.litellm_api_key`).
4. **Tenancy / RLS:** `qortia.db.tenant_transaction` sets session GUCs; RLS on
   tenant-scoped tables is the trust boundary. Clearance levels are global
   reference data (`qortia_clearance_levels`), not per-tenant Vault paths.
5. **Workers:** embedding / archival / idle-reflection / weekly-summary loops are
   library entrypoints meant for a **separate worker process**;
   `qortia.app` lifespan does **not** start them.
6. **Schema:** one squashed Flyway-style migration
   (`migrations/V1__initial_schema.sql`) owned by this repo — not the host's
   V2/V3/… history.
7. **New decisions** are MADR files under `docs/decisions/adrs/`. Do not append
   to the archived host-era ADR dump.

## Consequences

**Good:** docs and comments can point to one current model; archive retains
extraction provenance without contradicting CI/agents.

**Discipline:** when changing identity, workers, or schema ownership, update
`ARCHITECTURE.md` in the same PR and add/amend a MADR here — never “fix” the
archive to look current.
