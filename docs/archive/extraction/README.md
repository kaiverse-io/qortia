---
kind: archive
status: historical
last_reviewed: 2026-08-02
---

# Extraction archive (pre-standalone)

These documents were written when Qortia lived inside a larger host platform
(API mounted under that platform's app tree, JWT/Vault identity, in-process
workers, work-order routers). They were copied into this repo at extraction and
**must not be treated as current behavior**.

## Canonical docs for standalone Qortia

| Need | Read |
|------|------|
| What the code does now | [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) |
| What was rebuilt at extraction | [`docs/explanation/extraction-requirements.md`](../../explanation/extraction-requirements.md) |
| Active doc index | [`docs/index.md`](../../index.md) |
| New decisions | [`docs/decisions/adrs/`](../../decisions/adrs/) (MADR) |

## What lives here

| File | Why archived |
|------|----------------|
| `00-overview.md` | Diagrams and paths still say `platform/app/qortia/`, JWT, work orders |
| `01-design.md` | Host-platform design: Vault identity, `mcp_bridge`, monolith workers |
| `04-api-contracts.md` | JWT-derived identity; migration IDs from the host schema history |
| `legacy-adrs.md` | Host-era ADR dump (useful provenance; several decisions superseded) |

When a fact from these files is still true, restate it in `ARCHITECTURE.md` or a
new MADR under `docs/decisions/adrs/` — do not cite this archive as current.
