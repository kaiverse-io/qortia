---
kind: index
status: active
owner: platform
last_reviewed: 2026-08-02
---

# Qortia — Documentation Index

Portable Postgres+pgvector memory layer for AI agents
(remember / recall / reflect / knowledge).

## Start here (canonical)

| Doc | Purpose |
|-----|---------|
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | **Current** component map, auth, RLS, workers, module contracts |
| [`explanation/extraction-requirements.md`](explanation/extraction-requirements.md) | What was rebuilt at extraction from the host platform |
| [`explanation/competitive-landscape.md`](explanation/competitive-landscape.md) | Positioning vs mem0 / Hindsight / Graphiti / etc. |
| [`decisions/adrs/adr-001-standalone-extraction.md`](decisions/adrs/adr-001-standalone-extraction.md) | Standalone identity, secrets, workers (active ADR) |

## Eval & benchmarking (still useful; paths may lag)

| Doc | Purpose |
|-----|---------|
| [`02-benchmarking.md`](02-benchmarking.md) | REH / ALB / PIB benchmarking guide |
| [`03-eval-strategy.md`](03-eval-strategy.md) | Why memory quality is measured this way |
| [`evals/README.md`](../evals/README.md) | Harness entrypoints in this repo |

## Enhancements

Design notes under [`enhancements/`](enhancements/). Many still mention
host-platform paths (`platform/app/...`); treat those path strings as historical.
Behavior claims must match `ARCHITECTURE.md` / `src/qortia/` — see
[`enhancements/README.md`](enhancements/README.md).

## Archive (do not cite as current)

Host-platform-era design docs and ADR dump:

→ [`archive/extraction/README.md`](archive/extraction/README.md)

## Module map

Package root: `src/qortia/` (not `platform/app/qortia/`). Module sections live in
`ARCHITECTURE.md` and are gated by `just ci-arch`.
