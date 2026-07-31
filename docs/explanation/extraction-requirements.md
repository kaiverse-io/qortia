# Extraction Requirements

Status: historical record. Captured 2026-07-25, at the point of extraction from an
internal memory engine into this standalone repository.

This documents what was actually coupled to the original host platform, what ported
over cleanly, and what still needs building. It was produced by reading every file in
the original package directly — not by reading its own docs as a substitute for the
code — before the extraction cut.

## What ported cleanly

The algorithmic core has no platform coupling beyond ordinary database access
patterns: real RRF fusion (`RRF_K=60`), MMR dedup, BFS entity-graph traversal (2-hop,
1.5×/hop boost, 0.5 decay), confidence-decay-on-outcome, and the scheduled
reflect/consolidation pass are all pure `asyncpg` + LLM-proxy calls. `models.py`
(request/response validation), `embedding_cache.py`, and the pure-function halves of
`recall_helpers.py`/`recall_rerank.py` ported essentially verbatim. Test coverage
(~4,046 lines across integration and unit tests, using `testcontainers` against a real
Postgres) is real and maintained, not aspirational — it ported over largely intact and
is the regression safety net for everything below.

## What required rebuilding, and why

**Auth / tenancy.** Every recall/remember/reflect/knowledge call inline-joined the host
platform's own `agents`/`tenants`/`tenant_clearance_levels`/`tenant_divisions` tables,
and every router endpoint depended on the host's JWT/JWKS verification middleware. None
of this was behind an interface — it was direct SQL and direct dependency injection
against tables this package doesn't own. Qortia needed its own minimal identity + RBAC
schema (tenants, agents with clearance/division) and its own auth layer from scratch.

**Background workers.** Four in-process `asyncio` loops — the embedding worker, an
archival task, an idle-reflection trigger, and a weekly-summary job — rode the host
platform's single long-lived process and its own leader-election helper. A standalone
deployment needs a real worker/scheduler process; the loop logic itself (~400 lines)
ports over close to verbatim once decoupled from the host's supervisor list.

**Secrets.** A client resolved per-tenant provider keys from the host platform's own
secrets manager. Replaced with plain environment-variable configuration (pluggable
later if a real secrets-manager need appears) — this is not something a standalone OSS
project should require operators to stand up extra infrastructure for.

**The client / integration surface didn't exist at all.** The original design describes
an agent-side bridge that injects work-order context and calls the memory API from
inside the agent's own runtime — but that bridge lived in a different repo entirely
(the agent runtime), not in the memory engine's codebase. There was nothing to extract
here; it's net-new. This is also, not coincidentally, the same surface that becomes the
integration point for calling Qortia from any external agent platform.

**Migrations.** All seven core tables carried hard foreign keys into the host
platform's own `tenants`/`agents` tables, and RLS policies referenced session GUCs the
host's connection-pooling layer set on every transaction. Schema-only extraction would
not apply as-is; it needed a minimal standalone tenant/agent schema of its own, with
those foreign keys re-pointed rather than copy-pasted.

## Config surface (for reference)

The full set of tunables inherited from the original implementation: embedding model
and dimension, LLM-proxy URL, connection pool sizing, reflection threshold (10 new
episodics), embedding-cache size/TTL, per-scope result limits (private/org/knowledge),
dedup similarity threshold (0.95) and lookback window (7 days), rerank model, idle
reflection interval/window, `RRF_K` (60), link-similarity threshold (0.70), embedding
batch size, MMR lambda (0.5) and dedup threshold (0.85), confidence-decay multipliers
(success ×1.05, minor-failure ×0.85, critical-failure ×0.60, floor 0.10 / cap 1.0). One
known issue carried into this repo as a tracked fix rather than resolved silently: a
specific reranking/consolidation model string was hardcoded in one module rather than
reading from config — needs to become fully pluggable for genuine provider portability
(see the competitive-landscape doc's priority list).

## Related

See `docs/explanation/competitive-landscape.md` for how these priorities were shaped by
comparison against mem0, Hindsight, cognee, Graphiti, and Memoria.
