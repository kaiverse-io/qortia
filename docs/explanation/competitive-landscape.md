# Competitive Landscape

Status: living document. Last refreshed: 2026-07-31.

This is the historical record of the market research that shaped Qortia's design and
extraction priorities. Every finding below was produced by cloning the real repository
and reading actual source code — not by reading READMEs or marketing pages. File:line
citations were captured at research time; treat them as snapshots, not live references,
since all five projects move fast (Hindsight in particular releases near-daily).

See the tracking issue for the cadence this should be refreshed on.

## Method

For each project: clone at the current release tag, read the write/ingest path and the
read/recall path in actual source, check the license file directly (not the README
badge), and compare against Qortia's spec:

- hybrid lexical+vector recall with real RRF fusion
- entity-graph traversal with multi-hop adjacency boosting
- memory-links between related memories
- scheduled background "reflect" consolidation that supersedes raw episodic memories
  into higher-level `mental_model`/`lesson` records
- confidence-decay scoring tied to downstream task-outcome feedback
- temporal validity / `as_of` queries
- RBAC-clearance-scoped `org` / `private` / `knowledge` memory tiers
- self-hostable on a dedicated Postgres+pgvector instance, no other required services

## Summary matrix

| | RRF fusion | Graph traversal | Consolidation | Outcome-feedback decay | Temporal / bitemporal | RBAC tiers | Pgvector-only | License |
|---|---|---|---|---|---|---|---|---|
| **Qortia** | real (`RRF_K=60`) | BFS 2-hop, 1.5×/hop decay | scheduled `reflect` → supersede | **yes — only one with this, across all 5** | single `valid_until` | org/private/knowledge + RLS | native | Apache-2.0 |
| mem0 v2.0.13 | weighted-sum, not RRF | 1-hop entity boost only (real graph memory is closed-source/hosted-only) | none — even the older update/delete conflict-resolution logic was dropped from the default path | none | **explicitly rejected in OSS** — raises `ValueError`, docstring says "platform-only" | none, flat filters only | pgvector first-class | Apache-2.0 |
| Hindsight v0.8.5 | real RRF + custom interleave-fusion | pluggable graph, typed causal links (`causes`/`enables`/`prevents`) | consolidator with LLM dedup-adjudicate + cron "mental model" refresh | **verified zero** — grepped the whole engine, no reward/feedback loop | genuine bitemporal columns + point-in-time query | tags + schema-per-tenant only, no fixed clearance model | embedded Postgres, zero-config self-host | MIT |
| cognee v1.4.0 | real RRF + BM25 | recursive-CTE k-hop, Postgres-native | on-demand only (`memify` pipeline), not scheduled | EMA feedback-weight, but manually rating-triggered, not automatic task outcome | none native (optional Graphiti borrow) | dataset + role ACL | genuinely Postgres-only capable | Apache-2.0 |
| Graphiti v0.29.2 | real RRF | native graph traversal | community detection/summarize only, no supersede of raw facts | none | **best-in-class** — full `valid_at`/`invalid_at`/`expired_at` with LLM contradiction resolution | flat `group_id` only | **hard Neo4j dependency, no pgvector path anywhere in the driver layer** | Apache-2.0 |
| Memoria v0.4.0 (matrixorigin) | **none** — fixed linear weighted blend (`0.3·vector + 0.2·keyword + 0.2·time + 0.3·confidence`), not reciprocal-rank | BFS/spreading-activation + trust-tier lifecycle (T4→T3→T2 promotion/demotion via contradiction detection) | real — contradiction detection on edge-similarity drop + tier lifecycle | explicit user-vote (`useful`/`irrelevant`/`outdated`/`wrong`), not measured task success/failure | none found | none — per-user physical DB isolation instead, plus explicit groups | **no — hard MatrixOne dependency** (MySQL wire protocol, proprietary `vecf32` type), no pgvector path at all | Apache-2.0, verified clean, no CLA/dual-license |

## The one finding that matters most

Qortia's outcome-feedback confidence decay — a task's measured success or failure
adjusting the trust multiplier on the memories used in it — is real and, across every
competitor reviewed, unmatched. Not "differently implemented": **absent**. This should
be the thing Qortia leads with, not a fix-it-later item.

## Per-project detail

### mem0 (mem0ai/mem0)

`add()` runs a single LLM extraction call (`ADDITIVE_EXTRACTION_PROMPT`) and stores only
extracted facts, not raw messages. `search()` is genuinely hybrid — vector + Postgres/
Qdrant/Elasticsearch keyword search — fused via a weighted-sum in
`mem0/utils/scoring.py`, not RRF. No `mem0/graphs` module exists in the OSS repo at all;
graph memory is retired from OSS and now hosted-platform-only per
`docs/platform/features/graph-memory.mdx`. What OSS has instead is a lighter inverted
entity index with 1-hop lookup and a capped similarity boost — not a traversable graph.
Reranking is real (multiple pluggable local/hosted rerankers) but off by default.
Temporal/`reference_date` params are explicitly rejected in the OSS path with a
"Platform-only... Not supported in OSS" error. 25 vector-store backends, Postgres/
pgvector genuinely first-class with real hybrid full-text search. Local LLM paths
(Ollama, vLLM, LM Studio) are first-class, no hosted API key required. Apache-2.0,
unmodified.

### Hindsight (vectorize-io/hindsight)

Despite the "Vectorize" company name, this is **not** a thin client to a hosted service
— confirmed no mandatory calls to any Vectorize.io API anywhere in the engine. Ships an
embedded Postgres path (`pg0.py`) so `pip install hindsight-api && hindsight-api` runs
with zero external dependencies — the self-host bar to match. Real LLM-based fact/causal
extraction (`fact_extraction.py`, 2,775 lines), trigram entity resolution with LLM
fallback. Recall is true RRF plus a custom `interleave_fusion` built specifically to fix
an RRF dedup blind spot, followed by a real cross-encoder reranker. Consolidation is
comparable to a reflect pass: LLM-adjudicated create/update/delete on observations, plus
a distinct "mental models" table of user-defined saved queries refreshed on a cron.
Genuine bitemporal columns (`occurred_start`/`occurred_end`/`event_date`/`mentioned_at`)
and point-in-time `query_timestamp` support. "Agent Memory That Learns" does **not**
refer to any reward/reinforcement loop — grepped for `reward`/`bandit`/`reinforcement`,
zero real hits. It means extraction + consolidation + static user-configured "disposition"
traits, nothing outcome-driven. Multi-tenancy is schema-per-tenant via a pluggable
`TenantExtension` — no fixed org/private/knowledge clearance concept, just general
boolean tag-scoping. MIT license. 18.7k★, 309 test files, 49 framework integrations
(LangGraph, CrewAI, AutoGen, Agno, ...), near-daily release cadence since Dec 2025.

### cognee (topoteretes/cognee)

Genuine DAG pipeline architecture (`cognify` → chunk → extract-graph-and-summarize →
persist). Default vector store is LanceDB and default graph store is an embedded
Kuzu-derived engine (`ladybug`) — **but** a real Postgres relational graph adapter exists
(`graph/postgres/adapter.py`, 1,605 lines) with plain `graph_node`/`graph_edge` tables and
k-hop traversal via a single recursive CTE, not app-level BFS loops. Combined with the
pgvector adapter, cognee genuinely **can** run fully on Postgres alone — the only other
project reviewed, besides Qortia itself, with that property. Real hybrid RRF + BM25
recall. Consolidation exists (`consolidate_entity_descriptions.py`, LLM-driven merge of
duplicate entity descriptions) but is part of the opt-in `memify` pipeline family,
invoked on demand, not autonomously scheduled — no cron/APScheduler/celery anywhere in
the API or memify pipelines. A real analog to outcome-feedback exists
(`apply_feedback_weights.py`, streaming EMA update against a 1–5 rating) but it's
manually rating-triggered, not driven by measured task success/failure. Real dataset +
role ACL (`ACL`/`Role`/`Permission` models). Apache-2.0. Very active: v1.4.0 as of
2026-07-17, ~50 CI workflows, 368 test files, near-weekly releases.

### Graphiti (getzep/graphiti)

Graph-first, and **structurally excluded** from a pgvector-only requirement — `GraphProvider`
lists only `NEO4J, FALKORDB, KUZU, NEPTUNE`; `neo4j>=5.26.0` is a hard, non-optional
dependency in `pyproject.toml`; no pgvector adapter exists anywhere in the driver layer.
Kuzu is the only embedded/no-extra-service option and it's not the default. What it does
better than anything else reviewed: **bi-temporal edges**, not a single `valid_until`
column. `EntityEdge` carries `expired_at` (transaction-time), `valid_at`/`invalid_at`
(valid-time), and `reference_time` as independent fields; contradiction resolution is
LLM-driven and closes edges rather than deleting them, so history stays queryable. Real
RRF + BM25 + vector + MMR + graph-distance reranking, all implemented. No reflect-style
consolidation — community detection with LLM-generated summaries clusters and
summarizes but never supersedes raw episodic edges. Isolation is a flat `group_id`
filter with no ACL model. Apache-2.0. Notably, cognee itself ships an optional
integration with `graphiti-core` specifically to borrow this bitemporal model rather
than building its own — a signal worth taking seriously.

### Memoria (matrixorigin/Memoria)

License is clean: unmodified Apache-2.0, verified against the actual `LICENSE` file
(not just the README badge), no NOTICE file, no CLA, no dual-licensing/open-core
pattern — unlike MatrixOrigin's other commercial products. Safe to reference and learn
from. **Architecturally disqualifying for direct pattern-borrowing on storage**, though:
built on MatrixOne (matrixorigin's own distributed database), talking MySQL wire
protocol, using MatrixOne's proprietary `vecf32` vector type — no Postgres/pgvector code
path exists anywhere in the storage crate, and `docker-compose.yml` requires a MatrixOne
container. Started as a Python project, since rewritten in Rust (confirmed via a comment
noting parity with "Python's `_LLM_EXTRACT_PROMPT`"). Recall does **not** use RRF —
verified zero hits for `rrf`/`reciprocal_rank` in the codebase — instead a fixed linear
weighted blend (`store.rs:5678`, `W_VEC=0.3, W_KW=0.2, W_TIME=0.2, W_CONF=0.3`) plus a
separate genuine BFS/spreading-activation graph pass. Two ideas worth designing toward,
properly attributed to "trust-tier lifecycle models in graph-memory systems generally"
rather than re-derived from Memoria's Rust: a **trust-tier promotion/demotion lifecycle**
(T4→T3→T2, driven by age/confidence/cross-session count, with contradiction detection via
association-edge cosine-similarity drop below 0.7), and **per-user physical database
isolation** chosen deliberately to make snapshot/branch/rollback operations safe — an
interesting alternative to RLS, though it's solving a different problem than Qortia's
shared org/private/knowledge clearance tiers. Young (~5 months old, created 2026-03-09)
but CI-disciplined: 524★, 71 forks, 12 contributors (heavily MatrixOrigin-employee-driven),
real CI running integration tests against a live MatrixOne container.

## What this means for Qortia's priorities

In rough priority order, from the code-grounded gap analysis:

1. **Turnkey self-host.** Hindsight's embedded-Postgres, zero-config `pip install &&
   run` is the adoption bar. Qortia today needs a full Postgres + secrets store +
   LLM-proxy stack just to boot — see the extraction requirements doc for what has to
   be rebuilt (auth/tenancy, background workers, secrets) versus what ports cleanly
   (the recall/reflect/consolidation algorithms themselves).
2. **Bitemporal modeling.** Single `valid_until` is behind both Hindsight and especially
   Graphiti. Worth moving toward `valid_at`/`invalid_at`/`expired_at`, following
   Graphiti's model rather than reinventing one — this is exactly the piece cognee
   itself chose to borrow rather than build.
3. **Config-driven model selection.** One module currently hardcodes a specific model
   string. Needs to be fully pluggable for genuine provider portability — mem0 and
   Hindsight both support local LLMs out of the box, Qortia should too.
4. **Typed/causal link taxonomy.** The memory-links table today is just cosine-threshold
   pairs. Hindsight's `causes`/`enables`/`prevents` typed edges are a straightforward,
   valuable addition.
5. **Consider a trust-tier lifecycle**, inspired by (not copied from) Memoria's
   promotion/demotion model, as a refinement to the existing confidence-decay mechanism.

## What to lead with, not fix

The outcome-feedback confidence decay and the RBAC clearance tiers. Nothing reviewed
across five real, actively-developed competitors has either at this level.

## Provenance and licensing note

All five projects reviewed here are permissively licensed (Apache-2.0 ×4, MIT ×1) — no
copyleft anywhere in the comparison set, so no license obligation attaches merely to
having read and described their behavior. The obligation that does apply: no code from
any of these five was, or should be, copy-pasted into Qortia. Techniques are attributed
to their actual origin (e.g. RRF to the Cormack et al. IR literature, not to whichever
OSS repo happened to be read first) rather than to a specific competitor's
implementation. See the tracking issue for the recurring compliance checklist this
feeds into before any public release.
