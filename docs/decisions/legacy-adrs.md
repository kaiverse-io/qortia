---
kind: decisions
theme: qortia
status: active
owner: platform
last_reviewed: 2026-05-20
---

# Qortia — Architecture Decisions

> Memory service: episodic recall, reflection, org knowledge, embeddings, RBAC, quality.

---

# ADR-002 — Memory: Qortia + Postgres + pgvector

Single Postgres cluster for memory means same RLS, same connection pool, same backup strategy,
no sync problem between stores. pgvector HNSW gives vector search without a separate service —
works correctly at any dataset size, no training data required unlike IVFFlat. Three tables:
`hindsight_memories` (private per-agent), `org_memory` (shared structured memory), and
`org_knowledge` (shared document corpus, section-aware PageIndex). Rejected: SQLite (the agent harness
native — not multi-tenant, not scalable), Qdrant/Pinecone (separate service, separate auth,
separate failure domain), IVFFlat (requires ~100K rows to be effective, wrong for early scale).

---

---

# ADR-014 — Async Embedding Worker

Writing memory rows immediately and computing embeddings asynchronously means `remember()`
is fast (one DB insert) and BM25 search works instantly. Vector search is available within
seconds under normal load. The worker is a simple asyncio background task — no separate
queue table, no separate process. Rejected: synchronous embedding on write (adds LiteLLM
latency to every `remember()` call, blocks the agent).

---

---

# ADR-015 — Agent-Driven Reflection

Platform cron for reflection creates a burst problem at scale — all agents reflecting
simultaneously hammers LiteLLM. Agent-driven reflection is activity-proportional: an agent
that writes 10 episodic memories triggers one reflection. An idle agent triggers none.
The in-process counter in `mcp_bridge.py` (seeded from Postgres at startup) means no
platform coordination needed. Rejected: platform cron (burst problem), per-session reflection
(platform doesn't know when sessions end).

---

---

# ADR-016 — Weekly Summary: Deterministic Concatenation, No LLM

Handoffs are already human-readable summaries written by agents. Synthesising them
with an LLM adds marginal value and costs tokens on every weekly run per tenant.
Deterministic structured concatenation (sort by date, format with agent name + date
header, join with separator) produces an equally useful
weekly summary at zero cost. Running as an asyncio background task inside the FastAPI
monolith means it works identically in Docker Compose and K8s — no pg_cron dependency,
no K8s CronJob. Staggering by `hash(tenant_id) % 7` prevents thundering herd.
Rejected: LiteLLM brain-haiku synthesis (token cost, probabilistic output, unnecessary
for structured handoff data), pg_cron (Postgres extension dependency), K8s CronJob
(different behaviour in Docker Compose vs K8s).

---

---

# ADR-019 — Org Knowledge Corpus: PageIndex Pattern (spaCy, Zero Tokens)

A separate `org_knowledge` table for documents keeps the atomic-unit model of
`hindsight_memories` intact — agent memories are complete thoughts, not file fragments.
Section-aware splitting (by markdown headings) preserves document structure — fixed-token
chunking destroys cross-referential context in dense design docs. PageIndex fields
(`index_summary`, `index_questions`, `index_entities`) are computed synchronously at ingest
time using spaCy `en_core_web_sm` — zero LLM tokens, deterministic, ~5-10ms per section:
`index_summary` = first 2 sentences after heading (regex); `index_entities` = spaCy NER;
`index_questions` = heading + spaCy noun chunks. BM25 searches `index_tsv` (reasoning
artifacts), vector searches `embedding` of `index_summary` — both match against meaning,
not raw text fragments. No async PageIndex worker needed — index fields are always populated
before the row is inserted. Only the embedding remains async (LiteLLM call).
Rejected: LLM-generated index fields (token cost, latency, non-deterministic, async worker
complexity), fixed-token chunking (destroys document structure), external vector store
(separate service, separate auth, separate failure domain).

---

---

# ADR-020 — Type-Routed Recall + Count-Based LLM Re-ranking

Different memory types have different optimal retrieval strategies. A single hybrid pipeline
for all types is wrong: decisions should be retrieved by keyword + recency (you want the
latest decision on a topic), lessons by vector similarity (experiential, semantic match is
correct), episodes by temporal range. Routing by type at the API layer adds no schema
complexity — it is purely query logic.

Reflection fetches are handled entirely inside the `POST /v1/reflect` platform handler
via direct SQL (ORDER BY created_at DESC for episodic, ORDER BY importance DESC for
mental models) — no embedding call, no recall API involvement. This keeps the public
recall API clean with no internal-only parameters leaking through.

LLM re-ranking is triggered when total results < 3 (count-based, not score-based)
or explicit `rerank: true`. Count-based trigger is provider-agnostic — BM25 scores
are corpus-relative and not comparable to a fixed threshold without normalisation.
If fewer than 3 results are returned, the query is genuinely ambiguous and LLM
reasoning adds value. Uses the agent's configured model — never free-worker (memory content is confidential).
Rejected: single pipeline for all types (wrong retrieval strategy for decisions and
episodes), score-threshold auto-trigger (BM25 scores not normalised, fragile),
`strategy` parameter on public recall API (internal concern leaking into public contract),
explicit-only re-rank (never fires in practice — agents don’t know when results are poor).

---

---

# ADR-021 — Memory History: Append-Only Audit Trail (Agent Operations Only)

An append-only `memory_history` table logging every agent-initiated memory operation
(remember, forget, knowledge_ingest, reflect) gives full auditability without touching
the memory tables themselves. `qortia_platform` role has INSERT only — no UPDATE,
no DELETE — making tampering structurally impossible at the DB layer.

Scope: agent-initiated operations only. Platform-internal writes (`org_chart` on
provisioning/deletion, `weekly_summary` from the background task) are exempt —
they have no authoring agent, making the `agent_id NOT NULL` FK structurally
unfillable without a sentinel value. Platform-internal writes are observable via
structured logs and Grafana. The audit trail is not the right mechanism for them.

Rejected: application-level logging only (not tamper-evident), modifying memory
tables with audit columns (adds complexity to hot write path), nullable agent_id
in memory_history (weakens the audit contract — every logged row must have an owner).

---

---

# ADR-027 — Reflect Output: Blind Supersede, Minimal Schema

The platform always flips ALL existing consolidated mental models and lessons to
`is_consolidated = false` after every reflection, regardless of which specific
memories the LLM synthesised. The LLM returns only new memories to write — it
never references existing row IDs. The output schema is minimal:
`{memories: [{type, content, importance}]}`.

This is correct because: the consolidated set is capped at boot context load time
regardless (20 mental models + 20 lessons); the LLM receives the full existing
consolidated set as input and can re-state anything worth keeping; blind supersede
means the transaction never depends on LLM-generated UUIDs, eliminating an entire
class of hallucination-driven data corruption. Every reflection produces a clean,
fully synthesised consolidated set.
Rejected: selective supersede with IDs (LLM hallucination risk on UUIDs, transaction
complexity, no architectural benefit at this scale).

---

---

# ADR-028 — remember() Batch Atomicity

A batch of memories submitted to `POST /v1/remember` executes in a single transaction:
all inserts + the reflection counter increment (by the count of episodic memories in
the batch) succeed or all roll back together. The `_episodic_counter` in-process is
incremented by the episodic count only after the platform returns 200.
Rejected: per-memory transactions with partial success (adds partial failure handling
complexity; if the platform is healthy enough to write one memory it should write all).

---

---

# ADR-031 — org_knowledge Dedup: Embedding Copy Only, Index Fields Always Recomputed

`index_questions` includes the section heading, which is document-specific context —
not derived from content alone. Copying it from a different document's section on a
hash match produces a misleading field regardless of how similar the content is.
spaCy recomputation costs ~5ms — there is no meaningful cost to always recomputing
index fields from the actual section heading and text. Only `embedding` (the expensive
async LiteLLM call) is copied on a hash match.
Rejected: copy all four fields (data integrity violation on index_questions heading).

---

---

# ADR-032 — org_knowledge Re-ingest: Dedup Before Delete

The naive delete + reinsert on re-ingest destroys existing embeddings then immediately
triggers the async worker to recompute them — wasted spend on unchanged content.
Checking all chunk hashes before deleting costs one cheap SELECT and eliminates
unnecessary embedding recomputation for unchanged documents entirely. For partially
changed documents, the per-section dedup (embedding copy on hash match) still applies
during the insert path.
Rejected: always delete + reinsert (unnecessary embedding spend on unchanged re-ingests).

---

---

# ADR-037 — org_knowledge: content_tsv Removed

`content_tsv` and its GIN index are removed from `org_knowledge`. The recall pipeline
uses `index_tsv` (BM25 on PageIndex reasoning artifacts) — `content_tsv` (BM25 on raw
section text) is not referenced in any current query path. A GIN index on a generated
tsvector column adds ~20-30% write overhead on every ingest INSERT. Retaining dead
indexes is not free. If raw-content BM25 is ever needed, add it back with a migration.

Rejected: retain for future use (dead indexes have real write cost, migrations are cheap).

---

# ADR-041 — Embedding Worker: Per-Row Error Isolation via embedding_attempts

The embedding worker processes rows per-row inside the batch loop rather than
batching all LiteLLM calls into a single request. A per-row LiteLLM failure
increments `embedding_attempts` on that row and continues to the next — the
rest of the batch is not abandoned.

Rows that reach `embedding_attempts = 3` are excluded from future worker cycles
via the `WHERE embedding IS NULL AND embedding_attempts < 3` filter. They remain
in Postgres with `embedding = NULL` — BM25 search still works on them, vector
search does not. A structured `embedding_failed` warning is emitted per exhausted
row for Alertmanager to surface.

This closes the infinite-retry loop where a single row with content that
persistently fails LiteLLM (token limit exceeded, malformed text, provider
rejection) blocks the entire embedding backlog for that table indefinitely.

`embedding_attempts SMALLINT NOT NULL DEFAULT 0` added to all three memory tables.
Partial indexes updated to `WHERE embedding IS NULL AND embedding_attempts < 3`.

Rejected: whole-batch transaction rollback on any per-row failure (one bad row
starves the entire backlog), silent skip with no counter (no way to detect or
alert on persistently failing rows), separate dead-letter table (unnecessary
schema complexity — the attempts counter on the row itself is sufficient).

---

# ADR-053 — _run_reflect() Failure: Structured Log Before Silent Swallow

Reflection failures in `mcp_bridge.py` are swallowed to prevent crashing the agent.
Without a structured log, failures accumulate invisibly — an agent that has stopped
reflecting for days produces no signal. The platform-side 500 is logged server-side
but cannot be correlated to a specific agent's reflection cadence without querying
`auth.agents.reflection_counter` directly.

Fix: emit `{event: reflection_failed, agent_id, tenant_id, error}` before the
except block swallows the exception. Alertmanager threshold: >3 failures for the
same `agent_id` within 1 hour — a single failure is noise, repeated failures
indicate a persistent outage.

Rejected: no log (silent failure accumulation, no operator signal), raise exception
(crashes the agent process — reflection is non-critical, agent must continue).

---

# ADR-054 — NER Entity Extraction at Write Time, Not Recall Time

Named entity extraction via spaCy `en_core_web_sm` runs at memory write time
(`POST /v1/remember`, `POST /v1/remember-org`) and stores entities in a JSONB column.
The alternative — extracting at recall time — would require running spaCy on every
candidate row during search, making entity filtering O(n) per query. Write-time
extraction is O(1) per write and makes the entity filter a cheap GIN-indexed JSONB
containment check at recall time.

spaCy `en_core_web_sm` is already loaded at platform startup (step 4 of the [platform startup sequence](../platform/02-api-contracts.md#platform-startup-sequence)) for
PageIndex extraction in the knowledge ingestion pipeline. The `extract_entities`
function is added to `qortia/knowledge.py` alongside `extract_index_fields` — same
model instance, no new dependency, no new startup cost.

Entity extraction is best-effort: if spaCy raises on malformed input, the platform
logs a warning and writes `entities=[]`. The write never fails due to extraction failure.
`org_knowledge` already has `index_entities` populated at ingest — no change needed there.

The `entities` filter on `POST /v1/recall` is additive with BM25/vector — it narrows
the candidate set before ranking. It is optional and does not change the recall contract
for callers that do not use it.

Rejected: recall-time extraction (O(n) per query, adds spaCy latency to the hot recall
path), LLM-based entity extraction (token cost, latency, non-deterministic, unnecessary
given spaCy is already in the stack).

**Note (commit `dcf1920`, updated `a4965dd`):** `extract_entities` routes Indic
languages (`hi`, `bn`, `ta`, `te`, `mr`) to `xx_ent_wiki_sm` (spaCy multilingual
model). `hi_core_news_sm` was originally listed but does not exist in spaCy model
releases — corrected in `a4965dd`. English and all other languages continue to use
`en_core_web_sm`. See ADR-081 for the full embedding model consolidation.

---

---

# ADR-055 — Dynamic Importance: recall_count + last_recalled_at, Fire-and-Forget Update

The static per-type importance scores (episodic=0.3 … lesson=0.95) are augmented with
a dynamic signal blending access frequency (log1p(recall_count)/10) and recency
(linear decay over 30 days, max +0.2). The RRF fusion already multiplies by importance
as a tiebreaker — replacing the static lookup with `dynamic_importance` improves ranking
without changing the pipeline architecture.

Access tracking runs as a fire-and-forget `asyncio.create_task` after the recall response
is assembled. It acquires its own connection from `main_pool` and issues one UPDATE per
table (hindsight_memories, org_memory, org_knowledge). Failure is non-fatal — logged as
a warning, never propagated to the caller. This ensures zero latency impact on the recall
response path.

`recall_count` uses SMALLINT (max 32767) — sufficient in practice, smaller storage
footprint than INTEGER on every row of the largest table. `last_recalled_at` is nullable
— NULL means never recalled, which correctly produces zero recency_boost.

`recall_count` and `last_recalled_at` are internal ranking signals only — never returned
in the `RecallResponse` schema. They are set as `PrivateAttr` on `RecallResult` and
populated by `_to_result` from the SELECT columns.

Boot context assembly (`GET /v1/context`) is unaffected — it uses fixed ORDER BY clauses,
not importance-ranked recall.

Rejected: synchronous access tracking on the recall path (adds DB write latency to every
recall), exposing recall_count in the response (internal signal, not useful to the LLM),
INTEGER instead of SMALLINT (unnecessary storage cost on high-row-count tables).

---

---

# ADR-056 — Thought Trace Preservation (Cognitive Persistence)

Modern reasoning models (e.g., Qwen3.6, DeepSeek-R1) generate internal chain-of-thought (CoT) traces, often wrapped in `<thought>` tags. While many platforms strip these to save tokens or hide "messy" logic, the Qwen3.6 series features "Thinking Preservation," which allows the model to leverage these traces across multi-turn conversations to maintain logic stability and eliminate redundant reasoning.

Decision:
1. **Preserve Traces:** The the platform platform (Qortia, Work Order state machine, and mcp_bridge) will treat `<thought>` blocks as first-class conversation data. They will NOT be stripped from the history sent back to the model in multi-turn loops.
2. **Persistence:** For models supporting "Thinking Preservation," the platform will prioritize maintaining the KV cache or passing the full CoT history to ensure the model "remembers" its previous reasoning steps.
3. **Transparency:** Thought traces remain "hidden" from the end-user UI in Mission Control by default (to keep the cockpit clean) but are stored in the `hindsight_memories` and `telemetry_events` tables for operator audit and debugging.

Consequences:
- **Pros:** Drastically improved coherence in long agent loops (50+ turns). Significant reduction in "logic drift" where an agent forgets its original plan.
- **Cons:** Increased context window usage per turn. Potential for "reasoning loops" if the model gets stuck in its own thought history (mitigated by reflection cycles).

Rejected: Stripping thoughts (destroys the primary benefit of reasoning models), storing thoughts in a separate table only (makes them inaccessible to the model's native context window).

---

# ADR-062 — recall() Output: Strip Internal Ranking Signals

**Status:** Accepted
**Date:** 2026-04-24

### Context

`mcp_bridge.py` was returning the full `RecallResponse` JSON to the LLM, including `importance`, `id`, `created_at`, and `scope` fields. The chief agent surfaced these to users ("Mental model (0.8 importance)") — treating platform internals as content.

### Decision

`_recall()` in `mcp_bridge.py` strips all fields except `type` and `content` before returning to the LLM:

```python
results = [
    {"type": r["type"], "content": r["content"]}
    for r in resp.json().get("results", [])
]
return json.dumps({"results": results})
```

`importance` is a ranking signal used by the platform pipeline (dynamic importance in RRF fusion). The platform already used it to decide what to return — the agent has no use for the raw value. `id` and `created_at` are internal identifiers. `scope` is pipeline metadata.

### Why importance is not redundant

`importance` is essential to the platform — it drives `dynamic_importance()` which boosts frequently-recalled and recently-recalled memories in the RRF fusion step. It is stored per row and updated by the embedding worker. It is never redundant. It is simply not agent-facing.

### Consequences

- LLM context is cleaner — no numeric signals that the LLM cannot act on
- Agents cannot accidentally surface internal platform metadata to users
- `RecallResult` model retains all fields server-side — no schema change needed
- `PrivateAttr` fields (`_recall_count`, `_last_recalled_at`, `_score`) remain internal as designed (ADR-055)

---

---

# ADR-073: Recall Evaluation Harness

## Status
Accepted — implemented, baseline established

## Context

As Qortia matures, changes to `recall.py` and `reflect.py` are becoming more
frequent. Unit tests verify functional correctness but not retrieval quality.
We needed a regression gate that:

1. Prevents quality regressions on known difficult queries
2. Quantifies the gain/loss of new features (e.g., entity boost, re-ranking)
3. Provides an objective basis for competitive comparison vs Mem0/Zep

## Decision

Three-layer evaluation system. Layer 1 (REH) is the primary regression gate —
runs on every PR to `recall.py`. Layers 2 and 3 run weekly on staging.

### Layer 1 — Retrieval Evaluation Harness (REH)

- **Scope:** `recall.py` in isolation
- **Mechanism:** Static JSON dataset with ground truth IDs and hard negatives
- **Metrics:** Recall@5, Recall@10, MRR, Semantic Drift gap
- **Run frequency:** Every PR affecting Qortia (smoke: 10 cases; full: 55 cases)
- **Scoring:** Deterministic — ground truth ID in result list or not. No LLM-as-judge.

### Layer 2 — Agentic Loop Benchmarking (ALB)

- **Scope:** End-to-end agent behaviour
- **Mechanism:** Sandbox agents with pre-seeded memories, semi-automated scoring
- **Metrics:** Memory Utilization, Hallucination Rate, Context Window Hygiene
- **Run frequency:** Weekly on staging

### Layer 3 — Infrastructure Benchmarking (PIB)

- **Scope:** Platform performance
- **Mechanism:** 1,000-fact corpus, latency/throughput/cost measurement
- **Metrics:** p99 latency, embedding throughput, reflection cost, HNSW overhead
- **Run frequency:** Weekly on staging

## Implementation Details

### Dataset format

Each case specifies:
- `setup.memories` — memories to seed (ground truth is one of these)
- `setup.hard_negatives` — semantically similar but wrong memories
- `setup.org_memories` — org-scoped memories (for org recall cases)
- `setup.knowledge` — knowledge corpus entries (for knowledge recall cases)
- `ground_truth_index` — index into the relevant setup collection
- `ground_truth_source` — `"memories"`, `"org_memories"`, or `"knowledge"`
- `query` — the recall request body
- `expected` — pass criteria including `must_contain_in_top_result`

### Seeding order invariant

Hard negatives are seeded before ground truth memories. This ensures `created_at`
ordering always favours the ground truth when BM25 scores tie. `_recall_episodic`
and `_recall_decisions` use `ORDER BY rank DESC, created_at DESC` — if hard
negatives were newer, they would win the tiebreaker.

### Knowledge ground truth resolution

Knowledge chunks are not known at seed time (ingest pipeline splits content).
Ground truth is resolved by fingerprinting: `split_into_sections(content)[0]["text"][:40]`
matched against returned result contents. This mirrors the ingest pipeline exactly.

### `'simple'` tsconfig constraint

All BM25 queries use `plainto_tsquery('simple', ...)`. No stemming. Dataset queries
must use exact tokens present in the content. `"removal"` does not match `"remove"`.

### 50-token floor

`split_into_sections()` discards sections with fewer than 50 tokens. Knowledge
content in the dataset must produce sections that clear this floor.

## Rejected Alternatives

**LLM-as-judge scoring:** Non-deterministic, expensive, circular. Rejected.

**Live production data:** PII risk, non-reproducible. Rejected.

**Substring matching instead of ID lookup:** Does not detect ranking failures —
the correct answer could be present but ranked 20th. Rejected.

## Baseline (commit `2fba526`, smoke dataset — 10 cases)

| Metric | Score | North Star | Floor |
|---|---|---|---|
| Recall@5 | 0.90 | > 0.85 | ≥ 0.80 |
| Recall@10 | 0.90 | > 0.95 | — |
| MRR | 0.80 | > 0.75 | ≥ 0.65 |
| Semantic Drift gap | 0.389 | > 0.15 | — |
| Regression gate | **PASS** | | |

Regression floors in `run_reh.py` are set to these values minus 5% tolerance.

## Production Baseline (full 55-case dataset)

Established after full dataset validation and 16k keyword boost fix (commit `cf65af7`).
Regression floors updated in `run_reh.py`.

| Metric | Score | North Star | Floor |
|---|---|---|---|
| Recall@5 | 1.000 | > 0.85 | ≥ 0.95 |
| Recall@10 | 1.000 | > 0.95 | — |
| MRR | 0.982 | > 0.75 | ≥ 0.86 |
| Semantic Drift gap | 0.384 | > 0.15 | — |
| Regression gate | **PASS** | | |

**Intermediate baseline (commit `76a1d1e`, 54/55):** Recall@5=0.982, MRR=0.897.
One failing case: reh-055 (`scope=all`, knowledge ground truth, paraphrased query).
Fixed by ADR-074 (keyword boost + `min_score` lowered to 0.30 for knowledge).

## Bugs Surfaced by the Harness

The following bugs were invisible to the existing test suite and were only
discovered by running the evaluation harness:

1. **BM25 config mismatch:** `content_tsv` used `'english'`, queries used `'simple'`.
   All BM25 hybrid-path searches silently returned zero rows. Fixed: V6 migration.

2. **asyncpg vector type:** All four vector SQL sites passed `list[float]` to asyncpg
   without the pgvector codec. Exceptions swallowed by `asyncio.gather`. Fixed:
   `str(embedding)` at all four call sites.

3. **Recency bias in episodic/decision recall:** `ORDER BY created_at DESC, rank DESC`
   let hard negatives win on recency when BM25 scores tied. Fixed: flipped to
   `ORDER BY rank DESC, created_at DESC` with CASE-based tier for the recency fallback.

4. **`'simple'` config and stemming:** Dataset queries used stemmed forms not present
   verbatim in content. Fixed: audited all queries against actual tsvector output.

5. **Knowledge 50-token floor:** Short knowledge content produced zero chunks.
   Fixed: expanded dataset content to clear the floor.

## Consequences

- **Positive:** Objective regression gate for all Qortia changes. Bugs 1–5 above
  were caught before reaching production.
- **Positive:** Reproducible baseline for competitive comparison vs Mem0/Zep.
- **Negative:** ~3 minutes added to CI for smoke eval on `recall.py` PRs.
- **Negative:** Dataset requires maintenance — queries must be validated against
  `'simple'` tsvector output when content changes.
- **Neutral:** `EVAL_MODE` infrastructure required. Seed endpoint returns 404 in
  non-eval environments.

---

# ADR-074: Knowledge Candidate Keyword Boost Before MMR

## Status
Accepted — implemented in commit `cf65af7`

## Context

The recall evaluation harness (ADR-073) surfaced one failing case in the full
55-case `recall_v1.json` dataset after Phase 1 shipped (commit `76a1d1e`):

**reh-055:** `scope=all`, `ground_truth_source=knowledge`
- Query: `"where work order notifications are written on the agent filesystem"`
- Ground truth: knowledge chunk about `wo_watcher.sh` writing to `/sandbox/qortia_inbox.ndjson`
- Result: `pass=false`, `mrr=0.0`

### Root Cause

The query is a paraphrase — it does not share exact BM25 tokens with the ground
truth chunk. The hybrid pipeline produces knowledge candidates via vector search
(BM25 returns nothing for this query against the knowledge corpus), but the raw
cosine score of the ground truth chunk is comparable to episodic memories about
work orders that were seeded as hard negatives.

The MMR step for knowledge candidates used `min_score=0.35`. The ground truth
chunk's raw cosine score was at or below this threshold after competing with
episodic memories in the fused result set, causing it to be excluded before MMR
selection.

The original 16k spec (see `docs/archive/implementation-plans/16k-recall-quality.md`)
proposed a full cross-encoder reranking step via LiteLLM's reranker endpoint.
This was evaluated and rejected for this iteration for the following reasons:

1. **No Cohere API key in the stack.** The LiteLLM config has no reranker model
   configured. Adding one requires a new external dependency and Vault secret.
2. **Latency budget.** A cross-encoder adds 200–500ms per recall call. The
   `balanced` profile target is < 300ms total. A reranker would blow this budget
   for every `scope=all` query, not just the failing case.
3. **Blast radius.** The failing case is a single paraphrased knowledge query.
   A cross-encoder affects all 55 eval cases. The risk of regression on passing
   cases outweighs the benefit of fixing one.
4. **Simpler fix available.** The root cause is that paraphrased queries produce
   lower raw cosine scores against knowledge chunks than against episodic memories.
   A lightweight token overlap signal applied before MMR directly addresses this
   without a new model dependency.

## Decision

Add `_keyword_boost(query, content) -> float` — a pure function that computes
normalised token overlap between the query and a knowledge candidate's content.
Apply it as a multiplicative score modifier on knowledge candidates only, before
the MMR step. Lower the MMR `min_score` for knowledge from `0.35` to `0.30`.

### Implementation

```python
def _keyword_boost(query: str, content: str) -> float:
    query_tokens = {t.lower() for t in query.split() if len(t) > 2}
    if not query_tokens:
        return 0.0
    content_lower = content.lower()
    matched = sum(1 for t in query_tokens if t in content_lower)
    return matched / len(query_tokens)
```

Applied in `_recall_hybrid()` before the `_mmr` call for knowledge candidates:

```python
for kc in knowledge_candidates:
    boost = _keyword_boost(body.query, kc.content)
    kc._score = kc._score * (1.0 + boost)
knowledge_results = _mmr(
    query_embedding=query_embedding,
    candidates=knowledge_candidates,
    min_score=0.30,
)
```

**Why multiplicative, not additive?** An additive boost would require calibrating
the boost magnitude against the raw cosine score range. A multiplicative boost
scales proportionally — a chunk with a 0.25 cosine score and 60% token overlap
becomes `0.25 × 1.6 = 0.40`, which clears the `min_score=0.30` threshold. A
chunk with 0.10 cosine score and 60% overlap becomes `0.16` — still excluded.
This preserves the semantic signal as the primary gate.

**Why `min_score=0.30` for knowledge only?** Paraphrased queries produce lower
raw cosine scores against knowledge chunks than against episodic memories, because
knowledge chunks are longer and more topically dense. The `0.35` floor was
calibrated for episodic/experiential memories. Knowledge candidates need a lower
floor to survive into MMR selection when the query is paraphrased.

**Why tokens > 2 chars?** Stop words and short prepositions (`in`, `on`, `at`,
`to`, `of`) are noise — they appear in almost every chunk and would inflate the
boost for irrelevant candidates. Filtering to tokens > 2 chars removes the most
common stop words without a stop-word list dependency.

**Scope: knowledge candidates only.** The boost is not applied to episodic or
org memory candidates. Those go through RRF fusion with `dynamic_importance` as
the scoring signal. Applying keyword boost there would double-count BM25 signal
that is already present in the RRF fused score.

## Rejected Alternatives

**Cross-encoder reranking (original 16k spec):** Requires a new external API
dependency (Cohere or similar), adds 200–500ms latency to every `scope=all`
recall, and affects all 55 eval cases. Deferred — remains the right approach
for a future `recall_profile=thorough` mode.

**Lower global MMR `min_score`:** Would increase noise in knowledge results for
all queries, not just paraphrased ones. The keyword boost is targeted — it only
lifts candidates that share tokens with the query.

**BM25 on knowledge content (not `index_tsv`):** Knowledge BM25 searches
`index_tsv` (the PageIndex reasoning artifacts), not raw `content`. The ground
truth chunk's `index_tsv` does not contain the paraphrased query tokens. Changing
the BM25 target would affect all knowledge queries and require a migration.

## Results

After implementation (commit `cf65af7`):

| Metric | Before (76a1d1e) | After (cf65af7) |
|---|---|---|
| Recall@5 | 0.982 (54/55) | 1.000 (55/55) |
| Recall@10 | 0.982 | 1.000 |
| MRR | 0.897 | 0.982 |
| Semantic Drift | 0.410 | — |
| reh-055 | FAIL | PASS |
| Regression gate | PASS | PASS |

All 239 unit tests pass. 6 new unit tests added for `_keyword_boost` and the
knowledge boost integration path.

## Consequences

- **Positive:** reh-055 fixed. Full 55-case dataset now passes at Recall@5=1.000.
- **Positive:** Zero new dependencies. Pure Python. No latency impact.
- **Positive:** `_keyword_boost` is a pure function — fully unit-testable without
  DB or LiteLLM.
- **Negative:** Token overlap is a weak signal for highly paraphrased queries
  where no surface tokens overlap. The cross-encoder approach (deferred) handles
  this case correctly.
- **Neutral:** `min_score=0.30` for knowledge is now a separate constant from the
  `0.35` default in `_mmr`. This is intentional — knowledge and memory candidates
  have different score distributions.

## Future Work

The cross-encoder reranking path (original 16k spec) remains the right long-term
solution for `recall_profile=thorough`. When a reranker model is available in the
LiteLLM config, implement `_cross_encoder_rerank()` as described in
`docs/archive/implementation-plans/16k-recall-quality.md` and gate it behind
`recall_profile=thorough`. The keyword boost can remain as a pre-MMR signal even
when the cross-encoder is present — it is cheap and complementary.

---

# ADR-075 — Incremental Entity Summaries: LLM Throttle and Recall Surface

**Status:** Accepted
**Date:** 2025-07-27
**Deciders:** Platform team
**Supersedes:** ADR-077 (planned, never created — 16f spec reference)

---

## Context

`qortia_entities` tracks which memories mention each named entity. As an agent
accumulates memories, an entity node can be linked to dozens of memories. Without
a summary, the only entity context available at recall time is the raw `entity_text`
(e.g. "OpenAI") — no narrative about what the agent knows about that entity.

The 16f spec (Graphiti §11 pattern) calls for a running summary per entity node,
updated as new memories are linked. This surfaces richer entity context during
recall without additional LLM calls at query time.

---

## Decision

### Summary column

Add `summary TEXT` (nullable) to `qortia_entities` via `V3__entity_summary.sql`.
NULL = not yet summarised. No DEFAULT — existing rows remain NULL until their
first new memory link after the migration.

### Update trigger: `_maybe_update_entity_summary`

Called inside `_populate_graph_batch` after each entity upsert. Logic:

| Link count | Action |
|---|---|
| 1 | Bootstrap: `summary = memory_content[:500]`. No LLM call. |
| 2 | No-op. |
| 3, 6, 9, … (count % 3 == 0) | LLM call: merge existing summary with new memory content. |
| All other counts | No-op. |

**Why count % 3 == 0 (not every link)?**
Entity nodes for common entities (e.g. "OpenAI") can accumulate many links quickly.
Calling the LLM on every link would create unbounded LLM cost proportional to entity
popularity. Every 3rd link is a practical throttle: a 30-link entity gets 10 LLM
summary updates — sufficient to keep the summary current without runaway cost.

**Why not count >= 3 only (update once)?**
A one-time update at count=3 would leave the summary stale as the entity accumulates
more memories. The periodic update pattern keeps the summary converging toward the
agent's full knowledge of the entity.

### LLM model

`anthropic/claude-3-haiku-20240307` — the cheapest capable model in the LiteLLM
config. Summary updates are low-stakes (non-critical path, failure is non-fatal).
Max 200 output tokens — summaries are 2-3 sentences.

### Failure handling

`_update_entity_summary` catches all exceptions, logs `entity_summary_update_failed`
at WARNING level, and returns the existing summary unchanged. The entity node is
never left in a broken state. The embedding worker continues processing other rows.

### Recall surface

After the entity boost query in `_recall_hybrid`, the `summary` column is fetched
alongside `linked_memory_ids`. The first non-null summary from matched entities is
attached to `RecallResult.entity_summary` on the top result. This gives the LLM
immediate entity context without a second round-trip.

`entity_summary` is an optional field on `RecallResult` (None when no entity match
or no summary yet). Agents that don't use it are unaffected.

---

## Alternatives Considered

### Update on every link
Rejected: unbounded LLM cost for popular entities. A 100-link entity would trigger
100 LLM calls over its lifetime — disproportionate to the value of marginal updates.

### Update only at count == 3 (once)
Rejected: summary becomes stale as the entity accumulates more memories. The agent's
knowledge of an entity evolves — the summary should too.

### Separate background task (not inline with graph population)
Rejected: adds scheduling complexity. The graph population batch already has the
memory content in scope. Doing the summary update inline avoids a second DB read.

### Embed the summary and use it for entity boost
Deferred: the entity embedding already captures the entity text. Embedding the summary
would require a separate embedding column and a second HNSW index. The value of
summary-based entity boost is unvalidated. Revisit if entity recall quality degrades
for entities with many diverse memories.

---

## Consequences

- `qortia_entities` gains a `summary TEXT` column (V3 migration, nullable, no DEFAULT)
- `_populate_graph_batch` makes one additional `fetchrow` + one `execute` per entity
  upsert (reads link count + summary, writes updated summary). This is a constant
  overhead per entity, not per memory.
- LLM cost: ~200 tokens per summary update, every 3rd link. For an agent with 100
  distinct entities each accumulating 10 links: ~333 summary updates × 200 tokens
  = ~66K tokens total over the agent's lifetime. Negligible.
- `RecallResult` gains `entity_summary: str | None` — backward-compatible (optional field).
- `audit_rls.py` unaffected — no new table, no RLS change.
- `lint-tenant-isolation.sh` unaffected — no new cross-tenant queries.

---

# ADR-076 — Platform Embed Key: Dedicated Vault Secret for validate_embedding_dimensions

**Status:** Accepted
**Date:** 2025-07-27

---

## Context

`validate_embedding_dimensions()` in `reflect.py` is called once at platform startup
to verify the embedding model returns the expected 1024-dimensional vectors (ADR-081:
bge-m3). It needs a LiteLLM key to make the test embedding call.

The original implementation used `settings.litellm_master_key` directly — the LiteLLM
master key read from Vault at bootstrap. This meant the master key remained accessible
via `settings` to any application code path for the entire lifetime of the process.

This is unrelated to ADR-059 (inference proxy). ADR-059 concerns the agent container
holding the tenant's LiteLLM virtual key. This ADR concerns the platform process
holding the LiteLLM master key longer than necessary.

---

## Decision

Provision a dedicated platform-level LiteLLM virtual key at `platform/litellm_embed_key`
in Vault. Use this key exclusively for `validate_embedding_dimensions()`.

`bootstrap_vault_secrets()` is extended to:
1. Read `litellm_master_key` from Vault (unchanged — needed to provision the embed key)
2. Call `_ensure_platform_embed_key(master_key)` — idempotent: loads the embed key from
   Vault if it already exists, otherwise calls LiteLLM `/key/generate` with the master
   key and writes the result to `platform/litellm_embed_key`
3. Clear `settings.litellm_master_key = ""` immediately after — the master key is no
   longer accessible to any application code path after startup completes

`validate_embedding_dimensions()` calls `get_platform_embed_key()` instead of
`settings.litellm_master_key`.

---

## Why a dedicated key rather than a tenant key

`validate_embedding_dimensions()` is a platform-level startup check — it is not
scoped to any tenant. Using a tenant key would require picking an arbitrary tenant,
which is wrong. The master key works but is over-privileged. A dedicated platform
virtual key with no budget cap and `metadata: {purpose: platform_embed}` is the
correct scope.

---

## Vault path

`platform/litellm_embed_key → {key: "sk-..."}`

This path is under `platform/` — readable only by the `qortia-platform` Vault role.
Agent policies do not grant access to `platform/*` paths.

---

## What this does NOT resolve

This does not resolve ADR-059. The LiteLLM master key is still present in the
container environment variable (`LITELLM_MASTER_KEY`) — it is read by
`bootstrap_vault_secrets()` at startup. Removing it from the environment entirely
requires infrastructure changes (docker-compose, Helm values) and is deferred.

The improvement is narrower: the master key is no longer accessible via `settings`
to application code after startup. A memory-inspection attack post-startup cannot
retrieve it from the settings object.

---

## Consequences

- `vault.py` gains `_platform_embed_key: str | None`, `_ensure_platform_embed_key()`,
  and `get_platform_embed_key()` — all within the `vault.py` boundary (Invariant #1)
- `reflect.py` no longer imports or uses `settings.litellm_master_key`
- `settings.litellm_master_key` is `""` for the entire application lifetime after
  `bootstrap_vault_secrets()` returns
- `_ensure_platform_embed_key` raises `SystemExit(1)` on LiteLLM failure — startup
  is aborted rather than proceeding with a broken embed key
- Idempotent: re-deploying the platform does not generate a new key on every startup

---

# ADR-077 — Cross-Memory Linking: Similarity Threshold, Scope, and Recall Integration

**Status:** Accepted
**Date:** 2025-07-27
**Part:** 16i

---

## Context

Part 16i implements the A-MEM / Zettelkasten pattern: at embedding time, identify
semantically similar memories and store bidirectional links. At recall time, traverse
those links to surface causal context — e.g. a lesson automatically surfaces the
decision that caused it.

The key decisions are: what similarity threshold to use, which memory tables to link,
how to integrate with the recall pipeline without displacing primary results, and how
to handle link cleanup on memory deletion.

---

## Decisions

### 1. Similarity threshold: 0.70 cosine

0.70 is the minimum cosine similarity for a link to be created. Below this, the
relationship is too weak to be reliably causal — it is more likely topical coincidence
than a meaningful connection. The entity boost in the recall pipeline already handles
topical proximity; cross-memory links are for structural relationships (cause → effect,
decision → lesson).

Rejected: 0.80 (too restrictive — few links would form in practice), 0.60 (too loose —
produces noise links between unrelated memories that share a domain).

### 2. Top-N links per memory: 3

At most 3 links are created per memory write. This bounds write amplification in the
embedding worker. The top-3 by cosine similarity are the most structurally relevant;
links beyond 3 have diminishing causal value.

### 3. Bidirectional storage (two rows per pair)

Each link is stored as two rows: `(source→target)` and `(target→source)`. This allows
the recall expansion query to use a simple `WHERE source_id = ANY(...)` without a UNION.
The `UNIQUE (source_id, target_id)` constraint prevents duplicates; `ON CONFLICT DO NOTHING`
makes the write idempotent.

### 4. Scope: `hindsight_memories` only

Cross-memory linking applies to private agent memories only. Org memory and knowledge
corpus linking is deferred to community detection (16m), which operates at the entity
graph level rather than the memory level.

### 5. No `agent_id` column on `memory_links`

Links are between memory UUIDs. Tenant isolation is enforced by the `tenant_id` column
and RLS policy. Agent isolation is inherited: `hindsight_memories` already enforces
agent-scoped RLS, so any memory UUID that appears in `memory_links` was already
validated against the agent's identity when it was written. Adding `agent_id` to
`memory_links` would be redundant and would complicate the bidirectional insert.

### 6. Recall expansion: top-5 results, max 2 linked memories per result

Expansion only touches the top-5 fused results to bound the number of DB queries.
At most 2 linked memories are appended per result to prevent result set explosion.
Linked memories are appended after the primary results — they do not displace them.
Memories already present in the fused set are not duplicated.

### 7. `linked_via` field on `RecallResult`

Linked memories carry `linked_via: str` set to the ID of the primary result that
surfaced them. This gives the LLM provenance — it can reason about why a memory
appeared and what its relationship to the primary result is.

### 8. Forget cleanup: atomic with the forget transaction

When a memory is deleted via `POST /v1/forget`, all `memory_links` rows where
`source_id = memory_id OR target_id = memory_id` are deleted in the same transaction.
This prevents dangling link rows that point to non-existent memories.

### 9. Embedding worker trigger: after successful embedding write

Link population runs inside `_embed_single_row()` immediately after the embedding is
written to `hindsight_memories`. This reuses the existing worker cycle without adding
a new background task. Failure is non-fatal — a warning is logged and the embedding
write is not rolled back.

---

## Consequences

- `memory_links` table added with RLS, indexed on both `source_id` and `target_id`.
- `audit_rls.py` updated to verify `memory_links` on every CI run.
- `RecallResult` gains `linked_via: str | None` field (null for non-linked results).
- Recall pipeline extended with `_expand_with_links()` call after RRF fusion.
- Embedding worker extended with `_find_similar_memories()` + `_upsert_memory_links()` after each `hindsight_memories` embedding write.
- `forget()` transaction extended with `memory_links` cleanup.
- No new background task, no new Vault path, no new LiteLLM call.

---

# ADR-078 — Temporal Fact Bounds: valid_from / valid_until

## Context

Graphiti review §2 identified point-in-time querying as the most unique capability
a memory system can offer without requiring a graph database. The question "what did
the agent believe to be true in early 2024?" is unanswerable with the current schema —
superseded memories are marked `is_consolidated = false` but carry no timestamp
recording when they were superseded.

## Decisions

**1. Two columns, not one.**
`valid_from TIMESTAMPTZ NOT NULL DEFAULT now()` and `valid_until TIMESTAMPTZ`.
`valid_until IS NULL` means currently valid. `valid_until IS NOT NULL` means
superseded at that timestamp. This is the standard bi-temporal pattern — a single
`superseded_at` column would work but is less expressive for range queries.

**2. `valid_from` defaults to row creation time.**
No extraction prompt change in this iteration. Temporal markers in episodic content
("We decided to use Redis in early 2024") are not parsed into `valid_from` — that
requires prompt engineering and LLM extraction, deferred to a follow-on. The column
default is correct for all memories written after this migration ships.

**3. `valid_until` is set atomically with `is_consolidated = false`.**
The supersede UPDATE in `reflect()` sets both columns in the same statement, inside
the same transaction. ADR-027 supersede-first ordering is fully preserved — the
UPDATE runs before any INSERT of new memories. No new crash-safety risk introduced.

**4. Default recall excludes superseded rows.**
`_temporal_filter_clause(as_of=None)` returns `AND valid_until IS NULL`. This is
the correct default — agents should not see facts they have already superseded.
Previously, superseded memories (`is_consolidated = false`) could still appear in
the hybrid recall pipeline because there was no `valid_until` filter. This is a
correctness fix, not just a new feature.

**5. `as_of` parameter scope is private memories only.**
`org_memory` and `org_knowledge` have no supersede semantics — processes and
knowledge chunks are upserted, not versioned. The temporal filter applies only to
`_bm25_private` and `_vector_private`. Type-routed strategies (decisions, lessons,
episodic, short_term) are not filtered by `valid_until` — decisions are
point-in-time records that are never superseded; short_term memories expire via
`expires_at`, not `valid_until`.

**6. No backfill of `valid_from` for existing rows.**
`DEFAULT now()` sets `valid_from` to the migration timestamp for all pre-existing
rows. Point-in-time queries are only meaningful for memories written after this
migration ships. This is acceptable — the alternative (backfilling `valid_from`
from `created_at`) requires a data migration that adds no value when there is no
production data.

## Rejected Alternatives

- **Single `superseded_at` column:** Simpler but requires a separate `is_current`
  boolean for the default filter. Two columns with NULL semantics is cleaner.
- **Prompt-extracted `valid_from` in this iteration:** Adds LLM cost and prompt
  complexity. The schema ships first; extraction is a follow-on enhancement.
- **Applying temporal filter to type-routed strategies:** Decisions are never
  superseded by reflection (they are `is_consolidated = false` by default and
  never set to true). Applying `valid_until IS NULL` to decisions would silently
  exclude all of them. Scope is private hybrid pipeline only.

---

# ADR-079 — Dual Embedding Model Routing: IndicSBERT for Indic Languages

## Status
Superseded — see Amendment below. The dual-model routing design was never deployed.
The stack migrated to BGE-M3 (single model, 1024-dim) before E3 shipped. ADR-081
will document the BGE-M3 decision formally.

## Context

E1 added a `lang` column to all memory tables (V3 migration). E2 added Indic NER routing (originally via Stanza; replaced by spaCy in commit `dcf1920` — see Amendment A below). E3 was intended to complete the multilingual stack by routing embedding generation to a purpose-built Indic model.

`reflect.py` at the time of this ADR hardcoded `EMBEDDING_MODEL = "text-embedding-3-small"` (served locally via `ollama/nomic-embed-text`). This model has reasonable multilingual coverage but was not purpose-trained on Indian languages. Semantic recall quality for Hindi, Tamil, Telugu, Bengali, Marathi, and other Indic languages is measurably lower than for English.

`ai4bharat/IndicSBERT` was the proposed replacement for Indic languages. It outputs 768-dim vectors — the same dimension the schema used at the time (`vector(768)` on all embedding columns). A model swap would require no migration.

**This design was not implemented.** Before E3 shipped, the stack migrated to BGE-M3 as the single embedding model for all languages. See Amendment B below for the full account.

## Decision (Original — Not Implemented)

Introduce per-memory embedding model routing based on the `lang` field:

```
lang = "en"          →  text-embedding-3-small  (unchanged)
lang in INDIC_LANGS  →  indic-embedding          (ai4bharat/IndicSBERT)
lang = other         →  text-embedding-3-small  (safe fallback)
```

`INDIC_LANGS` covers the 11 languages IndicSBERT was trained on:
`hi`, `ta`, `te`, `bn`, `kn`, `ml`, `mr`, `gu`, `pa`, `or`, `as`.

Both models output 768-dim vectors. The same `vector(768)` column stores both.

**This decision was superseded before implementation. See Amendment B.**

## Alternatives Rejected (at time of original decision)

- **Single multilingual model for all content** (`intfloat/multilingual-e5-large`, 1024-dim): requires a schema migration to change `vector(768)` to `vector(1024)` across 4 tables. Deferred.
- **Replace `text-embedding-3-small` globally with IndicSBERT**: IndicSBERT is weaker on English. Would degrade English recall quality.
- **OpenRouter for IndicSBERT**: not listed on OpenRouter. Self-hosted only.

---

## Amendment A — E2 NER: Stanza replaced by spaCy (commit `dcf1920`)

**Original E2 decision:** Stanza (`stanza>=1.10`) was added to `platform/pyproject.toml` to provide NER for 5 Indic languages (`hi`, `bn`, `ta`, `te`, `mr`) not covered by `en_core_web_sm`. Stanza was loaded lazily in `knowledge.py` via `_get_stanza_pipeline(lang)`.

**Problem:** Stanza carries PyTorch as a hard transitive dependency (~530 MB wheel). This bloated every platform image build regardless of whether Indic NER was exercised, and violated the principle of keeping the platform image lean.

**Replacement decision:** Stanza removed. Indic NER now routes to two spaCy models pinned at `3.8.0` (matching the existing `en_core_web_sm` pin):

| Language | Model |
|---|---|
| `hi` | `hi_core_news_sm-3.8.0` |
| `bn`, `ta`, `te`, `mr` | `xx_ent_wiki_sm-3.8.0` (multilingual) |

Both models are declared as direct URL dependencies in `pyproject.toml` — same pattern as `en_core_web_sm`. The `_get_stanza_pipeline` function and `_stanza_pipelines` dict in `knowledge.py` are replaced by `_get_indic_pipeline` and `_indic_pipelines` keyed by model name (so `bn`/`ta`/`te`/`mr` share the `xx_ent_wiki_sm` singleton). Entity labels (`PER`, `ORG`, `LOC`, `MISC`) and the call contract are unchanged — spaCy uses `ent.label_` directly, eliminating the `doc.sentences` traversal Stanza required.

**No behaviour change:** The NER label set and the 20-entity cap are identical. The `STANZA_NER_LANGS` constant is renamed `INDIC_NER_LANGS`; the covered language set is unchanged.

---

## Amendment B — E3 Superseded: BGE-M3 replaces dual-model routing

**What was discovered:** During eval runs (2026-05-02), live inspection of the DB
revealed the stack had already migrated to a single embedding model before E3 shipped:

```python
# platform/app/qortia/reflect.py — actual current state
EMBEDDING_MODEL = "bge-m3"
```

```yaml
# litellm.config.yaml — actual current state
- model_name: bge-m3
  litellm_params:
    model: ollama/bge-m3
    api_base: http://host.docker.internal:11434
```

Live dimension check confirmed: `vector_dims(embedding) = 1024` across all 118
embedded rows in `hindsight_memories`. The schema is `vector(1024)`, not `vector(768)`.
The dual-model routing code (`_embedding_model_for`, `INDIC_LANGS`, `EMBEDDING_MODEL_EN`,
`EMBEDDING_MODEL_INDIC`) does not exist in the codebase.

**Why BGE-M3 supersedes the dual-model design:**

| Property | text-embedding-3-small + IndicSBERT | BGE-M3 |
|---|---|---|
| Architecture | Two models, language-routed | Single model, all languages |
| Dimensions | 768 (both) | **1024** |
| Indic coverage | IndicSBERT: 11 languages purpose-built | Native multilingual, 100+ languages |
| English quality | text-embedding-3-small: strong | BGE-M3: MTEB top-tier as of 2025 |
| Hosting | Two services (LiteLLM + separate IndicSBERT) | Single ollama instance |
| Operational complexity | High — two models to validate, two failure modes | Low — one model, one endpoint |
| Cross-language recall | Not supported (two embedding spaces) | Supported (single embedding space) |
| MRL support | text-embedding-3-small: yes (256/512/1024/1536) | **Yes — 256/512/1024 dims** |
| Cost | Per-call API cost for text-embedding-3-small | Local inference, no per-call cost |

BGE-M3 (BAAI/BGE-M3, 566M params, F16 quantisation via ollama) eliminates the
fundamental limitation of the dual-model design: two separate embedding spaces in
the same column. With BGE-M3, a Hindi query can retrieve English memories and vice
versa — cross-language recall works because all content lives in the same 1024-dim
space.

**Schema impact:** The migration from `vector(768)` to `vector(1024)` has already
been applied. All existing rows use 1024-dim vectors. The `validate_embedding_dimensions`
check in `reflect.py` now asserts `actual == 1024`.

**Storage impact:** Each embedding is now `1024 × 4 = 4,096 bytes + 4 bytes overhead
= 4,100 bytes` per row (confirmed by live `pg_column_size` measurement). This is
33% larger than the 768-dim design. The trade-off is accepted: better recall quality,
simpler architecture, no per-call API cost.

**MRL opportunity:** BGE-M3 natively supports Matryoshka Representation Learning —
embeddings can be truncated to 256 or 512 dims at query time with minimal quality
loss (~2–3% on MTEB). This is documented as a future optimisation in
`docs/enhancements/qortia-memory-quality.md` (G5 — not yet numbered, to be added).

**Indic NER routing (Amendment A) is unaffected.** BGE-M3 handles embedding for
all languages. The spaCy Indic NER models (`hi_core_news_sm`, `xx_ent_wiki_sm`)
remain correct for entity extraction — NER and embedding are independent subsystems.

**ADR-081** will formally document the BGE-M3 adoption decision, the schema migration
from `vector(768)` to `vector(1024)`, and the MRL roadmap.

---

# ADR-080 — Org Memory RBAC: Two-Axis Access Control (Clearance + Division)

## Status
Accepted

## Context

Every agent in a tenant shares the same `org_memory` and `org_knowledge` tables. The
current access model is binary: tenant membership grants full read access. This is
correct for homogeneous fleets and incorrect for heterogeneous fleets where agents
serve different principals with different information entitlements.

Code review of `remember.py`, `recall.py`, `db.py`, and `provisioning.py` identified
nine concrete gaps (G1–G9) documented in `docs/enhancements/org-memory-rbac.md`.
The most critical: all agents read all handoffs in their boot context (G1), all agents
can recall all org memory regardless of sensitivity (G2–G3), and the proposed RLS
migration would silently zero all org reads if `tenant_transaction` is not updated
first (G4).

## Decision

Implement a tenant-configurable two-axis RBAC model enforced at the PostgreSQL RLS
layer. Both axes must pass for a row to be readable.

**Axis 1 — Clearance level (hierarchical, inclusive):** Tenants define ordered levels.
Higher order includes all lower. Platform seeds three defaults on tenant creation:
`external=1`, `internal=2`, `restricted=3`. Stored in `tenant_clearance_levels`.
Agent assignment in `auth.agents.clearance_level`. RLS compares integer orders via
`app.memory_clearance_order` session variable.

**Axis 2 — Division / audience (set membership):** Tenants define divisions. Memory
rows carry `audience TEXT[]`. Agent must be in the audience or audience must include
`all`. Platform seeds one default division (`all`). Stored in `tenant_divisions`.
Agent assignment in `auth.agents.division`. RLS checks via `app.agent_division`
session variable.

**Combined access rule (enforced by RLS):**
```
readable = agent.clearance_order >= memory.min_clearance_order
           AND (agent.division = ANY(memory.audience) OR 'all' = ANY(memory.audience))
```

**G4 safety guard:** The RLS policy uses `coalesce(nullif(current_setting(...), ''), '2')`
to default to order 2 (`internal`) when the session variable is unset. This prevents
silent zero-read regression during the rollout window when existing call sites have
not yet been updated to pass clearance params.

**Backward compatibility:** Default config seeds `external/internal/restricted` + `all`
division. All existing agents default to `internal` clearance and `all` division. All
existing memory rows default to `min_clearance='internal'`, `audience='{all}'`. Existing
behaviour is preserved exactly for tenants that do not customise their RBAC config.

## Alternatives Rejected

**Option A — Static three-level enum:** Hardcoded `external/internal/restricted` levels
with no division axis. Rejected because it cannot prevent intra-clearance leakage: two
`internal`-clearance agents in different business functions (engineering vs. sales) can
read each other's handoffs. The division axis closes this gap.

**Option B — Memory namespaces:** Many-to-many join table (`agent_namespaces`). Rejected
because the correlated subquery inside every RLS policy is evaluated per row scan with
no index benefit. Performance regression at scale. Option E achieves the same result
with a single column and a GIN index.

**Option C — Separate tables per audience:** Doubles schema surface area, breaks HNSW
vector index (cross-table UNION cannot use a single index), and creates synchronisation
problems for the weekly summary background task.

**Option D — Application-layer filtering only:** Bypassable by any future code path that
queries `org_memory` directly without going through the filter. The existing `get_context()`
query is exactly this pattern. Defence-in-depth requires the database to be the enforcement
layer.

## Consequences

**Schema changes (V4 migration):**
- New tables: `tenant_clearance_levels`, `tenant_divisions` (both RLS-enabled)
- `auth.agents`: `clearance_level TEXT DEFAULT 'internal'`, `division TEXT DEFAULT 'all'`
- `org_memory`: `min_clearance TEXT DEFAULT 'internal'`, `audience TEXT[] DEFAULT '{all}'`
- `org_knowledge`: same two columns
- `qortia_entities`: `max_clearance_order INTEGER DEFAULT 2` (G8 side-channel fix)
- `org_memory` RLS policy `tenant_read` replaced by `tenant_visibility_read`
- `org_knowledge` RLS policy `tenant_read` replaced by `tenant_visibility_read`
- `org_memory` org_chart rows backfilled to `min_clearance='external'` (G9 fix)

**Application changes:**
- `db.py`: `tenant_transaction` gains optional `memory_clearance_order: int | None` and
  `agent_division: str | None` params. Always sets both session variables (safe defaults
  prevent regression).
- All agent-authenticated Qortia endpoints: fetch clearance order + division from DB at
  request start, pass into `tenant_transaction`.
- `remember.py`: `forget()` gains `AND agent_id = $2` on the `hindsight_memories` SELECT
  (G5/G6 defence-in-depth). `RememberOrgRequest` gains `min_clearance` and `audience`
  fields with role-based defaults.
- `recall.py`: `_record_recall_access` passes clearance params (G7). Entity boost query
  filters `max_clearance_order <= clearance_order` (G8).
- `provisioning.py`: org_chart inserts hardcode `min_clearance='external'`, `audience='{all}'`
  (G9). `ProvisionAgentRequest` gains `clearance_level` and `division` fields.
- `knowledge.py`: `KnowledgeIngestRequest` gains `min_clearance` and `audience` fields.
- `audit_rls.py`: `tenant_clearance_levels` and `tenant_divisions` added to
  `EXPECTED_RLS_TABLES`; `tenant_read` replaced by `tenant_visibility_read` for
  `org_memory` and `org_knowledge`.

**Performance:** One JOIN query per Qortia request to resolve clearance order. Primary
key lookup on `auth.agents` + index lookup on `tenant_clearance_levels`. ~0.5ms. Not
cacheable in-process — clearance can change via tenant admin action without agent restart.

**Known remaining gaps:** LLM inference leakage across clearance boundaries via work
order content, prompt injection via visible content, and intra-clearance leakage for
tenants that do not configure divisions. These are architectural properties that cannot
be solved at the memory layer alone.

**Deferred:** `platform/app/vault.py` public function for writing `clearance_level` and
`division` to the agent's Vault path at provisioning time. Currently `auth.agents` is
the source of truth; agents do not yet read clearance from Vault at boot. Required
before the MCP bridge can include `X-Agent-Clearance` / `X-Agent-Division` headers
without a DB lookup on every request.

## Implementation

Shipped in commit `86ea33f` (or nearest). 24 files changed, 1095 insertions, 76 deletions.

Files changed:
- `platform/migrations/V4__org_memory_rbac.sql` — full schema migration
- `platform/app/db.py` — `tenant_transaction` signature extended
- `platform/app/qortia/remember.py` — `_fetch_agent_clearance`, G1/G5/G6 fixes
- `platform/app/qortia/recall.py` — G7/G8 fixes
- `platform/app/qortia/reflect.py` — clearance passed into both transactions
- `platform/app/qortia/knowledge.py` — `min_clearance`/`audience` on ingest
- `platform/app/qortia/models.py` — new fields on request models
- `platform/app/auth/provisioning.py` — G9 fix, chief lock, clearance at provision
- `platform/ci/audit_rls.py` — new tables and updated policy names
- `platform/tests/unit/test_rbac.py` — 19 new unit tests
- `platform/tests/unit/test_recall_pipeline.py` — mock updated
- `platform/tests/unit/test_incremental_reflect.py` — mock updated
- `docs/decisions/adrs/adr-080.md` — this file
- `docs/decisions/adr-log.md` — ADR-080 row added

Test gate at merge: 296/296 unit tests passing, 3 skipped.

---

# ADR-081 — Unified Embedding Model: BGE-M3 Replaces Dual-Model Routing

## Status
Accepted — supersedes ADR-079

## Context

ADR-079 introduced dual embedding model routing: `text-embedding-3-small` (via `ollama/nomic-embed-text`) for English and `indic-embedding` (via `ai4bharat/IndicSBERT`) for 11 Indic languages. This required:

- A separate embedding server on port 8001 serving IndicSBERT via `sentence-transformers`
- Per-memory `lang`-based routing in `reflect.py` (`_embedding_model_for`) and `recall.py`
- Two LiteLLM model aliases in `litellm.config.yaml`
- `INDIC_EMBED_REQUIRED` env var to control startup behaviour
- Graceful degradation logic when the Indic server was unavailable

In practice the IndicSBERT server was never containerised — it was an undocumented external dependency on port 8001. `ai4bharat/indic-bert` has no published GGUF format and cannot be run via ollama. The dual-model architecture added operational complexity without a viable local dev path.

`BAAI/bge-m3` is available in the ollama registry, supports 100+ languages including all 11 Indic languages covered by IndicSBERT, and benchmarks above both `nomic-embed-text` and `IndicSBERT` on multilingual retrieval tasks. It outputs 1024-dim vectors.

## Decision

Replace both embedding models with a single `BAAI/bge-m3` instance served via the host ollama process:

```
all languages  →  text-embedding-3-small  (LiteLLM alias → ollama/bge-m3)
```

- `EMBEDDING_MODEL_EN`, `EMBEDDING_MODEL_INDIC`, `INDIC_LANGS`, and `_embedding_model_for()` are removed from `reflect.py`
- A single `EMBEDDING_MODEL = "bge-m3"` constant — the LiteLLM alias is renamed to match
  the actual model name for clarity
- `indic-embedding` alias removed from `litellm.config.yaml`
- `recall.py` imports `EMBEDDING_MODEL` directly instead of `_embedding_model_for`
- `validate_embedding_dimensions()` checks a single model against 1024-dim, 60s timeout

## Schema Migration

All four `vector(768)` embedding columns updated to `vector(1024)` in `V1__initial_schema.sql` directly (no new migration required — nothing exists in production).

Affected tables: `hindsight_memories`, `org_memory`, `org_knowledge`, `qortia_entities`.

## Embedding Speed

`bge-m3` (570M params) is slower than `nomic-embed-text` (137M params) on CPU. This is acceptable because:

1. Embeddings are generated asynchronously by the embedding worker — not on the hot API path
2. The query embedding at recall time (~100-200ms on CPU) is the only synchronous cost
3. In staging/prod the model runs on GPU where latency is single-digit ms regardless of size

## Consequences

**Positive:**
- Single model, single server, single LiteLLM alias — no routing logic anywhere
- Full Indic language coverage (hi, ta, bn, te, kn, ml, mr, gu, pa, or, as) plus 90+ other languages
- No external IndicSBERT service dependency
- Better multilingual recall quality than either predecessor model
- `INDIC_EMBED_REQUIRED` env var and graceful degradation logic eliminated

**Negative / Risks:**
- 1024-dim vectors use ~33% more storage per row than 768-dim. Negligible at current scale.
- `bge-m3` is slower on CPU than `nomic-embed-text`. Acceptable given async embedding worker architecture.
- Cross-language recall (Hindi query → English memory) remains unsupported — same limitation as ADR-079. Symmetric routing is preserved: query and stored memories always use the same model.

## Alternatives Rejected

- **Keep dual-model routing, fix IndicSBERT deployment**: IndicSBERT has no GGUF format, requires a custom FastAPI wrapper, and is outperformed by bge-m3 on Indic benchmarks.
- **Use `paraphrase-multilingual-mpnet-base-v2` (768-dim, ollama)**: covers Hindi but not Tamil or Bengali — insufficient for v2 eval dataset.
- **Matryoshka truncation of bge-m3 to 768-dim**: ollama does not honour the `dimensions` parameter for bge-m3. Truncation is not available without a custom wrapper.

## Startup Validation & Warmup

`validate_embedding_dimensions()` makes a single embedding call through LiteLLM at
platform startup to confirm bge-m3 is reachable and returns 1024-dim vectors. If the
dimension does not match or the call fails, the platform refuses to start (`RuntimeError`).

**Timeout:** 60s (increased from the original 10s). bge-m3 (570M params) takes
~30-60s to load into memory on first call on CPU. The 10s timeout was calibrated for
`nomic-embed-text` (137M params) and is insufficient for bge-m3 cold load.

**Warmup pattern:** The warmup is handled by LiteLLM's `model_warmup: true` setting
in `litellm.config.yaml`. LiteLLM probes all configured models at startup, loading
bge-m3 into ollama memory before the platform's `validate_embedding_dimensions` call.

The platform never calls ollama directly — LiteLLM is the only gateway. This is an
invariant. An earlier implementation attempt added `warmup_embedding_model()` to
`reflect.py` which called ollama's `/api/embeddings` directly using a derived
`OLLAMA_URL` config setting. This was reverted: the platform has no business knowing
about ollama's URL. The correct fix is `model_warmup` at the LiteLLM layer.

## Linux Docker Networking

`host.docker.internal` does not resolve automatically on Linux Docker (unlike macOS/Windows).
The `litellm` service in `docker-compose.yml` requires:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

This maps `host.docker.internal` to the host machine's gateway IP where ollama runs.
Without this, LiteLLM cannot reach ollama and all embedding calls fail silently with
a connection error.

## spaCy NER Correction

`hi_core_news_sm` was listed in `_INDIC_MODEL` in `knowledge.py` and as a direct URL
dependency in `pyproject.toml`. This model does not exist in spaCy's model releases
(verified against the GitHub releases API — only `xx_ent_wiki_sm` and `xx_sent_ud_sm`
exist at 3.8.0). Hindi NER now routes to `xx_ent_wiki_sm` alongside `bn`, `ta`, `te`,
`mr`. The unit test in `test_qortia_models.py` that patched `hi_core_news_sm` was
updated to patch `xx_ent_wiki_sm`. This was discovered during the platform image build.

## Eval Baseline (commit `a4965dd`)

| Dataset | Cases | Recall@5 | Recall@10 | MRR | Semantic Drift | Gate |
|---|---|---|---|---|---|---|
| `recall_v1.json` | 55 | 1.000 | 1.000 | 0.942 | 0.452 | ✅ PASS |
| `recall_v2.json` | 20 | 0.900 | 0.900 | 0.775 | 0.436 | ❌ FAIL |

v2 regression gate fails on 2 cases: `reh-v2-003` (Hindi decision BM25) and
`reh-v2-012` (Bengali decision BM25). Both are superseded-decision hard negative
cases where the BM25 `simple` text search configuration does not tokenise
Devanagari/Bengali script well enough to rank the ground truth above the hard
negative. This is a BM25 tokenisation issue, not an embedding issue. Under
investigation — tracked separately from this ADR.

---

# ADR-096 — Multilingual NER Hardening: Canonical Label Set, Lang-Keyed Cache, and Startup Warm-Up

## Status
Accepted

## Context

The multilingual NER feature (E2, shipped in `dcf1920`) introduced `xx_ent_wiki_sm`
for Indic language routing but left seven gaps that were identified during a post-ship
review:

1. `extract_index_fields` always called `get_nlp()` (English) regardless of `lang` —
   knowledge chunks ingested in Hindi/Bengali/etc. got English NER applied silently.
2. Three divergent label sets existed across `extract_entities`, `extract_entities_with_types`,
   and `extract_index_fields`. The phantom `TECH` label (not produced by `en_core_web_sm`)
   appeared only in `extract_index_fields`. `WORK_OF_ART` was missing from `extract_index_fields`.
3. No `lang` normalisation at the API boundary — `"EN"`, `"en-US"`, `""`, `None` all
   reached the NER router as-is, causing silent fallthrough to English.
4. `_indic_pipelines` cache was keyed on model name (`"xx_ent_wiki_sm"`) rather than
   `lang`. Callers pass `lang`; the cache abstraction was inverted.
5. `_get_indic_pipeline` had no error boundary — an `OSError` on missing model propagated
   as an unhandled exception on the first Indic memory write, with no structured log.
6. `xx_ent_wiki_sm` was not declared in `platform/pyproject.toml` (it was only in
   `the agent runtime/requirements.txt`). The platform NER path would fail silently if the model
   was absent.
7. Unsupported languages (`fr`, `de`, `zh`, etc.) fell through to `en_core_web_sm`
   with no warning logged.

## Decision

### 1. Single canonical label set — `EN_ENTITY_LABELS`

Replace the three divergent label sets with one `frozenset` constant used by all
extraction functions:

```python
EN_ENTITY_LABELS = frozenset(
    {"ORG", "PERSON", "PRODUCT", "GPE", "NORP", "FAC", "WORK_OF_ART"}
)
```

`TECH` is removed — it is not a valid `en_core_web_sm` label and was dead code.
`WORK_OF_ART` is now present in all paths including `extract_index_fields`.

### 2. Cache keyed by `lang`, not model name

`_indic_pipelines: dict[str, Any]` is keyed on `lang` (e.g. `"hi"`, `"bn"`).
The model name lookup (`_INDIC_MODEL[lang]`) is an internal implementation detail
of `_get_indic_pipeline`. Two languages that share a model get separate cache entries
pointing to the same loaded object — spaCy pipelines are reentrant.

### 3. Error boundary and startup warm-up

`_get_indic_pipeline` catches `OSError`, emits `spacy_model_load_failed` structured
log, and re-raises. `load_spacy_model()` calls `_get_indic_pipeline("hi")` at startup
so both models fail fast together rather than the Indic path failing silently on the
first memory write.

### 4. `extract_index_fields` accepts `lang`

Signature changed to `extract_index_fields(heading, text, lang="en")`. Routes to
`_get_indic_pipeline` for Indic languages. Noun chunks are English-only (spaCy
`xx_ent_wiki_sm` does not produce them). Call site in `ingest_knowledge` passes
`lang=body.lang`.

### 5. `lang` normalisation at the Pydantic boundary

A shared `_normalise_lang(v)` helper normalises BCP-47 tags before they reach any
routing logic: `"EN"` → `"en"`, `"en-US"` → `"en"`, `""` → `"en"`, `None` → `"en"`.
Applied via `@field_validator("lang", mode="before")` on `MemoryItem`,
`RememberOrgRequest`, `KnowledgeIngestRequest`. `RecallRequest` preserves `None`
(means search all languages).

### 6. `xx_ent_wiki_sm` declared in `platform/pyproject.toml`

Pinned at `3.8.0` to match the `spacy>=3.8.7` floor, consistent with `en_core_web_sm`.

### 7. Unsupported lang warning

`_SUPPORTED_LANGS = INDIC_NER_LANGS | {"en"}`. Any lang outside this set logs
`ner_lang_unsupported` at WARNING level before falling back to English NER. No error
is raised — English NER is better than nothing.

## Consequences

- Entity extraction for knowledge ingestion is now consistent with episodic/org memory
  extraction — same label set, same routing logic.
- The `TECH` phantom label is gone. Any code that relied on it (none found) would have
  been silently producing no results anyway.
- Adding a new language requires: entry in `_INDIC_MODEL`, entry in `_SUPPORTED_LANGS`,
  model install in `pyproject.toml` and `the agent runtime/requirements.txt`, warm-up call in
  `load_spacy_model()` if the model is large.
- `RecallRequest.lang = None` semantics are unchanged — None means no lang filter.
  The normaliser preserves None for this model only.

## Known Limitations (Not Addressed Here)

- `xx_ent_wiki_sm` covers only `PER→PERSON`, `ORG→ORG`, `LOC→GPE` for Indic languages.
  `PRODUCT`, `NORP`, `FAC`, `WORK_OF_ART` have no Indic equivalents in this model.
  The entity graph is structurally thinner for Indic content. A richer multilingual
  model is a future enhancement.
- Languages outside `_SUPPORTED_LANGS` fall back to English NER. This is logged but
  not blocked. Cross-language entity quality is undefined for these cases.

---

# ADR-105 — Memory Quality: MRL + Dedup Strategy

**Status:** Accepted
**Date:** 2026-05-14
**Phase:** 3 (Observability + Memory Hardening)

---

## Context

BGE-M3 at 1024-dim is used for all memory types. Three quality problems were identified:

1. **Index degradation:** The HNSW index on `hindsight_memories` covered all rows including archived ones. Archived rows degrade graph navigation quality because the query planner filters `WHERE tier = 'active'` but the index still navigates through archived nodes.

2. **Redundant embeddings:** `short_term` memories are never vector-recalled — they are BM25-only (TTL-bounded, ephemeral). Embedding them wastes storage and index space.

3. **Duplicate memories:** Episodic memories have high dedup rates (~35% per MemGPT research). Two dedup mechanisms are needed: exact (same content within 24h) and semantic (cosine similarity ≥ threshold within 7 days).

4. **Content floor:** Empty or trivially short memories degrade recall quality and waste embedding compute.

---

## Decisions

### G1: Partial HNSW index on active rows only (V13 migration)

Replace the full HNSW index with a partial index `WHERE tier = 'active'`. The query planner auto-uses this for all recall queries (which already have `WHERE tier = 'active'`). Archived rows no longer participate in graph navigation.

### G2: Content length floor

- `MemoryItem.content`: minimum 5 words (raises `ValueError` at validation time)
- `RememberOrgRequest.content`: minimum 10 words (org memory is always a deliberate structured write)

### G3: Post-embed semantic dedup for episodic/experiential

After `_embed_single_row` writes a vector for an episodic or experiential memory, `_maybe_dedup_memory` queries for the nearest neighbour within the last 7 days for the same agent and type. If cosine similarity ≥ `DEDUP_SIMILARITY_THRESHOLD = 0.95`, the new memory is archived with `metadata.dedup_of = <neighbour_id>`.

**Threshold note:** 0.95 was calibrated for 768-dim models. BGE-M3 at 1024-dim may allow a lower threshold (0.92–0.93) without false positives. Production telemetry will inform a future reduction. This ADR will be amended when data is available.

Only `episodic` and `experiential` types are deduped. `decision`, `lesson`, and `mental_model` are deliberate records that may legitimately repeat.

### G4: Exact content hash dedup at write time

Before inserting a new episodic or experiential memory, `remember()` checks for an existing active row with the same `SHA-256(content)` within the last 24 hours for the same agent. If found, the existing ID is returned without a new insert. The `content_hash` column is added in V13.

### G5: Skip embedding for short_term

In `_embed_single_row`, short_term memories return early before the ollama call. No MRL truncation at this time — query-time cast to 256-dim for episodic is deferred until recall@5 is measured in production.

---

## Consequences

- Storage reduction: archived rows excluded from HNSW index
- Index quality improvement: graph navigation only traverses active rows
- Recall quality improvement: fewer near-duplicate results in top-k
- Embedding compute reduction: short_term memories never embedded
- Dedup threshold is provisional — will be amended with production data
- V13 migration adds `content_hash TEXT` column and replaces the HNSW index


---

# ADR-120 — Recall Reranking Architecture: Opt-In Profiles + Cross-Encoder via Infinity (Not Ollama)

**Status:** Accepted
**Date:** 2026-05-30
**Phase:** Post-Tenant-0 hardening (Advanced Qortia)
**Supersedes:** the implementation approach in `docs/qortia/enhancements/recall-cross-encoder-profiles.md` §2.1

---

## Context

`recall()` currently exposes a single reranking option: `RecallRequest.rerank: bool`.
When `True`, `_llm_rerank` (`recall_rerank.py`) makes a full LLM call (~500ms,
~$0.001/call) reading the agent's `domain_md` to pick a model, falling back to
`settings.rerank_model`. The enhancement doc proposed two additions: (1) a
cross-encoder rerank option and (2) `fast`/`balanced`/`thorough` recall profiles
with candidate over-fetch.

The enhancement doc's central premise — that BGE-Reranker-v2-M3 "runs on the same
infrastructure with no new operational dependency" by adding
`model: ollama/bge-reranker-v2-m3` to `litellm.config.yaml` and calling LiteLLM's
`/rerank` — was **verified false** during pre-implementation review:

1. **LiteLLM does not support ollama as a rerank provider.** Calling `/rerank` with
   an `ollama/*` model returns `Unsupported provider: ollama`
   (BerriAI/litellm#12187, open as of 2026). Our pin is `v1.83.7-stable`
   (`docker-compose.yml`). LiteLLM rerank providers are Cohere, Jina, Infinity,
   HuggingFace TEI, AWS Bedrock, Azure AI, Voyage — not ollama.
2. **Ollama's own `/api/rerank` is experimental.** It is a llama.cpp-derived
   addition (ollama/ollama#7219, #10467) not reliably present in stable releases;
   relying on it would float an unpinned, community-maintained capability —
   violating our exact-pin discipline.

The "no new infra" claim is therefore unachievable as designed. This ADR records
the corrected architecture so the feature is built on a true premise.

---

## Decisions

### D1: Reranking stays a clean seam; profiles are opt-in and default-preserving

`recall()` applies rerank at exactly one point (`recall.py` — `if body.rerank ...`).
Recall profiles (`fast`/`balanced`/`thorough`) are added as an **opt-in**
`RecallRequest.profile` field defaulting to `None`. When `profile is None`, the
pipeline executes byte-identically to today — no change to the default code path,
no risk to the competitive recall eval gates. Profiles only alter behaviour when
an agent explicitly requests one.

`rerank: bool` is widened to `Literal["llm", "cross_encoder"] | bool` with
`True → "llm"` for backward compatibility.

### D2: Cross-encoder is served by a self-hosted Infinity container, proxied by LiteLLM

BGE-Reranker-v2-M3 is served by an in-cluster **Infinity** container
(`michaelf34/infinity`, exact-pinned tag) exposing a Cohere/OpenAI-compatible
rerank endpoint. LiteLLM proxies it as an `infinity` rerank provider — a path
LiteLLM **does** support. `recall_rerank.py` calls LiteLLM `/rerank` exactly as
the enhancement doc's `_cross_encoder_rerank` sketch intended; only the LiteLLM
model entry changes from `ollama/...` to an Infinity-backed entry.

This is a **new operational dependency** (one container). The enhancement doc's
"no new infra" framing is retracted. Infinity is free, self-hosted, and runs the
same BGE-Reranker-v2-M3 weights — so the *model* choice stands; only the *serving*
mechanism changes.

### D3: Candidate over-fetch is gated behind the live eval harness

The valuable half of profiles — `candidate_multiplier` over-fetch — cannot be
implemented by a single seam edit: the per-method result limits are hardcoded
inside the six search functions (`_bm25_private`, `_vector_private`, `_bm25_org`,
`_vector_org`, `_bm25_knowledge`, `_vector_knowledge`). Threading a limit through
all of them changes candidate sets feeding RRF/MMR and therefore can move
Recall@5 / MRR. Over-fetch MUST NOT be merged without the eval-regression gates
passing on a live stack (Recall@5 ≥ 0.95, MRR ≥ 0.86). Until then, profiles ship
with `candidate_multiplier = 1` (current fetch counts) and only toggle
stage-enablement + rerank mode.

### D4: Cross-encoder input is tenant memory content — in-cluster serving only

A cross-encoder scores `(query, memory_content)` pairs. Memory content is
tenant-sensitive (`domain`/`soul`-adjacent). External rerank APIs (Cohere, Jina)
would egress tenant memory off-cluster — rejected on tenant-isolation grounds.
Self-hosted Infinity keeps all rerank traffic inside the cluster, consistent with
the tenant-isolation invariant.

---

## Consequences

- The feature splits into two independently-shippable slices:
  - **Recall profiles (stage-toggle only)** — no new dependency, eval-safe because
    default path is untouched; `thorough` routes to existing `_llm_rerank`.
  - **Cross-encoder + candidate over-fetch** — requires the Infinity container and
    live-stack eval verification (D2 + D3).
- A new Infinity container must be added to `docker-compose.yml` and the K8s
  deployment, exact-pinned, before cross-encoder can be enabled.
- `recall-cross-encoder-profiles.md` §2.1 is corrected to reference Infinity, not
  ollama, and the "no new operational dependency" line is removed.

---

## Rejected Alternatives

| Alternative | Why rejected |
|---|---|
| `ollama/bge-reranker-v2-m3` via LiteLLM `/rerank` | LiteLLM returns `Unsupported provider: ollama` (litellm#12187). Infeasible on our pin. |
| Ollama `/api/rerank` called directly from `recall_rerank.py` | Experimental, not in stable ollama; would float an unpinned capability against our exact-pin discipline. |
| Cohere / Jina hosted rerank API | Egresses tenant memory content off-cluster — violates tenant isolation. |
| Implement candidate over-fetch now | Touches 6 hardcoded search limits; moves Recall@5/MRR; cannot verify offline. Gated behind live eval harness. |
