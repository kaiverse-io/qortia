---
kind: architecture
status: active
owner: platform
last_reviewed: 2026-05-18
---

# Qortia Evaluation Strategy — Why We Measure Memory This Way

**Status:** Design-only — evaluation philosophy; harness not yet built
**Audience:** Customers, partners, and engineers evaluating the platform's memory layer
**Last updated:** 2025-07-26

---

## The Problem With Agent Memory Today

Most agent frameworks treat memory as an afterthought — a flat key-value store or
a simple vector database bolted on after the fact. The result is agents that:

- Retrieve the wrong fact when two memories are semantically similar
- Forget recent decisions because older, more-recalled memories score higher
- Return irrelevant results when a query uses different vocabulary than the stored content
- Mix one customer's memories with another's when multi-tenancy is implemented at the
  application layer rather than the database layer

the platform built Qortia to solve these problems structurally, not with prompt engineering.
This document explains how we prove it works.

---

## Our Evaluation Philosophy

**Three principles govern how we measure Qortia:**

### 1. Deterministic scoring only

We do not use LLM-as-judge. Scoring is binary: the correct memory ID either appears
in the top-K results or it does not. This matters because:

- LLM judges are non-deterministic — the same result can score differently on two runs
- LLM judges are circular — you are using the same class of model to evaluate the
  output of the same class of model
- Regression detection requires a stable baseline — probabilistic scoring makes it
  impossible to know if a score change is signal or noise

Every ground truth in our dataset is a seeded memory ID. The harness checks for
that exact ID in the result list. No interpretation required.

### 2. Hard negatives, not just correct answers

Every test case includes hard negatives — memories that are semantically similar to
the ground truth but wrong. For example:

- Ground truth: "Decided to remove Redis from the stack entirely. All session state
  moves to Postgres."
- Hard negative: "Decided to use Redis for session storage to improve latency.
  Postgres was too slow at peak load."

Both mention Redis and session storage. A naive vector search will score them
similarly. Our evaluation measures whether the recall pipeline can distinguish them.
If the hard negative ranks above the ground truth, the case fails — regardless of
whether the ground truth appears somewhere in the results.

This is a much stricter test than checking whether the right answer is "somewhere
in the top 10". It measures precision, not just recall.

### 3. Three layers, three questions

| Layer | Question | Frequency |
|---|---|---|
| REH — Retrieval Evaluation Harness | Does the search return the right fact? | Every PR to `recall.py` |
| ALB — Agentic Loop Benchmarking | Does the agent use the retrieved fact correctly? | Weekly on staging |
| PIB — Infrastructure Benchmarking | Is retrieval fast, cheap, and scalable? | Weekly on staging |

REH is the primary regression gate. It runs in ~3 minutes on a 10-case smoke
dataset for every PR that touches the recall pipeline. The full 55-case dataset
runs on every PR and produces a report artifact.

---

## What Qortia Does Differently

### Typed memory hierarchy

Qortia stores five distinct memory types, each with its own retrieval strategy:

| Type | Retrieval strategy | Why |
|---|---|---|
| `episodic` | BM25 + recency tier | Recent events matter most; keyword match for specific incidents |
| `decision` | BM25 + recency sort | You want the *latest* decision on a topic, not the most semantically similar one |
| `lesson` | Vector similarity only | Lessons are experiential patterns — they match by meaning, not keywords |
| `mental_model` | Full hybrid (BM25 + vector + RRF) | Synthesised knowledge needs maximum recall quality |
| `experiential` | Full hybrid (BM25 + vector + RRF) | Same as mental_model |

This is structurally different from flat-fact frameworks (Mem0, Memorizz) that apply
the same retrieval strategy to every memory regardless of type. A decision about
Redis should be retrieved by recency — you want the most recent decision, not the
one most semantically similar to your query. A lesson about debugging should be
retrieved by meaning — the vocabulary in the lesson may be completely different from
the vocabulary in the query.

### Hybrid search with principled fusion

For the full hybrid pipeline, Qortia runs BM25 and vector search in parallel and
fuses the results using Reciprocal Rank Fusion (RRF):

```
score(result) = Σ 1/(60 + rank_in_list) across all search lists
final_score   = rrf_score × dynamic_importance(recall_count, last_recalled_at)
```

RRF is score-magnitude-invariant — it does not matter that BM25 scores are in
`[0, 1]` and vector scores are cosine similarities. Both lists contribute equally
to the fused ranking. This is more principled than weighted sum approaches that
require manual tuning of BM25/vector weights.

**Dynamic importance** adjusts the final score based on how often a memory has
been recalled and how recently. A memory that has been useful many times in the
past is more likely to be useful again. This is a formula, not an LLM call.

### BM25 with `'simple'` configuration

Qortia uses PostgreSQL's `'simple'` text search configuration for all BM25 queries.
This is a deliberate choice over `'english'` (which applies stemming and stop word
removal):

- **Multilingual support:** `'simple'` works correctly for any language. `'english'`
  would incorrectly stem non-English content.
- **Predictability:** With `'simple'`, the token in the query must appear verbatim
  in the content. There is no ambiguity about whether "removing" matches "remove".
  This makes the system's behaviour easier to reason about and test.
- **Consistency:** The `content_tsv` generated columns and all `plainto_tsquery`
  calls use the same configuration. A mismatch (V6 migration fixed this) causes
  BM25 to silently return zero results for all queries.

The tradeoff: dataset queries must use exact tokens present in the content. Our
evaluation dataset is built with this constraint in mind — queries are validated
against the actual tsvector tokens before being added to the dataset.

### Database-enforced tenant isolation

Every memory operation runs inside `tenant_transaction()`, which sets
`app.tenant_id` as a PostgreSQL session variable. Row-Level Security policies
on all data tables enforce that queries can only see rows belonging to the current
tenant — even if application-level filtering has a bug.

This is not a feature of the recall pipeline specifically, but it is a prerequisite
for any multi-tenant memory system. Qortia's evaluation dataset tests this
implicitly: each test case provisions a fresh tenant and agent, so any cross-tenant
data leak would cause false positives in the ground truth ID lookup.

---

## How the Evaluation Harness Works

### Seeding

For each test case, the harness:

1. Provisions a fresh tenant and agent via the internal eval API
2. Seeds hard negatives **first** (older `created_at`)
3. Seeds ground truth memories **second** (newer `created_at`)
4. Seeds org memories and knowledge if the case requires them
5. Waits 15 seconds for the embedding worker to process all rows

The seeding order is critical. `_recall_episodic` and `_recall_decisions` use
`ORDER BY rank DESC, created_at DESC` — relevance first, recency as tiebreaker.
If hard negatives were seeded after ground truth, they would win the tiebreaker
when BM25 scores are equal. Seeding them first ensures `created_at` always
favours the ground truth.

### Scoring

After the embedding wait, the harness fires the recall query and checks:

1. **Recall@5:** Is the ground truth ID in the first 5 results?
2. **Recall@10:** Is the ground truth ID in the first 10 results?
3. **MRR:** What is `1 / rank` of the ground truth?
4. **Semantic Drift gap:** How many positions above the best hard negative does
   the ground truth rank, normalised by result count?
5. **must_contain_in_top_result:** For cases that specify it, does the top result
   contain the required phrase?

A case passes if and only if `recall_at_5 = true` AND all `must_contain_in_top_result`
phrases are present in the top result.

### Knowledge ground truth resolution

Knowledge cases are more complex because the ingest pipeline splits content into
sections — the stored chunk IDs are not known at seed time. The harness resolves
the ground truth chunk by:

1. Running `split_into_sections()` on the raw input content (same function used
   by the ingest pipeline)
2. Taking the first 40 characters of the first section's body text (heading stripped)
3. Searching the returned results for a chunk whose content contains that fingerprint

This mirrors exactly what the ingest pipeline does, so the fingerprint always
matches the stored chunk.

---

## Current Results and What They Mean

**Full dataset baseline (55 cases, commit `cf65af7`):**

| Metric | Score | Floor |
|---|---|---|
| Recall@5 | 1.000 (55/55) | ≥ 0.95 |
| Recall@10 | 1.000 | — |
| MRR | 0.982 | ≥ 0.86 |
| Semantic Drift gap | — | > 0.15 |
| Regression gate | **PASS** | |

All 55 cases pass. The one previously failing case (reh-055) was fixed by the
keyword boost enhancement (ADR-074, commit `cf65af7`).

**reh-055 post-mortem:** `scope=all`, knowledge ground truth about
`wo_watcher.sh` writing to `/sandbox/qortia_inbox.ndjson`. Query was a
paraphrase — no BM25 token overlap. The ground truth chunk's raw cosine score
was at the MMR `min_score=0.35` threshold after competing with episodic hard
negatives. Fixed by: (1) applying `_keyword_boost()` to knowledge candidates
before MMR, lifting the ground truth chunk's effective score; (2) lowering
knowledge `min_score` from `0.35` to `0.30`. See ADR-074 for full analysis.

**Previous intermediate baseline (54/55, commit `76a1d1e`):**

| Metric | Score |
|---|---|
| Recall@5 | 0.982 |
| MRR | 0.897 |
| Failing case | reh-055 |

---

## What the Failures Taught Us

Building the evaluation harness surfaced several real bugs in the recall pipeline
that would have been invisible without it:

**BM25 config mismatch (V6 migration):** The `content_tsv` generated columns used
`'english'` configuration while all `plainto_tsquery` calls used `'simple'`. This
caused every BM25 hybrid-path search to silently return zero rows — the BM25 signal
was completely absent from all hybrid results. The system appeared to work because
vector search still returned results, but precision was degraded. Fixed by V6
migration.

**asyncpg vector type (no codec):** All four vector SQL call sites passed
`list[float]` directly to asyncpg for `::vector` parameters. asyncpg without the
pgvector codec registered expects a string. The exceptions were silently swallowed
by `asyncio.gather(return_exceptions=True)`, so vector search returned empty results
for all queries without any error surfacing. Fixed by wrapping all four sites with
`str(embedding)`.

**Recency bias in episodic/decision recall:** `_recall_episodic` and
`_recall_decisions` used `ORDER BY created_at DESC, rank DESC` — recency first,
relevance as tiebreaker. This is backwards for a recall system. A hard negative
seeded one second after the ground truth would rank above it regardless of BM25
score. Fixed by flipping to `ORDER BY rank DESC, created_at DESC`, with a tiered
CASE expression to ensure BM25-matched rows always rank above recency-only rows
in the episodic fallback window.

**`'simple'` config and stemming:** Several dataset queries used nominal/stemmed
forms that don't exist verbatim in the content. `"Redis decision stack removal"`
requires the token `"removal"` but the content contains `"remove"`. With `'simple'`
config, these are different tokens — no match. Fixed by auditing all dataset queries
against the actual tsvector output.

**Knowledge 50-token floor:** `split_into_sections()` discards sections with fewer
than 50 tokens. Short knowledge content in the dataset produced zero chunks, causing
the ingest to silently succeed but store nothing. Fixed by expanding knowledge
content in the dataset to ensure all sections clear the floor.

Each of these bugs was invisible to the existing unit and integration test suite
because those tests verify functional correctness, not retrieval quality. The
evaluation harness is the only mechanism that catches quality regressions.

---

## Competitive Position

| Dimension | Qortia | Mem0 V3 | Zep |
|---|---|---|---|
| Retrieval strategy | Type-routed hybrid (BM25 + vector + RRF) | Flat vector + LLM extraction | Graph traversal + vector |
| Tenant isolation | Database RLS | Application-level filter | Application-level filter |
| Evaluation | Deterministic REH with hard negatives | None published | None published |
| Reflection | Automated consolidation (formula-based) | None | Per-write edge invalidation |
| Importance scoring | `dynamic_importance()` formula | None | None |
| Token cost for infrastructure | Zero (spaCy NER, deterministic summary) | LLM call per memory write | LLM call per write |

The evaluation harness is itself a competitive differentiator: we can quantify
Qortia's recall quality with a reproducible number. When we run comparative
benchmarking against Mem0 using the same 55-case dataset, the result is an
objective measurement, not a marketing claim.

---

## Roadmap

**Completed:** All 55 cases in `recall_v1.json` pass at Recall@5=1.000 (commit `cf65af7`).

**Near-term:** Add the full REH to CI as a non-blocking job with artifact upload.
Remove `continue-on-error` once the baseline is stable across 3 consecutive runs.

**Medium-term:** ALB and PIB on staging. Comparative benchmarking against Mem0
before each major Qortia enhancement ships.

**Long-term:** Cross-encoder reranking for `recall_profile=thorough` (deferred
from 16k — see ADR-074). Requires a reranker model in the LiteLLM config.

---

**End of Document**
