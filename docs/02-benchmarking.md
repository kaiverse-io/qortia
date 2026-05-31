---
kind: architecture
status: active
owner: platform
last_reviewed: 2026-05-18
---

# Qortia — Memory Layer Benchmarking Guide

**Status:** Full eval stack implemented — 6 harnesses covering retrieval, temporal, extraction, longitudinal, infrastructure, and LongMemEval
**Scope:** Quantitative evaluation of the Qortia memory service vs. enterprise SoA
**Last updated:** 2026-05-31

---

## 1. The Benchmarking Philosophy

Memory is the most difficult agent component to evaluate because it is
**non-deterministic, high-cardinality, and context-dependent**.

We benchmark Qortia at three distinct layers to ensure that architectural
hardening (RLS, typed hierarchies, hybrid search) actually translates into
better agent performance:

1. **Retrieval Evaluation Harness (REH):** Does the search return the right fact?
2. **Agentic Loop Benchmarking (ALB):** Does the agent use the retrieved fact correctly?
3. **Infrastructure Benchmarking (PIB):** Is the retrieval fast, cheap, and scalable?

**Scoring is always deterministic.** No LLM-as-judge. Ground truth is a seeded
memory ID — either it appears in the top-K results or it does not. This is a hard
requirement: probabilistic scoring creates circular evaluation (the LLM judges the
LLM) and makes regression detection unreliable.

---

## 2. Layer 1: Retrieval Evaluation Harness (REH)

### 2.1 What It Measures

REH measures `recall.py` in isolation, bypassing the agent entirely. It seeds
memories via the internal eval API, fires a recall query, and checks whether the
ground truth memory ID appears in the top-K results.

**What it does NOT measure:**
- Whether the agent uses the retrieved memory correctly (that is ALB)
- Latency or cost (that is PIB)
- LLM reasoning quality

### 2.2 Core Metrics

| Metric | Definition | North Star | Regression Floor |
|---|---|---|---|
| **Recall@5** | Ground truth ID in top 5 results | > 85% | ≥ 80% |
| **Recall@10** | Ground truth ID in top 10 results | > 95% | — |
| **MRR** | Mean Reciprocal Rank — 1/rank of ground truth | > 0.75 | ≥ 0.65 |
| **Semantic Drift gap** | Rank gap between ground truth and best hard negative, normalised | > 0.15 | — |

Regression floors are enforced by `run_reh.py` — the script exits 1 if either
floor is breached. North star targets are aspirational; floors are the gate.

**Current baseline (smoke dataset, 10 cases, commit `2fba526`):**

| Metric | Score | Status |
|---|---|---|
| Recall@5 | 0.90 | ✓ above north star |
| Recall@10 | 0.90 | ✗ below north star |
| MRR | 0.80 | ✓ above north star |
| Semantic Drift gap | 0.389 | ✓ above north star |
| Regression gate | **PASS** | |

### 2.3 Dataset Structure

**Files:**
- `platform/evals/datasets/recall_v1.json` — 55 cases, full dataset
- `platform/evals/datasets/recall_smoke.json` — 10 cases, CI smoke subset

Each case has the structure:

```json
{
  "id": "reh-001",
  "description": "episodic temporal — named entity, BM25 dominant",
  "setup": {
    "memories": [...],
    "hard_negatives": [...],
    "org_memories": [...],
    "knowledge": [...]
  },
  "ground_truth_index": 0,
  "ground_truth_source": "memories",
  "query": {
    "query": "rate limiting AuthService",
    "scope": "private",
    "type": "episodic",
    "rerank": false,
    "entities": null
  },
  "expected": {
    "ground_truth_must_rank_above_hard_negatives": true,
    "must_contain_in_top_result": ["AuthService"],
    "min_results": 1
  }
}
```

**`ground_truth_source`** — which setup collection contains the ground truth:
- `"memories"` — `setup.memories[ground_truth_index]`
- `"org_memories"` — `setup.org_memories[ground_truth_index]`, resolved via ID map
- `"knowledge"` — `setup.knowledge[ground_truth_index]`, resolved via section fingerprint

**Hard negatives** are semantically similar memories that should NOT rank above
the ground truth. They are the primary test of recall precision. Every case has
at least one hard negative.

### 2.4 Dataset Coverage

55 cases across all recall paths:

| Category | Cases | Recall path tested |
|---|---|---|
| Episodic — temporal (BM25 + recency tier) | 5 | `_recall_episodic` |
| Episodic — BM25 fallback (paraphrased query) | 5 | `_recall_episodic` |
| Decision — BM25 + recency | 5 | `_recall_decisions` |
| Lesson — vector only | 5 | `_recall_lessons` |
| Mental model — full hybrid pipeline | 5 | Full hybrid |
| Org memory — BM25 | 5 | `_bm25_org` |
| Org memory — vector | 6 | `_vector_org` |
| Knowledge corpus — BM25 | 6 | `_bm25_knowledge` |
| Knowledge corpus — vector | 6 | `_vector_knowledge` |
| Entity filter | 5 | `entities` parameter |
| Cross-scope (`scope=all`) | 2 | Full hybrid, all scopes |

**RRF sensitivity cases** (from §2.5) are distributed within the above categories:
- ≥ 5 BM25-dominant cases (unique technical keywords)
- ≥ 5 semantic-dominant cases (paraphrased queries)

### 2.5 RRF Sensitivity Analysis

We specifically test the two extremes of the BM25/vector fusion:

**BM25-dominant cases** — queries with unique technical tokens that appear
verbatim in the ground truth content. Example: `"remove Redis from stack"` →
ground truth contains "remove Redis from the stack entirely". BM25 should
dominate because the tokens are exact matches.

**Semantic-dominant cases** — paraphrased queries where the query vocabulary
does not overlap with the content. Example: `"the thing that monitors db
performance"` → ground truth contains "database performance monitor showed
that the query planner was choosing a sequential scan". Vector search must
carry this case because BM25 returns zero matches.

**Critical constraint for BM25 cases:** The `'simple'` text search configuration
(used post-V6 migration) performs no stemming. Query tokens must appear verbatim
in the content. `"removal"` does not match `"remove"`. Dataset queries must use
exact tokens present in the ground truth content.

### 2.6 Seeding Order Invariant

Hard negatives are seeded **before** ground truth memories in `dataset_loader.py`.
This is intentional and must not be changed.

`_recall_episodic` and `_recall_decisions` use `ORDER BY rank DESC, created_at DESC`
— relevance first, recency as tiebreaker. If hard negatives were seeded after ground
truth, they would have a newer `created_at` and win the tiebreaker when BM25 scores
are equal. Seeding hard negatives first ensures `created_at` always favours the
ground truth when scores tie.

### 2.7 Knowledge Ground Truth Resolution

Knowledge cases cannot use memory IDs for ground truth because the ingest pipeline
splits content into sections via `split_into_sections()` — the stored chunk IDs are
not known at seed time.

Resolution strategy: after recall, find the ground truth chunk by matching the first
40 characters of the first section's body text (heading stripped) against returned
result contents. This is computed from `split_into_sections(ground_truth_content)[0]["text"][:40]`.

**Why not source_path?** A single source_path can produce multiple chunks. The
ground truth is always the first section (`chunk_index=0`), so the fingerprint
approach is more precise than matching by path alone.

**50-token floor:** `split_into_sections()` discards sections with fewer than 50
tokens. Knowledge content in the dataset must be long enough that the ground truth
section clears this floor. Both sections of each knowledge case in the dataset
have been verified to produce ≥ 50 tokens.

### 2.8 Running the Harness

```bash
# Prerequisites: stack running with EVAL_MODE=true
cd platform

# Smoke eval (10 cases, ~3 minutes)
EVAL_MODE=true PYTHONPATH=. python3 evals/run_reh.py evals/datasets/recall_smoke.json

# Full eval (55 cases, ~15 minutes)
EVAL_MODE=true PYTHONPATH=. python3 evals/run_reh.py evals/datasets/recall_v1.json

# Report written to evals/results/reh_latest.json
```

**EVAL_MODE** must be `true` — the eval seed endpoint returns 404 otherwise.
Never set `EVAL_MODE=true` in staging or production.

---

## 3. Layer 2: Agentic Loop Benchmarking (ALB)

ALB measures whether Qortia serves the right memories for three gold-standard
agent scenarios. It does not require a live agent container — it evaluates the
memory layer's contract (recall quality and reflection output), leaving agent
reasoning scores for human annotation in the output report. Runs manually or
weekly on staging.

### 3.1 Gold Standard Tasks

**Task A — Temporal recency:**
Pre-seed two conflicting episodic memories (Monday: red, Tuesday: blue).
Query: "What is the current button color?"
Expected: Agent cites the Tuesday memory (blue). Tests recency ordering.

**Task B — Reflection consolidation:**
Pre-seed 10+ episodic memories about a user's coding style. Trigger reflection.
Query: "Summarise everything we've learned about the user's coding style."
Expected: Agent cites consolidated `mental_model` or `lesson` memories, not raw
episodic. Tests that reflection produces usable consolidated memories.

**Task C — Cross-memory linking:**
Pre-seed org_memory handoff with repo owner. Pre-seed weekly_summary referencing
the same project. Query: "Find the repo owner of the project in the weekly summary."
Expected: Agent traverses from weekly_summary → handoff to find the answer.
Tests multi-hop recall across memory scopes.

### 3.2 Cognitive Metrics

| Metric | Definition |
|---|---|
| **Memory Utilization** | % of reasoning steps where agent correctly cited a Qortia memory |
| **Hallucination Rate** | % of steps where agent claimed a memory not present in Qortia |
| **Context Window Hygiene** | Ratio of relevant vs filler memories in the final LLM prompt |

ALB scoring is semi-automated:

- **Auto-scored (deterministic):** Task A temporal ordering, Task B reflection
  promotion (consolidated type present in results), Task C scope coverage.
- **Human-scored (annotated in report):** Memory Utilization and Hallucination
  Rate — filled in after running a real agent against the same seeded scenarios.

Full LLM-as-judge automation is deferred — violates the determinism principle.

### 3.3 Running the Harness

```bash
# Prerequisites: full stack running (just up), EVAL_MODE=true, LiteLLM reachable
cd platform

EVAL_MODE=true python3 evals/run_alb.py

# Report written to evals/results/alb_latest.json
# Annotate 'memory_utilization' and 'hallucination_rate' fields after
# running a real agent against the seeded scenarios.
```

**Task B** makes a real LLM inference call via `/v1/reflect`. LiteLLM gateway
must be reachable. Budget: ~1 reflection call per run (~$0.002 at claude-haiku rates).

---

## 4. Layer 3: Infrastructure Benchmarking (PIB)

PIB measures the operational cost of Qortia. Runs weekly on staging against a
1,000-fact corpus.

### 4.1 Targets

| Metric | Target | Measurement |
|---|---|---|
| Recall p99 latency | < 400ms | `POST /v1/recall` end-to-end including embedding |
| Embedding throughput | > 100 memories/s sustained | Embedding worker batch rate |
| Reflection duration | < 10s per cycle | `POST /v1/reflect` wall time |
| Cost per reflection | < $0.02 | `agent_cost_ledger` sum for one reflect call |
| HNSW index overhead | < 2.5× raw data | `pg_relation_size` on HNSW vs table data |

---

## 5. Full Eval Stack — Coverage Map

Six harnesses covering all memory evaluation dimensions:

| Harness | Script | Datasets | What it tests | Run frequency |
|---|---|---|---|---|
| **REH** | `run_reh.py` | `recall_v1.json` (55 cases), `recall_smoke.json` (10) | Retrieval accuracy, hard negatives, all types and scopes, token efficiency | Every PR touching recall.py |
| **TEH** | `run_temporal_eval.py` | `recall_v2_temporal.json` (10), `recall_v2_supersede.json` (8) | Bi-temporal conflict resolution (ADR-078), selective forgetting (ADR-027) — expired facts must not surface | Weekly on staging |
| **ALB** | `run_alb.py` | Inline (3 tasks) | Temporal ordering, reflection consolidation surfaces, cross-scope recall | Weekly on staging |
| **EQE** | `run_extraction_eval.py` | Inline (10 cases) | Extraction precision/recall, noise rejection, type accuracy | Weekly on staging |
| **LEH** | `run_longitudinal_eval.py` | Inline (3 scenarios) | Multi-session reflection learning, rank improvement, context hygiene | Monthly |
| **PIB** | `run_pib.py` | Synthetic (50–1000 corpus) | p50/p95/p99 latency, embedding throughput, HNSW overhead | Weekly on staging |
| **LongMemEval** | `run_longmemeval.py` | Public dataset (500 cases) | Cross-session synthesis, temporal reasoning, knowledge updates vs. enterprise benchmarks | Before each major Qortia release |

**Regression gates (CI):** REH smoke (10 cases) on every PR to recall.py. All other harnesses run on staging weekly.

### Running the full stack

```bash
cd platform

# Layer 1 — Retrieval (55 cases, ~15 min)
EVAL_MODE=true python3 evals/run_reh.py evals/datasets/recall_v1.json

# Layer 1b — Temporal (18 cases, ~8 min)
EVAL_MODE=true python3 evals/run_temporal_eval.py

# Layer 2 — Agentic Loop (3 tasks, ~3 min)
EVAL_MODE=true python3 evals/run_alb.py

# Layer 2b — Extraction Quality (10 cases, ~5 min)
EVAL_MODE=true python3 evals/run_extraction_eval.py

# Layer 2c — Longitudinal Learning (3 scenarios, ~5 min)
EVAL_MODE=true python3 evals/run_longitudinal_eval.py

# Layer 3 — Infrastructure (50-corpus, ~5 min)
EVAL_MODE=true python3 evals/run_pib.py http://localhost:8080 <tenant_id> <agent_id>

# Layer 4 — LongMemEval (download once, then run)
python3 evals/run_longmemeval.py --download
EVAL_MODE=true python3 evals/run_longmemeval.py --max-cases 100
```

### SoA comparison targets

| Benchmark | Mem0 published | Zep published | Qortia current | Qortia target |
|---|---|---|---|---|
| LongMemEval Recall@5 | ~72% | ~69% | TBD (run `run_longmemeval.py`) | ≥75% |
| DMR accuracy | — | 94.8% | not yet run | ≥90% |
| REH Recall@5 | — | — | **1.000** (55/55) | 1.000 |
| REH MRR | — | — | **0.942** | ≥0.90 |
| p99 latency | <400ms | — | **415ms** (cold-start) | <400ms warm |
| Token efficiency | <7k/query | — | ~200 (estimated) | <2k/query |

---

## 6. Comparative Benchmarking vs. Mem0 and Zep

Qortia is benchmarked against Mem0 and Zep before each major release:

1. Ingest the same 1,000 facts into Qortia and Mem0 (local install).
2. Fire 50 standardised queries from `recall_v1.json`.
3. Compare Recall@10 and p99 latency.
4. Run `run_longmemeval.py` against all three systems with the same 100-case subset.
5. If Qortia Recall@10 < Mem0 Recall@10 → "L1 Gap" — open a GitHub issue
   with `category:qortia` label and the specific failing query categories.

This runs manually before each major Qortia enhancement ships.

---

## 6. Baseline Establishment Protocol

After the full 55-case dataset is validated:

1. Run `EVAL_MODE=true PYTHONPATH=. python3 evals/run_reh.py` against current codebase.
2. Record all four metric scores in ADR-073.
3. Set regression floors in `run_reh.py` to measured values minus 5% tolerance.
4. Remove `continue-on-error: true` from the full REH CI job.
5. Run PIB and record baseline latency/cost/storage numbers in ADR-073.
6. All Phase 1+ PRs must maintain or improve all four REH metrics.

---

**End of Document**
