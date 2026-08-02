---
kind: enhancement
owner: platform
last_reviewed: 2026-05-30
status: post-mvp
---

# Qortia: Cross-Encoder Reranking and Recall Profiles

**Status:** Post-MVP — not yet implemented. Architecture decided in **ADR-120**
(`docs/decisions/qortia.md`); see "When to Implement" + correction note below.
**Scope:** `platform/app/qortia/recall.py`, `platform/app/qortia/recall_rerank.py`,
`platform/app/qortia/models.py`, `the agent runtime/mcp_bridge.py`, `litellm.config.yaml`,
`docker-compose.yml` (new Infinity container)
**ADR:** ADR-120 — Recall Reranking Architecture: Opt-In Profiles + Cross-Encoder
via Infinity (Not Ollama). Supersedes §2.1 below.
**Depends on:** Self-hosted Infinity rerank container (ADR-120 D2)

> **⚠️ Correction (2026-05-30, pre-implementation review).** The §2.1 design below
> assumed BGE-Reranker-v2-M3 could be served via `ollama/bge-reranker-v2-m3` through
> LiteLLM's `/rerank` with "no new operational dependency". **This is false on our
> stack.** LiteLLM `v1.83.7-stable` returns `Unsupported provider: ollama` for rerank
> (BerriAI/litellm#12187), and ollama's own `/api/rerank` is experimental/unpinned
> (ollama/ollama#7219). Per **ADR-120**, the cross-encoder must be served by a
> self-hosted **Infinity** container (LiteLLM-supported rerank provider, in-cluster,
> tenant-safe) — a real new dependency. The model choice (BGE-Reranker-v2-M3) stands;
> only the serving mechanism changes from ollama to Infinity. The recall-profiles half
> (§2.2) is independently shippable as **opt-in** (`profile=None` preserves today's
> path); candidate over-fetch is gated behind the live eval harness (ADR-120 D3).
**GitHub issue:** #74
**Research source:** `docs/research/zep-graphiti-review.md` §5 (6 reranker options,
search config recipes) and §11 (Actionable Patterns #4, #5, #6). Graphiti's
cross-encoder reranking, pre-built search recipes, and candidate over-fetch pattern.

---

## When to Implement (Trigger Conditions)

**Classification: Post-MVP.** The current LLM rerank (`_llm_rerank`) is adequate at
Tenant-0 scale — do **not** build this speculatively. The two halves have different
triggers:

**Recall profiles (opt-in stage-toggle, no cross-encoder) — lighter:**
- Implement when an agent role explicitly needs a non-default latency/quality
  trade-off (e.g. a latency-sensitive path wanting `fast`, or a research role
  wanting `thorough`). Ships without new infra because `profile=None` keeps the
  default path byte-identical.
- Candidate over-fetch within profiles stays **gated** (see below) even though the
  stage-toggle does not.

**Cross-encoder rerank + candidate over-fetch — heavier, fully gated:**
1. **Telemetry trigger** — recall logs show `_llm_rerank` is a measurable
   bottleneck: p95 rerank latency on a hot path is material, rerank LLM cost is
   material, or recall-quality issues trace to the domain-model coupling
   (`qortia_llm_rerank` events / #65 context).
2. **Hard prerequisite** — a self-hosted **Infinity** rerank container is
   provisioned (ADR-120 D2). Until that container exists, the cross-encoder path
   cannot run at all (LiteLLM cannot rerank via ollama — verified).
3. **Verification prerequisite** — the live recall eval harness is runnable and
   passes the gates (Recall@5 ≥ 0.95, MRR ≥ 0.86). Candidate over-fetch changes the
   candidate set feeding RRF/MMR and **must not** merge without this.

**Do not implement if:** rerank is not a measured bottleneck, or no Infinity
container + eval harness exist. Until then ADR-120 is decided but the work is
deferred.

**GitHub issue:** #74 (open, labelled `post-mvp`). Decision: ADR-120.

---

## 1. The Problem

The current recall pipeline has two limitations:

**Single rerank option.** `RecallRequest.rerank = True` triggers `_llm_rerank` —
a full LLM call that reads the agent's `domain_md` to select a model (tracked in
#65). This is expensive (~500ms, ~$0.001/call) and couples rerank quality to the
agent's operational model. A cross-encoder is cheaper, faster, and purpose-built
for relevance scoring.

**Single pipeline for all queries.** Every recall query runs the same code path
regardless of latency requirements. An agent doing a quick context check before
responding needs a different cost/quality trade-off than an agent doing deep
research synthesis. There is no way to express this without changing the code.

Graphiti's search config recipes (15+ pre-built configurations) and cross-encoder
reranking (OpenAI, BGE, Gemini reranker models) address both problems. The BGE
family is already used in the platform (BGE-M3 via ollama for embeddings).
BGE-Reranker-v2-M3 reuses the same model family but **cannot** reuse the ollama
serving path for reranking (see correction note + ADR-120) — it is served by a
self-hosted Infinity container instead.

---

## 2. Design

### 2.1 Cross-encoder reranking

Add `cross_encoder` as a new `rerank` option. `RecallRequest.rerank` changes from
`bool` to `Literal["llm", "cross_encoder"] | bool`:

```python
class RecallRequest(BaseModel):
    ...
    rerank: Literal["llm", "cross_encoder"] | bool = False
    # True maps to "llm" for backward compatibility
    # False = no reranking
    # "llm" = existing _llm_rerank (LLM call)
    # "cross_encoder" = new _cross_encoder_rerank (BGE-Reranker-v2-M3)
```

The cross-encoder runs after RRF fusion, scoring each candidate against the full
query text:

```python
async def _cross_encoder_rerank(
    query: str,
    candidates: list[RecallResult],
    litellm_key: str,
    model: str = "bge-reranker-v2-m3",
) -> list[RecallResult]:
    """
    Score each candidate against the query using a cross-encoder.
    BGE-Reranker-v2-M3 via local ollama — same infra as BGE-M3, no new dependency.
    Returns candidates sorted by cross-encoder score descending.
    """
    pairs = [{"query": query, "passage": r.content} for r in candidates]
    resp = await get_litellm_client().post(
        "/rerank",
        headers={"Authorization": f"Bearer {litellm_key}"},
        json={"model": model, "query": query, "documents": [r.content for r in candidates]},
        timeout=15.0,
    )
    resp.raise_for_status()
    scores = {r["index"]: r["relevance_score"] for r in resp.json()["results"]}
    for i, candidate in enumerate(candidates):
        candidate.score = scores.get(i, candidate.score)
    return sorted(candidates, key=lambda r: r.score, reverse=True)
```

**LiteLLM config addition** (`litellm.config.yaml`) — Infinity-backed, **not** ollama
(ADR-120 D2). Requires an Infinity container in `docker-compose.yml` serving
BGE-Reranker-v2-M3 (exact-pinned tag):

```yaml
model_list:
  - model_name: bge-reranker-v2-m3
    litellm_params:
      model: infinity/bge-reranker-v2-m3
      api_base: http://infinity:7997
```

```yaml
# docker-compose.yml (new service — exact-pinned)
infinity:
  image: michaelf34/infinity:<exact-tag>
  command: ["v2", "--model-id", "BAAI/bge-reranker-v2-m3", "--port", "7997"]
```

### 2.2 Recall profiles

`RecallRequest` gains an optional `profile` parameter:

```python
class RecallRequest(BaseModel):
    ...
    profile: Literal["fast", "balanced", "thorough"] | None = None
```

When `profile` is set, it configures the pipeline. Individual fields still override
profile defaults when explicitly set.

```python
@dataclass
class RecallConfig:
    vector: bool
    bm25: bool
    entity_boost: bool
    rerank: str | None
    top_k: int
    candidate_multiplier: int  # fetch N × top_k candidates before reranking


RECALL_PROFILES: dict[str, RecallConfig] = {
    "fast": RecallConfig(
        vector=True,
        bm25=True,
        entity_boost=False,
        rerank=None,
        top_k=5,
        candidate_multiplier=1,
    ),
    "balanced": RecallConfig(  # current default behaviour
        vector=True,
        bm25=True,
        entity_boost=True,
        rerank=None,
        top_k=5,
        candidate_multiplier=2,
    ),
    "thorough": RecallConfig(
        vector=True,
        bm25=True,
        entity_boost=True,
        rerank="cross_encoder",
        top_k=10,
        candidate_multiplier=3,
    ),
}
```

**Candidate over-fetch** (Graphiti pattern): `candidate_multiplier` causes the
pipeline to fetch `multiplier × top_k` candidates from each search method before
fusion and reranking, then truncate to `top_k` after. This ensures the reranker
has sufficient candidates for quality ranking. The `balanced` profile fetches 10
candidates and returns 5; `thorough` fetches 30 and returns 10.

### 2.3 MCP bridge

`recall` tool gains optional `profile` parameter:

```python
Tool(name="recall", ..., inputSchema={..., "properties": {
    ...
    "profile": {
        "type": "string",
        "enum": ["fast", "balanced", "thorough"],
        "description": "fast: low latency, balanced: default, thorough: cross-encoder reranking with wider candidate set",
    },
}})
```

---

## 3. Files Affected

| File | Change |
|---|---|
| `platform/app/qortia/recall.py` | `_cross_encoder_rerank`, `RecallConfig`, `RECALL_PROFILES`, profile routing, candidate over-fetch |
| `platform/app/qortia/models.py` | `RecallRequest.rerank` type change; `RecallRequest.profile` |
| `the agent runtime/mcp_bridge.py` | `profile` on `recall` tool schema; `rerank` type update |
| `litellm.config.yaml` | Add `bge-reranker-v2-m3` model entry (local ollama) |

---

## 4. Test Gates

| Gate | What to verify |
|---|---|
| Unit — `test_recall_pipeline.py` | `rerank=True` maps to `"llm"` (backward compat) |
| Unit — `test_recall_pipeline.py` | Profile config resolution: `"thorough"` sets `rerank="cross_encoder"`, `top_k=10`, `candidate_multiplier=3` |
| Unit — `test_recall_pipeline.py` | Individual field overrides profile default |
| Unit — `test_recall_pipeline.py` | Candidate over-fetch: `balanced` fetches 10, returns 5 |
| Integration | `recall` with `profile="thorough"` returns results; cross-encoder latency logged |
| Integration | `recall` with `rerank="cross_encoder"` returns results sorted by cross-encoder score |
| Eval regression | `thorough` profile Recall@5 ≥ `balanced` profile Recall@5 |
| Eval regression | `balanced` profile Recall@5 ≥ 0.95, MRR ≥ 0.86 (no regression from candidate over-fetch) |
| Platform unit tests | `cd platform && python3 -m pytest tests/unit/ -q` — 798/798 |
| Full stack health | `python3 scripts/local_agents.py` — 21/21 checks |

---

## 5. Known Constraints

**BGE-Reranker-v2-M3 availability.** The cross-encoder is served by the Infinity
container (ADR-120 D2), **not** ollama. Infinity downloads `BAAI/bge-reranker-v2-m3`
on first boot. The `validate_embedding_dimensions` startup check does not cover the
reranker — if the Infinity service is unreachable, `_cross_encoder_rerank` fails at
call time, not at startup. `_llm_rerank` remains the safe fallback.

**`rerank` type change is backward compatible.** `True` maps to `"llm"`, `False`
maps to no reranking. Existing agents that pass `rerank: true` continue to get LLM
reranking. No MCP bridge change required for existing agents.

**`thorough` profile latency.** Cross-encoder reranking adds ~200–500ms per recall
call (30 candidates × cross-encoder scoring). Agents using `thorough` should not
be in latency-sensitive paths. The profile description in the MCP tool schema
makes this explicit.

**LiteLLM rerank endpoint.** Verified (2026-05-30): LiteLLM `v1.83.7-stable`
exposes `/rerank` but supports it **only** for specific providers (Cohere, Jina,
Infinity, HF TEI, Bedrock, Azure, Voyage) — **not** ollama (returns
`Unsupported provider: ollama`, BerriAI/litellm#12187). Infinity is the chosen
provider precisely because it is LiteLLM-supported, self-hosted, and tenant-safe
(ADR-120 D2/D4).

**Files Affected (corrected).** The §3 table should add `docker-compose.yml` /
K8s manifests (new Infinity service) and `platform/app/qortia/recall_rerank.py`
(where `_cross_encoder_rerank` lives alongside `_llm_rerank`), not `recall.py`.
