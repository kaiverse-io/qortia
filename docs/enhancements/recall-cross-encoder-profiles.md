---
kind: enhancement
owner: platform
last_reviewed: 2026-05-18
status: post-mvp
---

# Qortia: Cross-Encoder Reranking and Recall Profiles

**Status:** Open — not yet implemented
**Scope:** `platform/app/qortia/recall.py`, `platform/app/qortia/models.py`,
`the agent runtime/mcp_bridge.py`, `litellm.config.yaml`
**ADR required:** Yes — new model dependency (BGE-Reranker-v2-M3), `rerank`
type change (bool → union), new `profile` API parameter
**Depends on:** None
**GitHub issue:** #74
**Research source:** `docs/research/zep-graphiti-review.md` §5 (6 reranker options,
search config recipes) and §11 (Actionable Patterns #4, #5, #6). Graphiti's
cross-encoder reranking, pre-built search recipes, and candidate over-fetch pattern.

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
family is already deployed in the platform (BGE-M3 via ollama) — BGE-Reranker-v2-M3
runs on the same infrastructure with no new operational dependency.

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

**LiteLLM config addition** (`litellm.config.yaml`):

```yaml
model_list:
  - model_name: bge-reranker-v2-m3
    litellm_params:
      model: ollama/bge-reranker-v2-m3
      api_base: http://host.docker.internal:11434
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
        vector=True, bm25=True, entity_boost=False,
        rerank=None, top_k=5, candidate_multiplier=1,
    ),
    "balanced": RecallConfig(  # current default behaviour
        vector=True, bm25=True, entity_boost=True,
        rerank=None, top_k=5, candidate_multiplier=2,
    ),
    "thorough": RecallConfig(
        vector=True, bm25=True, entity_boost=True,
        rerank="cross_encoder", top_k=10, candidate_multiplier=3,
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
| Platform unit tests | `cd platform && python3 -m pytest tests/unit/ -q` — 296/296 |
| Full stack health | `python3 scripts/local_agents.py` — 21/21 checks |

---

## 5. Known Constraints

**BGE-Reranker-v2-M3 availability.** The cross-encoder requires the model to be
pulled in ollama before use. Add to the local dev setup: `ollama pull bge-reranker-v2-m3`.
The `validate_embedding_dimensions` startup check does not cover the reranker —
a missing reranker model causes `_cross_encoder_rerank` to fail at call time, not
at startup. The ADR must document this.

**`rerank` type change is backward compatible.** `True` maps to `"llm"`, `False`
maps to no reranking. Existing agents that pass `rerank: true` continue to get LLM
reranking. No MCP bridge change required for existing agents.

**`thorough` profile latency.** Cross-encoder reranking adds ~200–500ms per recall
call (30 candidates × cross-encoder scoring). Agents using `thorough` should not
be in latency-sensitive paths. The profile description in the MCP tool schema
makes this explicit.

**LiteLLM rerank endpoint.** LiteLLM's `/rerank` endpoint is available from
v1.40+. Confirm the pinned LiteLLM version (`v1.83.7-stable`) supports it before
implementation.
