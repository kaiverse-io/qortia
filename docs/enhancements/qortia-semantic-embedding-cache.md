---
kind: enhancement
owner: platform
last_reviewed: 2026-05-18
status: post-mvp
---

# Qortia: Semantic Embedding Cache for Recall Pipeline

**Status:** Open — ADR required before implementation
**Scope:** `platform/app/qortia/recall.py`, `platform/app/config.py`
**ADR required:** Yes — introduces a new caching layer with eviction semantics and
a cosine similarity threshold that must be documented and validated
**Depends on:** None
**Research source:** "Context Architecture for Production AI Agents" (Redis, 2026) —
Pillar 4 "Accelerate": "Semantic caching allows agents to short-circuit expensive
LLM calls when the query intent is semantically equivalent to a prior result.
For workloads with high query repetition, this can reduce inference costs
substantially." Also: "Managing Memory for AI Agents" (O'Reilly/Redis, 2026) —
Ch. 1, "semantic caching: frequently retrieved information gets prioritised."

---

## 1. The Problem

Every call to `POST /v1/recall` that uses the full hybrid pipeline invokes
`_embed_query`, which makes an HTTP call to LiteLLM → BGE-M3 (local ollama) to
produce a 1024-dim embedding for the query. This call is on the critical path:
the recall response cannot be returned until the embedding is available.

At current scale this is acceptable (~5–10ms on GPU, ~50–100ms on CPU). At
production scale — many agents, high query volume, repeated queries — this becomes
a latency and resource bottleneck:

- **Repeated queries.** Agents frequently recall with semantically identical or
  near-identical queries across sessions (e.g. "what are my current priorities",
  "what did I decide about X", "what is the org chart"). Each call re-embeds the
  same query from scratch.

- **Embedding worker contention.** The embedding worker (`run_embedding_worker`)
  and the recall pipeline share the same LiteLLM/ollama endpoint. Under load,
  recall embedding calls compete with background embedding jobs, increasing latency
  for both.

- **No cost floor.** Even with local BGE-M3 (no per-call API cost), the ollama
  inference time is a real resource cost. At 100 agents each recalling 10 times
  per hour, that is 1,000 embedding calls per hour that could be partially
  short-circuited.

The Redis research finding is directly applicable: semantic caching is the
highest-leverage latency and cost reduction for recall-heavy workloads.

---

## 2. Design Options

### Option A — In-Process LRU Cache (Minimal Viable)

An in-process LRU cache keyed by `(tenant_id, query_text_normalised)` with a
configurable TTL and a maximum size per tenant.

```python
from functools import lru_cache
from cachetools import TTLCache
import threading

# Per-tenant embedding cache: {tenant_id: TTLCache({query_hash: embedding})}
_embedding_cache: dict[str, TTLCache] = {}
_cache_lock = threading.Lock()

CACHE_TTL_SECONDS = 300        # 5 minutes
CACHE_MAX_SIZE_PER_TENANT = 256

def _get_tenant_cache(tenant_id: str) -> TTLCache:
    with _cache_lock:
        if tenant_id not in _embedding_cache:
            _embedding_cache[tenant_id] = TTLCache(
                maxsize=CACHE_MAX_SIZE_PER_TENANT,
                ttl=CACHE_TTL_SECONDS,
            )
        return _embedding_cache[tenant_id]
```

Cache key: `hashlib.sha256(query.lower().strip().encode()).hexdigest()` — exact
match only. No cosine similarity check at cache lookup time.

**Pros:** Zero infrastructure dependency. Works today. Negligible memory footprint
(256 entries × 4,100 bytes = ~1MB per tenant).

**Cons:** Exact match only — semantically equivalent queries with different phrasing
miss the cache. Cache is per-process — in a multi-replica platform deployment,
each replica has its own cache (no cross-replica hit rate). Cache is lost on
platform restart.

### Option B — Semantic Cache with Cosine Similarity Threshold

Cache stores `(query_embedding, result_set)` tuples. On a new query, embed it,
then check the cache for any stored embedding with cosine similarity ≥ threshold.
If found, return the cached result set without hitting Postgres.

```python
SEMANTIC_CACHE_THRESHOLD = 0.97   # must be validated against eval dataset
SEMANTIC_CACHE_TTL_SECONDS = 300
```

**Pros:** Catches paraphrased queries ("what are my priorities" vs "what should I
focus on"). Higher cache hit rate than Option A.

**Cons:** Still requires an embedding call to check the cache (the embedding is
needed for the similarity check). The latency saving is only on the Postgres
queries, not the embedding call itself. More complex eviction logic.

**Note:** This is only a net win if the Postgres queries are the dominant latency
contributor. With pgvector on local hardware, the vector search is typically
10–30ms. The embedding call is 5–50ms. The latency saving from Option B is
smaller than it appears.

### Option C — Shared Redis Cache (Production-Grade)

Use Redis (or a Redis-compatible store) as a shared semantic cache across all
platform replicas. The Redis research paper describes this as "LangCache" — a
fuzzy-match response cache with configurable similarity thresholds.

**Pros:** Cross-replica hit rate. Persistent across restarts. Configurable TTL
per tenant. The correct architecture for a multi-replica production deployment.

**Cons:** Adds Redis as a new infrastructure dependency. the platform currently uses
Postgres + pgvector for all data storage. Adding Redis requires a new service in
`docker-compose.yml`, a new Vault secret for the Redis connection string, and
operational overhead.

---

## 3. Recommendation

**Implement Option A first.** It is the minimal viable implementation with no
infrastructure dependency. It addresses the most common case (same agent, same
query, short time window) and establishes the cache interface that Option C can
replace later.

**ADR required before implementation.** The ADR must document:
1. The chosen option and rationale
2. The TTL value and its justification (5 minutes is a starting point — must be
   validated against observed query patterns in staging)
3. The cache key strategy (exact hash vs. semantic similarity)
4. The eviction policy and memory footprint calculation
5. The path to Option C if multi-replica deployment requires it

**Do not implement Option B.** The cosine similarity check at cache lookup time
requires an embedding call, which is the bottleneck being avoided. Option B only
makes sense if the Postgres queries are the dominant cost, which is not the case
at current scale.

---

## 4. Implementation Sketch (Option A)

```python
# platform/app/qortia/recall.py

async def _embed_query(
    query: str, tenant_id: UUID, lang: str = "en"
) -> list[float] | None:
    cache_key = hashlib.sha256(
        f"{tenant_id}:{lang}:{query.lower().strip()}".encode()
    ).hexdigest()
    tenant_cache = _get_tenant_cache(str(tenant_id))

    if cache_key in tenant_cache:
        return tenant_cache[cache_key]   # type: ignore[return-value]

    try:
        litellm_key = await get_litellm_key(str(tenant_id))
        resp = await get_litellm_client().post(
            "/embeddings",
            headers={"Authorization": f"Bearer {litellm_key}"},
            json={"model": EMBEDDING_MODEL, "input": query},
            timeout=10.0,
        )
        resp.raise_for_status()
        embedding: list[float] = resp.json()["data"][0]["embedding"]
        tenant_cache[cache_key] = embedding
        return embedding
    except Exception as exc:
        logger.warning({"event": "recall_embed_failed", "error": str(exc)})
        return None
```

**Dependency:** `cachetools` — add to `platform/pyproject.toml` as `cachetools>=5.3`.

---

## 5. Files Affected

| File | Change |
|---|---|
| `platform/app/qortia/recall.py` | Add `_get_tenant_cache`, cache lookup/write in `_embed_query` |
| `platform/app/config.py` | Add `embedding_cache_ttl_seconds`, `embedding_cache_max_size` settings |
| `platform/pyproject.toml` | Add `cachetools>=5.3` dependency |
| `docs/decisions/adrs/adr-NNN.md` | New ADR documenting cache design decisions |
| `docs/decisions/adr-log.md` | Add ADR row |

---

## 6. Test Gates

| Gate | What to verify |
|---|---|
| Unit test — `test_recall_pipeline.py` | Second call with identical query returns cached embedding (LiteLLM not called) |
| Unit test — `test_recall_pipeline.py` | Cache miss on different tenant_id for same query |
| Unit test — `test_recall_pipeline.py` | Cache miss after TTL expiry |
| Unit test — `test_recall_pipeline.py` | Cache miss on embed failure (no stale negative caching) |
| Recall eval | `evals/run_reh.py` — Recall@5 ≥ 0.95, MRR ≥ 0.86 (must not regress) |
| Platform unit tests | `cd platform && python3 -m pytest tests/unit/ -q` — 296/296 |

---

## 7. Known Constraints

**Thread safety.** `TTLCache` from `cachetools` is not thread-safe. The `_cache_lock`
in the design sketch above protects the dict of caches but not individual cache
operations. Use `cachetools.LRUCache` wrapped with `cachetools.cached` and a
`threading.RLock`, or use `cachetools.TTLCache` with explicit locking on all
read/write operations.

**Multi-replica invalidation.** Option A has no cross-replica cache invalidation.
If the platform runs with 2+ replicas, each replica has its own cache. This is
acceptable for the current deployment (single-replica platform in staging/prod as
of this writing). If horizontal scaling is added, migrate to Option C.

**Cache poisoning.** The cache stores embeddings, not recall results. A poisoned
embedding (e.g. from a transient ollama error returning a malformed vector) would
be cached and served for 5 minutes. The `resp.raise_for_status()` guard prevents
caching error responses. Malformed but non-error responses are not guarded — the
embedding dimension validation in `validate_embedding_dimensions()` runs at startup,
not per-call.

---

## 8. Related

- `_embed_query` definition: `recall.py`
- `run_embedding_worker` — shares the LiteLLM endpoint; cache reduces contention
- ADR-054 (embedding model selection) — relevant context
- ADR-081 (BGE-M3 adoption, MRL strategy) — pending; cache key must include model
  version if the embedding model changes
