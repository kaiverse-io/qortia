---
kind: enhancement
owner: platform
last_reviewed: 2026-05-18
status: implemented
---

# Multilingual Memory — Qortia Enhancement

> [!IMPORTANT]
> **E1 ✅ shipped (`865c663`) · E2 ✅ shipped (Stanza → spaCy, `dcf1920`) · E2 hardening ✅ shipped (ADR-096) · E3 ⚠️ superseded**
>
> E3 (IndicSBERT dual-model routing at 768-dim) was **never implemented**. The stack
> migrated directly to BGE-M3 (single model, 1024-dim, local ollama) in `a4965dd`.
> ADR-081 formally documents the BGE-M3 adoption, the `vector(768)` → `vector(1024)`
> schema migration, the `hi_core_news_sm` correction, and the v1/v2 eval baseline.
>
> The E3 design in Section 5 of this document is **historical only** — do not implement it.
> Refer to `docs/enhancements/qortia-memory-quality.md` G5 for the current MRL strategy.
>
> **E2 hardening (ADR-096):** Seven gaps in the original E2 implementation were fixed
> post-ship. See ADR-096 for the full decision record. Summary of changes:
> - `EN_ENTITY_LABELS` is now the single canonical label set used by all extraction functions.
>   `TECH` (phantom label) removed. `WORK_OF_ART` now present in all paths.
> - `_indic_pipelines` cache rekeyed from model name → `lang`.
> - `_get_indic_pipeline` has a structured error boundary (`spacy_model_load_failed`).
> - `load_spacy_model()` warms up `xx_ent_wiki_sm` at startup.
> - `extract_index_fields` now accepts and routes on `lang`.
> - `lang` normalised at the Pydantic boundary (`_normalise_lang`) on all three request models.
> - Unsupported languages log `ner_lang_unsupported` warning before English fallback.
> - `xx_ent_wiki_sm==3.8.0` declared in `platform/pyproject.toml`.

---

## 1. Problem Statement

Qortia is English-only today. Three independent subsystems enforce this implicitly:

1. **NER extraction** (`knowledge.py` → `extract_entities`) loads `en_core_web_sm` at startup. Running it against Hindi, Tamil, Telugu, or any other Indic script produces zero entities silently — no error, no fallback, just empty `entities: []` on every memory row. The entity graph (`qortia_entities`) is therefore blind to all non-English named entities, which degrades both the Obsidian Layer boost in recall and the graph-linked memory retrieval.

2. **BM25 full-text search** (`recall.py`) uses `plainto_tsquery('simple', ...)` against `tsvector` columns generated with `to_tsvector('simple', ...)` (V6 migration). The `simple` config is already language-agnostic (no stemming, no stop-word removal) — this is the one subsystem that already works correctly for Indic text. No change needed here.

3. **Embedding model** (`reflect.py` → `EMBEDDING_MODEL = "text-embedding-3-small"`) is a general-purpose multilingual model with reasonable Indic coverage, but it was not purpose-trained on Indian languages. Semantic recall quality for Hindi, Tamil, Telugu, Bengali, Kannada, Malayalam, Marathi, Gujarati, Punjabi, and Odia is measurably lower than for English. The purpose-built alternative (`ai4bharat/IndicSBERT`) outputs 768-dim vectors — the same dimension the schema already uses — making a model swap possible without a migration.

Additionally, there is a **missing metadata gap**: no memory row records what language it was written in. This means there is no way to filter, route, or audit memories by language, and no way to measure recall quality per language in production.

---

## 2. Scope of This Document

This document covers three enhancements in dependency order:

| Enhancement | Risk | ADR | Migration | Status |
|---|---|---|---|---|
| **E1** — Agent-reported language metadata | Low | No | V3 — add `lang` column | ✅ `865c663` |
| **E2** — Multilingual NER via Stanza | Medium | No | None | ✅ `a99e05b` |
| **E3** — Indic embedding model swap | Medium | ADR-079 (superseded ADR-081) | `vector(768)` → `vector(1024)` | ✅ `a4965dd` (BGE-M3) |

E1 must ship before E2 and E3 — the `lang` column is the routing signal both depend on.

---

## 3. E1 — Agent-Reported Language Metadata

### 3.1 Design

The agent's LLM already knows what language it is processing — it is responding in that language. Rather than adding a separate language-detection call (extra latency, extra cost), the LLM is instructed to surface the language as a structured field in the tool call arguments.

This is zero-cost: the LLM produces the `lang` field as part of the same inference pass that produces the memory content. No additional API call. No additional model.

The `lang` field is an optional BCP-47 language tag (`"en"`, `"hi"`, `"ta"`, `"te"`, `"bn"`, `"kn"`, `"ml"`, `"mr"`, `"gu"`, `"pa"`, `"or"`). It defaults to `"en"` if absent, preserving full backward compatibility with existing agents that do not set it.

### 3.2 Migration — V7

```sql
-- V7: Add lang column to memory tables for multilingual routing.
-- BCP-47 language tag. Defaults to 'en' — all existing rows are valid.
-- Used by NER routing (E2) and embedding model routing (E3).

ALTER TABLE hindsight_memories ADD COLUMN lang TEXT NOT NULL DEFAULT 'en';
ALTER TABLE org_memory         ADD COLUMN lang TEXT NOT NULL DEFAULT 'en';
ALTER TABLE org_knowledge      ADD COLUMN lang TEXT NOT NULL DEFAULT 'en';
```

No index needed at this stage — `lang` is a filter hint, not a primary lookup key.

### 3.3 API Changes — `remember` and `recall`

**`platform/app/qortia/models.py`** — add `lang` to request models:

```python
class MemoryItem(BaseModel):
    type: Literal["episodic", "experiential", "mental_model", "decision", "lesson"]
    content: str
    source_task_id: str | None = None
    metadata: dict | None = None
    lang: str = "en"  # BCP-47, optional


class RememberRequest(BaseModel):
    memories: list[MemoryItem]


class RememberOrgRequest(BaseModel):
    type: Literal["handoff", "process", "decision_log"]
    title: str
    content: str
    lang: str = "en"


class RecallRequest(BaseModel):
    query: str
    scope: Literal["private", "org", "knowledge", "all"] = "all"
    type: str | None = None
    rerank: bool = False
    entities: list[str] | None = None
    lang: str | None = None  # None = search all languages
```

**`platform/app/qortia/remember.py`** — pass `lang` to INSERT:

```python
# In remember() — hindsight_memories INSERT
row_id = await conn.fetchval(
    """
    INSERT INTO hindsight_memories
        (tenant_id, agent_id, type, content, importance,
         source_task_id, metadata, entities, lang)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    RETURNING id
    """,
    agent.tenant_id,
    agent.agent_id,
    mem.type,
    mem.content,
    IMPORTANCE[mem.type],
    mem.source_task_id,
    json.dumps(mem.metadata) if mem.metadata else "{}",
    json.dumps(entities),
    mem.lang,
)

# In remember_org() — org_memory INSERT (both branches)
# Add lang=$N to both the handoff INSERT and the upsert INSERT/UPDATE
```

**`platform/app/qortia/recall.py`** — optional lang filter on private and org queries:

```python
def _lang_filter_clause(lang: str | None, param: int) -> tuple[str, list]:
    if not lang:
        return "", []
    return f"AND lang = ${param}", [lang]
```

Add this clause to `_bm25_private`, `_vector_private`, `_bm25_org`, `_vector_org`. Knowledge search does not filter by lang — knowledge documents may be multilingual within a single chunk.

### 3.4 MCP Bridge Changes — `mcp_bridge.py`

Add `lang` to the `remember`, `remember_org`, and `recall` tool schemas and pass it through:

```python
# list_tools() — remember
Tool(name="remember", description=(
    "Write private memories. Triggers reflection after every 10 episodic memories. "
    "Each memory: {type: episodic|experiential|mental_model|decision|lesson, "
    "content: string, lang: BCP-47 language code (default: 'en')}. "
    "Set lang to the language of the content — e.g. 'hi' for Hindi, 'ta' for Tamil."
), inputSchema={"type": "object", "required": ["memories"], "properties": {
    "memories": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "content": {"type": "string"},
            "lang": {"type": "string", "description": "BCP-47 language code"},
        }
    }},
}}),

# list_tools() — recall
# Add lang to inputSchema properties (optional, no required change)
"lang": {"type": "string", "description": "Filter results by language. Omit to search all."},
```

```python
# _dispatch_tool — recall case
case "recall":
    return await _recall(
        args["query"], args.get("scope", "all"),
        args.get("type"), args.get("rerank", False),
        args.get("entities"), args.get("lang"),
    )

# _recall() signature
async def _recall(
    query: str,
    scope: str = "all",
    type_filter: str | None = None,
    rerank: bool = False,
    entities: list[str] | None = None,
    lang: str | None = None,
) -> str:
    body: dict = {"query": query, "scope": scope, "rerank": rerank}
    if type_filter:
        body["type"] = type_filter
    if entities:
        body["entities"] = entities
    if lang:
        body["lang"] = lang
    ...

# _remember() — memories list already passes through as-is to the API
# lang is inside each memory dict — no change needed in _remember()
```

### 3.5 Weekly Summary Lang Inference

The weekly summary background task (`knowledge.py → _summarise_tenant`) writes a `weekly_summary` row to `org_memory` by concatenating the week's handoff notes. After V7, this row gets `lang = 'en'` by default — the INSERT does not set `lang`.

If a tenant's handoffs are written in Hindi, the summary row is embedded with `bge-m3`
(single model for all languages since ADR-081). The `lang` column is still used for
BM25 routing and NER — it does not affect which embedding model is used.

The fix is to infer the dominant language from the source handoffs before writing the summary row. Since handoffs already carry `lang` after E1 ships, the inference is a simple majority vote:

```python
# knowledge.py — _summarise_tenant(), before the weekly_summary INSERT
from collections import Counter

lang_counts = Counter(h["lang"] for h in handoffs if h.get("lang"))
dominant_lang = lang_counts.most_common(1)[0][0] if lang_counts else "en"

await conn.execute(
    """
    INSERT INTO org_memory (tenant_id, type, title, content, author_id, entities, lang)
    VALUES ($1, 'weekly_summary', $2, $3, NULL, '[]', $4)
    """,
    tenant_id,
    f"Weekly Summary — {datetime.date.today().isoformat()}",
    summary_content,
    dominant_lang,
)
```

The handoff query in `_summarise_tenant` must also SELECT `lang`:

```sql
SELECT om.title, om.content, om.created_at, om.lang, a.name AS agent_name
FROM org_memory om
...
```

This is part of E1 — it requires the `lang` column on `org_memory` (V7) and the `lang` field on `RememberOrgRequest` to be in place first.

### 3.6 Test Gate

Add to `platform/tests/unit/test_qortia_models.py`:
- `lang` defaults to `"en"` when absent
- `lang` is preserved when set to `"hi"`, `"ta"`, etc.
- `RecallRequest.lang = None` produces no lang filter clause
- Weekly summary dominant lang inference: majority Hindi handoffs → summary `lang = "hi"`

---

## 4. E2 — Multilingual NER via Stanza

### 4.1 Why Not spaCy `xx_ent_wiki_sm`

spaCy's multilingual model covers 7 languages (EN, DE, ES, FR, IT, PT, RU). None of the major Indian languages are included. It is not a viable path for Indic NER.

### 4.2 Stanza Coverage for Indian Languages

[Stanford Stanza](https://stanfordnlp.github.io/stanza/) has trained NER models for:

| Language | BCP-47 | Stanza NER | Quality |
|---|---|---|---|
| Hindi | `hi` | ✅ | Good |
| Bengali | `bn` | ✅ | Good |
| Tamil | `ta` | ⚠️ | Partial — tokenisation strong, NER limited |
| Telugu | `te` | ⚠️ | Partial |
| Marathi | `mr` | ⚠️ | Partial |
| Kannada | `kn` | ❌ | Tokenisation only |
| Malayalam | `ml` | ❌ | Tokenisation only |
| Gujarati | `gu` | ❌ | Tokenisation only |
| Punjabi | `pa` | ❌ | Tokenisation only |
| Odia | `or` | ❌ | Tokenisation only |

For languages where Stanza has no NER model, `extract_entities` returns `[]` — the same behaviour as today. No regression.

### 4.3 Architecture — Language-Routed NER

The routing decision is made in `extract_entities` based on the `lang` parameter added in E1. spaCy remains the default for English. Stanza is loaded lazily on first use for each supported language — Stanza models are downloaded at container build time, not at runtime.

```python
# knowledge.py

import stanza  # new dependency

_stanza_pipelines: dict[str, object] = {}  # lang → stanza.Pipeline

STANZA_NER_LANGS = frozenset({"hi", "bn", "ta", "te", "mr"})


def _get_stanza_pipeline(lang: str) -> object:
    if lang not in _stanza_pipelines:
        _stanza_pipelines[lang] = stanza.Pipeline(
            lang,
            processors="tokenize,ner",
            download_method=None,  # models pre-downloaded at build time
            verbose=False,
        )
    return _stanza_pipelines[lang]


def extract_entities(text: str, lang: str = "en") -> list[str]:
    """
    Extract NER entities. Routes to spaCy (English) or Stanza (Indic).
    Best-effort — caller wraps in try/except and falls back to [].
    """
    if lang in STANZA_NER_LANGS:
        nlp = _get_stanza_pipeline(lang)
        doc = nlp(text)  # type: ignore[operator]
        entities = []
        for sent in doc.sentences:
            for ent in sent.ents:
                if ent.type in {"PER", "ORG", "LOC", "MISC"}:
                    entities.append(ent.text)
        return list(dict.fromkeys(entities))[:20]

    # Default: spaCy English
    doc = get_nlp()(text)  # type: ignore[operator]
    return list(dict.fromkeys(ent.text for ent in doc.ents if ent.label_ in ENTITY_LABELS))[:20]
```

All call sites in `remember.py` and `reflect.py` pass `lang` through:

```python
# remember.py — remember()
entities = extract_entities(mem.content, lang=mem.lang)

# remember.py — remember_org()
entities = extract_entities(body.content, lang=body.lang)

# reflect.py — inside the reflection write loop
entities = extract_entities(r["content"], lang=r.get("lang", "en"))
```

The recall pipeline's entity extraction for query boosting (`recall.py` → `extract_entities(body.query)`) does not have a `lang` signal from the query body today. After E1 ships, `body.lang` can be passed here too. For now, it falls back to English NER on the query — acceptable since entity boost is additive, not required for recall to function.

### 4.4 Dependency Changes

**`the agent runtime/requirements.txt`** — add Stanza with exact pin:

```
stanza==1.10.1
```

**`the agent runtime/Dockerfile`** — pre-download Stanza models at build time (not at runtime):

```dockerfile
# After pip install -r requirements.txt
RUN python3 -c "\
import stanza; \
stanza.download('hi', processors='tokenize,ner', verbose=False); \
stanza.download('bn', processors='tokenize,ner', verbose=False); \
stanza.download('ta', processors='tokenize,ner', verbose=False); \
stanza.download('te', processors='tokenize,ner', verbose=False); \
stanza.download('mr', processors='tokenize,ner', verbose=False); \
"
```

Stanza model downloads are ~50–150MB per language. Total image size increase: ~500MB for all 5 languages. This is acceptable for an agent base image — agents are long-running pods, not ephemeral functions.

**`platform/pyproject.toml`** — add Stanza as a platform dependency for the NER path:

```toml
"stanza>=1.10",
```

Note: the platform also calls `extract_entities` (via `knowledge.py` which is imported by `remember.py` and `reflect.py`). The platform must also have Stanza installed. The `the agent runtime/requirements.txt` exact pin and the `pyproject.toml` floor constraint follow the existing pattern for spaCy.

### 4.5 Startup Change

`knowledge.py` currently calls `load_spacy_model()` at startup. No equivalent eager load is needed for Stanza — pipelines are loaded lazily on first use. The Stanza model files are already on disk (downloaded at image build time), so lazy load is fast (~1–2s per language, once).

### 4.6 Test Gate

Add to `platform/tests/unit/test_qortia_models.py` (or a new `test_knowledge.py`):
- `extract_entities("Narendra Modi visited Mumbai", lang="en")` returns `["Narendra Modi", "Mumbai"]`
- `extract_entities("नरेंद्र मोदी मुंबई गए", lang="hi")` returns non-empty list (Stanza path)
- `extract_entities("some text", lang="kn")` returns `[]` without raising (unsupported lang passthrough)

---

## 5. E3 — Indic Embedding Model Swap

### 5.1 Current State

`reflect.py` hardcodes `EMBEDDING_MODEL = "text-embedding-3-small"`. This model is routed through the LiteLLM gateway. `validate_embedding_dimensions()` in `reflect.py` enforces that the model returns exactly 768-dim vectors at startup — if the dimension does not match, the platform refuses to start.

The schema uses `vector(768)` on `hindsight_memories.embedding`, `org_memory.embedding`, `org_knowledge.embedding`, and `qortia_entities.embedding`.

### 5.2 Model Comparison

| Model | Indic Coverage | Dimensions | Hosting | Notes |
|---|---|---|---|---|
| `text-embedding-3-small` (current) | Decent, not purpose-built | 1536 | OpenAI API | Current model — dimension mismatch with schema |
| `text-embedding-3-large` | Better, still not purpose-built | 3072 | OpenAI API | Dimension mismatch |
| `intfloat/multilingual-e5-large` | Broad, Indic included | 1024 | Self-host / OpenRouter | Dimension mismatch |
| `ai4bharat/IndicSBERT` | 11 Indian languages, purpose-built | 768 | Self-host via HuggingFace | **Exact dimension match** |
| `ai4bharat/indic-bert` | 12 Indian languages | 768 | Self-host | Masked LM, not sentence-transformer — weaker for semantic search |

**`ai4bharat/IndicSBERT` is the correct choice.** It is a sentence-transformer model (same architecture as what pgvector cosine search expects), purpose-trained on Hindi, Tamil, Telugu, Bengali, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Odia, and Assamese. Critically, it outputs 768-dim vectors — the schema requires no migration.

The 11 supported languages:

| Language | BCP-47 |
|---|---|
| Hindi | `hi` |
| Tamil | `ta` |
| Telugu | `te` |
| Bengali | `bn` |
| Kannada | `kn` |
| Malayalam | `ml` |
| Marathi | `mr` |
| Gujarati | `gu` |
| Punjabi | `pa` |
| Odia | `or` |
| Assamese | `as` |

### 5.3 Architecture — Dual-Model Routing

Rather than replacing `text-embedding-3-small` globally (which would degrade English recall quality — IndicSBERT is weaker on English than a purpose-built English model), the correct approach is **per-memory model routing** based on the `lang` field added in E1.

```
lang = "en"  →  text-embedding-3-small  (current behaviour, unchanged)
lang in INDIC_LANGS  →  ai4bharat/IndicSBERT
lang = other  →  text-embedding-3-small  (safe fallback)
```

Both models output 768-dim vectors. The same `vector(768)` column stores both. Cosine similarity is valid across both — the embedding space is different per model, but since queries are always embedded with the same model as the stored memory (routing is symmetric), cross-model comparison never occurs in practice.

### 5.4 LiteLLM Gateway Configuration

Add `ai4bharat/IndicSBERT` as a model in the LiteLLM config. It can be served via:

- A local HuggingFace inference endpoint (recommended for self-hosted deployments)
- OpenRouter (if they add it — not currently listed)
- A dedicated `sentence-transformers` FastAPI wrapper (simplest for self-host)

The LiteLLM config addition (in `litellm_config.yaml` or equivalent):

```yaml
model_list:
  - model_name: indic-embedding
    litellm_params:
      model: huggingface/ai4bharat/IndicSBERT
      api_base: http://indic-embed-svc:8080   # internal K8s service
```

The model name `"indic-embedding"` is what Qortia will use when routing Indic content.

### 5.5 Code Changes — `reflect.py`

```python
EMBEDDING_MODEL_EN = "text-embedding-3-small"
EMBEDDING_MODEL_INDIC = "indic-embedding"

INDIC_LANGS = frozenset({"hi", "ta", "te", "bn", "kn", "ml", "mr", "gu", "pa", "or", "as"})


def _embedding_model_for(lang: str) -> str:
    return EMBEDDING_MODEL_INDIC if lang in INDIC_LANGS else EMBEDDING_MODEL_EN
```

**`_get_embedding`** — add `lang` parameter:

```python
async def _get_embedding(text: str, litellm_key: str, lang: str = "en") -> list[float]:
    model = _embedding_model_for(lang)
    resp = await get_litellm_client().post(
        "/embeddings",
        headers={"Authorization": f"Bearer {litellm_key}"},
        json={"model": model, "input": text},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]
```

**`_embed_single_row`** — pass `lang` from the row:

```python
async def _embed_single_row(row: dict, litellm_key: str) -> None:
    if not row.get("text_to_embed"):
        return
    lang = row.get("lang", "en")
    try:
        embedding = await _get_embedding(row["text_to_embed"], litellm_key, lang=lang)
        ...
```

The embedding worker query in `_process_embedding_batch` must also SELECT the `lang` column:

```sql
SELECT id, tenant_id, content AS text_to_embed, lang, 'hindsight_memories' AS tbl
FROM hindsight_memories
WHERE embedding IS NULL AND embedding_attempts < 3
...
```

Same for `org_memory` and `org_knowledge`. `qortia_entities` does not have a `lang` column — entity text is language-agnostic at the graph level; use `EMBEDDING_MODEL_EN` for entities.

**`validate_embedding_dimensions`** — validate both models at startup:

```python
async def validate_embedding_dimensions() -> None:
    for model, expected_dim in [
        (EMBEDDING_MODEL_EN, 768),
        (EMBEDDING_MODEL_INDIC, 768),
    ]:
        resp = await get_litellm_client().post(
            "/embeddings",
            headers={"Authorization": f"Bearer {settings.litellm_master_key}"},
            json={"model": model, "input": "dimension check"},
            timeout=10.0,
        )
        resp.raise_for_status()
        actual = len(resp.json()["data"][0]["embedding"])
        if actual != expected_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch for {model}: "
                f"schema expects {expected_dim}, got {actual}."
            )
```

**`recall.py` — `_embed_query`** — pass `lang` from the request:

```python
async def _embed_query(query: str, tenant_id: UUID, lang: str = "en") -> list[float] | None:
    try:
        litellm_key = await get_litellm_key(str(tenant_id))
        model = _embedding_model_for(lang)
        resp = await get_litellm_client().post(
            "/embeddings",
            headers={"Authorization": f"Bearer {litellm_key}"},
            json={"model": model, "input": query},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as exc:
        logger.warning({"event": "recall_embed_failed", "error": str(exc)})
        return None
```

The `recall` endpoint passes `body.lang` (defaulting to `"en"`) to `_embed_query`.

### 5.6 ADR-079

This change requires ADR-079 because:
- It introduces a second embedding model into the platform
- It changes the semantic meaning of the `embedding` column (rows may now contain vectors from two different embedding spaces)
- Cross-model cosine similarity is technically valid (both 768-dim) but semantically undefined — the ADR must document that query routing is always symmetric (query embedded with same model as stored memory)
- The `validate_embedding_dimensions` startup check now covers two models — the ADR documents the startup failure mode if `indic-embedding` is unavailable

### 5.7 Re-embedding Existing Memories

Existing memories written before E3 ships have `lang = "en"` (the V7 migration default). They were embedded with `text-embedding-3-small`. After E3 ships, new Indic memories will be embedded with IndicSBERT. No re-embedding of existing rows is needed — the routing is symmetric.

If a tenant later updates existing memories to set `lang = "hi"` (e.g. via a backfill script), those rows would need re-embedding. That is a separate operational concern, not part of this enhancement.

---

## 6. Dependency Order and Shipping Sequence

```
E1 (lang column + API + bridge)
  └─ V7 migration
  └─ models.py changes
  └─ remember.py / recall.py lang pass-through
  └─ mcp_bridge.py tool schema update
  └─ Unit tests

E2 (Stanza NER routing)  ← depends on E1 (needs lang signal)
  └─ knowledge.py extract_entities routing
  └─ stanza dependency + Dockerfile model download
  └─ All extract_entities call sites updated
  └─ Unit tests

E3 (IndicSBERT embedding routing)  ← depends on E1 (needs lang on rows)
  └─ ADR-079
  └─ LiteLLM gateway config (indic-embedding model)
  └─ reflect.py dual-model routing
  └─ recall.py _embed_query lang routing
  └─ validate_embedding_dimensions covers both models
  └─ Staging canary: provision agent, write Hindi memory, verify embedding non-null
```

E2 and E3 are independent of each other — they can ship in either order after E1.

---

## 7. What This Does Not Solve

**Script rendering in Mission Control**: The Next.js dashboard renders memory content as plain text. Devanagari, Tamil, and other Indic scripts render correctly in modern browsers without any code change — this is not a gap.

**LLM inference quality for Indic languages**: The agent's reasoning quality in Hindi or Tamil depends on the model configured in `domain_md` (e.g. `claude-3-haiku`). Claude and GPT-4 class models have strong multilingual capability. This is outside Qortia scope.

**Transliteration**: Memories written in romanised Hindi (e.g. "Narendra Modi Mumbai gaye") will not match memories written in Devanagari. This is a known limitation of all embedding-based systems and is not addressed here.

**Cross-language recall**: A query in Hindi will not retrieve semantically equivalent memories written in English, and vice versa. The two embedding spaces are separate. Cross-lingual retrieval requires a single multilingual model for all content — that is a larger architectural decision deferred to a future enhancement.

---

## 8. Acceptance Criteria

### E1
- [x] V7 migration applies cleanly against a fresh stack (`flyway migrate` exit 0)
- [x] `remember` API accepts `lang` field per memory item, stores it in `hindsight_memories.lang`
- [x] `remember-org` API accepts `lang` field, stores it in `org_memory.lang`
- [x] `recall` API accepts optional `lang` filter, applies `AND lang = $N` when set
- [x] `mcp_bridge.py` `remember` tool schema includes `lang` in each memory item
- [x] `mcp_bridge.py` `recall` tool schema includes optional `lang` field
- [x] Existing agents that do not set `lang` continue to work — all rows default to `"en"`
- [x] Weekly summary row inherits dominant `lang` from source handoffs
- [x] Unit tests pass: `lang` default, `lang` preservation, `lang` filter clause, weekly summary lang inference

### E2
- [x] `extract_entities("नरेंद्र मोदी मुंबई गए", lang="hi")` returns non-empty list
- [x] `extract_entities("some text", lang="kn")` returns `[]` without raising (unsupported lang fallback)
- [x] `extract_entities("OpenAI released GPT-4", lang="en")` continues to use spaCy path
- [x] `xx_ent_wiki_sm` present in agent base image and `platform/pyproject.toml` (pinned `3.8.0`)
- [x] `extract_index_fields` routes on `lang` — Indic knowledge chunks use Indic pipeline
- [x] `EN_ENTITY_LABELS` is the single canonical label set — no divergence between extraction functions
- [x] `lang` normalised at Pydantic boundary on `MemoryItem`, `RememberOrgRequest`, `KnowledgeIngestRequest`
- [x] Unsupported lang logs `ner_lang_unsupported` warning, does not raise
- [x] `_indic_pipelines` cache keyed by `lang`, not model name
- [x] `_get_indic_pipeline` logs `spacy_model_load_failed` on `OSError` and re-raises
- [x] `load_spacy_model()` warms up `xx_ent_wiki_sm` at startup
- [x] Platform unit tests pass: 798/798
- [x] Full stack health: `python3 scripts/local_agents.py` 21/21

### E3
- [x] ADR-079 written and merged before code ships
- [x] LiteLLM gateway serves `bge-m3` model (ADR-081 — replaces `indic-embedding`)
- [x] `validate_embedding_dimensions()` passes for bge-m3 at platform startup (1024-dim, 60s timeout)
- [x] Hindi memory written via `remember` tool has non-null `embedding` in DB after worker cycle
- [x] Hindi recall query returns semantically relevant results (manual canary test)
- [x] English recall quality unchanged (regression canary: existing English memories still recalled correctly)
- [x] Platform unit tests pass: 798/798
- [x] Staging canary: agent reaches `status=active`, `boot_complete` logged, tool call succeeds
