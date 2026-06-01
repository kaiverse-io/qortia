---
kind: architecture
status: active
owner: platform
last_reviewed: 2026-06-01
---

# Qortia — System Overview

Read this first. Four diagrams that show the complete picture in order of complexity.
For detailed SQL, ADR rationale, and API schemas see [`01-design.md`](01-design.md) and
[`recall-pipeline.md`](../architecture/diagrams/recall-pipeline.md).

---

## 1. Component Map — What Connects to What

Every write, read, and async background job across the Qortia memory layer.

```mermaid
flowchart TD
    Agent["Agent\n(the agent runtime container)"]
    Bridge["mcp_bridge.py\nauto-injects X-Work-Order-Id"]

    Agent -->|MCP tool calls| Bridge

    Bridge -->|"POST /v1/remember\n(private episodic/lesson/decision…)"| Remember
    Bridge -->|"POST /v1/remember-org\n(process/handoff/org_chart…)"| RememberOrg
    Bridge -->|"POST /v1/knowledge\n(doc corpus ingestion)"| Knowledge
    Bridge -->|"POST /v1/recall\n(+ X-Work-Order-Id header)"| Recall
    Bridge -->|"POST /v1/reflect\n(or auto-triggered at ≥10 new episodics)"| Reflect

    subgraph Platform["Platform API  (platform/app/qortia/)"]
        Remember["remember.py\nspaCy NER → INSERT hindsight_memories"]
        RememberOrg["remember.py\nINSERT org_memory"]
        Knowledge["knowledge.py\nsplit_into_sections → INSERT org_knowledge"]
        Recall["recall.py\ntype-route → BM25+vector → RRF → response"]
        Reflect["reflect.py\nLiteLLM → SUPERSEDE old → INSERT consolidated"]
    end

    subgraph DB["PostgreSQL + pgvector"]
        HM["hindsight_memories\nRLS: agent-scoped\nvalid_until · confidence_multiplier"]
        OM["org_memory\nRLS: tenant + RBAC clearance\nvalid_until · confidence_multiplier"]
        OK["org_knowledge\nRLS: tenant + RBAC clearance\nchunked + indexed"]
        CE["qortia_entities\nBFS entity graph\nmax_clearance_order guard"]
        ML["memory_links\ncosine ≥ 0.70 pairs\nbidirectional"]
        SR["qortia_session_reads\nwork_order_id → memory_id log"]
        OR["qortia_outcome_records\noutcome → confidence decay record"]
    end

    subgraph Workers["Async Background Workers"]
        EmbW["Embedding Worker\nBGE-M3 1024-dim via LiteLLM\n_find_similar_memories → memory_links\n_populate_graph_batch → qortia_entities"]
        BgR["Background Reflection\nidle_reflection_interval_s\nruns reflect pipeline autonomously"]
    end

    subgraph WO["Work Orders (ADR-125)"]
        WORouter["work_orders/router.py\non completed/failed:\n_record_work_order_outcome()"]
    end

    Remember --> HM
    RememberOrg --> OM
    Knowledge --> OK
    Recall --> HM & OM & OK
    Recall -->|fire-and-forget\nif X-Work-Order-Id| SR
    Reflect --> HM

    HM & OM & OK -->|"embedding IS NULL"| EmbW
    EmbW --> HM & OM & OK
    EmbW --> ML & CE
    BgR --> Reflect

    SR -->|"query session reads\non WO terminal state"| WORouter
    WORouter -->|"UPDATE confidence_multiplier\nSUCCESS ×1.05 / MINOR ×0.85 / CRITICAL ×0.60"| HM & OM
    WORouter --> OR
```

---

## 2. Recall Pipeline — One Request End to End

Every path a `POST /v1/recall` request can take, including filters, scoring, and post-response side effects.

```mermaid
flowchart TD
    Req["POST /v1/recall\n{query, scope, type, entities, as_of}"]

    Req --> TypeRouter{"type?"}

    TypeRouter -->|episodic| Ep["_recall_episodic\nBM25 + 7-day recency window\nvalid_until filter"]
    TypeRouter -->|decision| De["_recall_decisions\nBM25 only\nrecency sort"]
    TypeRouter -->|lesson| Le["_recall_lessons\nvector cosine ≥ 0.35"]
    TypeRouter -->|mental_model\nexperiential| Hi["Full hybrid pipeline"]
    TypeRouter -->|none / all| Hi

    Hi --> Embed["Embed query\nBGE-M3 1024-dim\nvia LiteLLM"]
    Embed --> Scope{"scope?"}

    Scope -->|private / all| P1["BM25 private\nhindsight_memories\nvalid_until IS NULL OR > now()"]
    Scope -->|private / all| P2["Vector private\nhindsight_memories\nvalid_until IS NULL OR > now()"]
    Scope -->|org / all| O1["BM25 org\norg_memory\nRBAC clearance filter\nOR NOT EXISTS guard\nvalid_until IS NULL OR > now()"]
    Scope -->|org / all| O2["Vector org\norg_memory\nRBAC clearance filter\nvalid_until IS NULL OR > now()"]
    Scope -->|knowledge / all| K1["BM25 knowledge\norg_knowledge\nRBAC clearance filter\nOR NOT EXISTS guard"]
    Scope -->|knowledge / all| K2["Vector knowledge\nMMR λ=0.5\nkeyword_boost pre-MMR"]

    P1 & P2 & O1 & O2 --> RRF
    K1 & K2 --> RRF

    RRF["RRF Fusion\nscore = Σ 1/(60+rank)\nfinal_score × dynamic_importance\n  base_importance + log(recall_count) + recency_decay\n  × confidence_multiplier  ← ADR-125"]

    RRF --> EF{"entities\nfilter?"}
    EF -->|yes| EntFilter["filter: entities ∩ query_entities ≠ ∅"]
    EF -->|no| Links

    EntFilter --> EntityBoost["entity adjacency boost\nqortia_entities BFS 2 hops\n1.5× per hop, 0.5 decay"]
    EntityBoost --> Links

    Links["_expand_with_links()\nmemory_links cosine pairs\nvalid_until IS NULL OR > now()"]

    Links --> Response["Return top-K results\nRecallResult list"]

    Response -->|"fire-and-forget\n(separate asyncpg conn)"| FA1["_record_recall_access()\nrecall_count++ last_recalled_at=now()"]
    Response -->|"fire-and-forget\nif X-Work-Order-Id present"| FA2["_log_session_reads()\nINSERT qortia_session_reads"]

    classDef filter fill:#e8f4e8,stroke:#4a9e4a
    classDef async fill:#fff3cd,stroke:#b8860b
    classDef fusion fill:#dce8f8,stroke:#3a72b8
    class P1,P2,O1,O2,K1,K2 filter
    class FA1,FA2 async
    class RRF fusion
```

---

## 3. Memory Lifecycle — From Write to Decay

How a single memory travels from creation through to irrelevance.

```mermaid
stateDiagram-v2
    direction LR

    [*] --> Written : POST /v1/remember\nINSERT hindsight_memories\nembedding = NULL

    Written --> Embedded : Embedding worker\nBGE-M3 1024-dim\nUPDATE embedding

    Embedded --> Linked : _find_similar_memories()\ncosine ≥ 0.70\nINSERT memory_links

    Linked --> Graphed : _populate_graph_batch()\nUPSERT qortia_entities\nlinked_memory_ids array_append

    Graphed --> Active : Recallable\nrecall_count increments\nlast_recalled_at updated

    Active --> Active : Recalled by agent\ndynamic_importance boosts\nconfidence_multiplier adjusted\nby WO outcome (ADR-125)

    Active --> Superseded : reflect.py SUPERSEDE\nis_consolidated = false\nvalid_until = now()\n(ADR-027 + ADR-078)

    Active --> Consolidated : reflect.py produces\nmental_model / lesson\nis_consolidated = true\nimportance = 0.85

    Consolidated --> Active : Still recallable\nranks above raw episodic\nfor same query

    Superseded --> [*] : Excluded from default recall\nvalid_until IS NULL OR > now()\nfilter removes it from all 10 paths

    note right of Active
        confidence_multiplier default = 1.0
        SUCCESS WO  → ×1.05  (cap 1.0)
        MINOR fail  → ×0.85  (compounding)
        CRITICAL    → ×0.60  (floor 0.10)
    end note

    note right of Superseded
        Still in DB — never deleted.
        Queryable via as_of parameter.
        Audit trail via memory_history.
    end note
```

---

## 4. ADR-125 Causal Feedback Loop

How work order outcomes flow back into memory quality scores.

```mermaid
sequenceDiagram
    participant Bridge as mcp_bridge.py
    participant Platform as recall.py
    participant SR as qortia_session_reads
    participant WOR as work_orders/router.py
    participant OR as qortia_outcome_records
    participant HM as hindsight_memories

    Note over Bridge,HM: During a work order (X-Work-Order-Id header present)

    Bridge->>Platform: POST /v1/recall {X-Work-Order-Id: wo-123}
    Platform->>HM: BM25 + vector queries (valid_until filter + RBAC)
    Platform-->>Bridge: [memory-A, memory-B, memory-C] top results
    Platform->>SR: asyncio.create_task INSERT (wo-123, memory-A)<br/>INSERT (wo-123, memory-B)<br/>INSERT (wo-123, memory-C)<br/>fire-and-forget, never blocks recall

    Note over Bridge,HM: Agent completes or fails the work order

    Bridge->>WOR: PATCH /v1/work-orders/wo-123/advance {state: completed}
    WOR->>SR: SELECT memory_id WHERE work_order_id = wo-123
    SR-->>WOR: [memory-A, memory-B, memory-C]

    alt state = completed
        WOR->>HM: UPDATE confidence_multiplier = min(1.0, multiplier × 1.05)<br/>WHERE id IN [A, B, C]
    else state = failed (MINOR)
        WOR->>HM: UPDATE confidence_multiplier = max(0.10, multiplier × 0.85)
    else state = failed (CRITICAL)
        WOR->>HM: UPDATE confidence_multiplier = max(0.10, multiplier × 0.60)
    end

    WOR->>OR: INSERT qortia_outcome_records (wo-123, outcome, memory_count=3)

    Note over Bridge,HM: Next recall — confidence_multiplier already baked into DB row

    Bridge->>Platform: POST /v1/recall {query: ...}
    Platform->>HM: SELECT ... confidence_multiplier
    Note right of Platform: dynamic_importance() applies<br/>confidence_multiplier as final<br/>post-RRF scaling factor.<br/>Failed memories drift down.<br/>Successful memories drift up.
    Platform-->>Bridge: ranked results reflect outcome history
```

---

## Reading Order

| Want to understand… | Read |
|---|---|
| How the pieces connect (this page) | `docs/qortia/00-overview.md` |
| Data model, SQL, ADR rationale | [`01-design.md`](01-design.md) |
| Recall SQL details, sequence diagrams | [`recall-pipeline.md`](../architecture/diagrams/recall-pipeline.md) |
| How to measure and benchmark quality | [`02-benchmarking.md`](02-benchmarking.md) |
| Eval harnesses and current scores | [`03-eval-strategy.md`](03-eval-strategy.md) |
| API request/response schemas | [`04-api-contracts.md`](04-api-contracts.md) |
| All shipped ADRs | [`decisions/qortia.md`](../decisions/qortia.md) |
