---
kind: enhancement
owner: platform
last_reviewed: 2026-05-18
status: implemented
---

# Recall: Decouple LLM Rerank Model from Agent Domain Model

**Status:** Implemented — 2026-05-30
**Scope:** `platform/app/qortia/recall.py`
**ADR required:** No — no schema change, no API contract change; platform settings
addition is additive
**Depends on:** None
**Research source:** "Managing Memory for AI Agents" (O'Reilly/Redis, 2026) — Ch. 3,
"The Model Selection Matrix: A Multifactor Analysis" — task-specific model selection
vs. single-model-for-all-tasks anti-pattern

---

## 1. The Problem

`_llm_rerank` in `recall.py` reads the agent's `domain_md` to determine which model
to use for reranking:

```python
model = yaml.safe_load(domain_md_raw).get(
    "model", "anthropic/claude-3-haiku-20240307"
)
```

This couples the rerank model to the agent's operational model — the model the agent
uses for all its reasoning and task execution. The consequences:

**Silent quality degradation for cost-optimised agents.** An agent provisioned with
a cheap/fast model (e.g. `claude-haiku-4-5`) uses that same model for reranking.
Reranking is a reasoning task — it requires the model to assess semantic relevance
across multiple results and return a coherent ordering. Haiku is adequate but the
quality is lower than a dedicated rerank model or a stronger reasoning model.

**Inconsistent rerank quality across the fleet.** Two agents with identical memory
contents but different `domain_md` model configurations will produce different rerank
orderings for the same query. Rerank quality becomes a function of agent configuration
rather than a platform-level guarantee.

**No operator control.** There is no way to set a platform-wide rerank model floor
without modifying every agent's `domain_md`. If a better rerank model becomes
available, it cannot be adopted centrally.

The research finding that motivates this: the industry is converging on a multimodel
strategy where different models are selected for different task types — a strong
reasoning model for planning/evaluation, a fast model for execution. Reranking is
an evaluation task. It should use the best available model for that task, not
whatever model the agent happens to be configured with.

---

## 2. Fix

Add a `rerank_model` setting to `platform/app/config.py` with a sensible default.
`_llm_rerank` reads from settings instead of `domain_md`.

```python
# platform/app/config.py
class Settings(BaseSettings):
    ...
    rerank_model: str = "anthropic/claude-3-haiku-20240307"
```

```python
# platform/app/qortia/recall.py — _llm_rerank
async def _llm_rerank(
    query: str,
    results: list[RecallResult],
    agent: AgentIdentity,
) -> list[RecallResult]:
    if not results:
        return results
    try:
        litellm_key = await get_litellm_key(str(agent.tenant_id))
        model = settings.rerank_model          # ← was: read from domain_md
        ...
```

The `domain_md` fetch inside `_llm_rerank` is removed entirely. The function no
longer needs a DB round-trip to `auth.agents` — it only needs the LiteLLM key.

---

## 3. Why This Is Safe

- `settings.rerank_model` defaults to `"anthropic/claude-3-haiku-20240307"` — the
  same model that was the fallback in the old code. Behaviour is identical for any
  agent that was already using the default.
- Agents configured with a stronger model (e.g. `claude-opus-4`) were previously
  using that model for reranking. After this change they use `settings.rerank_model`.
  This is a deliberate trade-off: platform consistency over per-agent model
  inheritance. The operator can set `rerank_model` to a stronger model if needed.
- Removes one DB query per rerank call (the `auth.agents` fetch for `domain_md`).
  Minor latency improvement.
- No schema change. No migration. No API contract change.

---

## 4. Files Affected

| File | Change |
|---|---|
| `platform/app/config.py` | Add `rerank_model: str` setting with default |
| `platform/app/qortia/recall.py` | `_llm_rerank`: remove `domain_md` fetch, use `settings.rerank_model` |

---

## 5. Test Gates

| Gate | What to verify |
|---|---|
| Unit test — `test_recall_pipeline.py` | `_llm_rerank` uses `settings.rerank_model`, not agent's `domain_md` model |
| Unit test — `test_config.py` | `rerank_model` has correct default value |
| Platform unit tests | `cd platform && python3 -m pytest tests/unit/ -q` — 798/798 |
| Recall eval | `evals/run_reh.py` — Recall@5 ≥ 0.95, MRR ≥ 0.86 (must not regress) |

---

## 6. Future Extension

Once `rerank_model` is a platform setting, it can be promoted to a per-tenant
override (stored in Vault or a `tenant_settings` table) without touching agent
`domain_md`. This is the correct layering: platform default → tenant override →
no agent-level override (reranking is a platform concern, not an agent concern).

---

## 7. Related

- `_llm_rerank` definition: `recall.py`
- `settings` object: `platform/app/config.py`
- ADR-059 (LiteLLM master key / model routing) — relevant context for model
  selection at the platform layer
