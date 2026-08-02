---
kind: adr
status: accepted
owner: platform
last_reviewed: 2026-08-02
---

# ADR-003 — LiteLLM as multi-tenant gateway; engine stays swappable

- **Status:** Accepted (2026-08-02)
- **Deciders:** founder
- **Related:** ADR-002 (configurable embeddings)

## Context

Qortia needs real embeddings in OSS setups and clean multi-tenant attribution
(cost, rate limits, traces). Running the model *inside* the Qortia process
(sentence-transformers / FlagEmbedding) would couple deploy to a specific
engine and lose a shared auth/budget/trace seam. Talking to Ollama alone works
for single-tenant dogfood but has no virtual keys or per-tenant OTel routing.

## Decision

1. **Two layers:** LiteLLM Proxy = gateway; Ollama / TEI / vLLM = engine.
2. **Compose stack** (`docker-compose.yml`) ships db + ollama + litellm for
   local dogfood. `QORTIA_LITELLM_URL` points at the gateway (`:4000`).
3. **Per-tenant keys (v1.1 seam):** `QORTIA_LITELLM_TENANT_KEYS` is a JSON map
   `tenant_id → virtual_key`. `get_litellm_key(tenant_id)` returns the mapped
   key or falls back to `QORTIA_LITELLM_API_KEY` (master/shared). No Vault.
4. **Trace attribution on every embed:** `embed_text` sends OpenAI-compatible
   `user=<tenant_id>` and `metadata={"qortia.tenant_id": ...}` so LiteLLM OTel
   v2 / callbacks can attribute spans without Qortia owning exporter wiring.
5. **Platform probe** uses `user=platform` / `qortia.tenant_id=platform`.

## Consequences

**Good:** Swap engines without touching Qortia; multi-tenant budgets/traces live
where they belong (gateway); dogfood path is one `just stack-up`.

**Trade-off:** Operators must provision LiteLLM virtual keys (or accept the
shared fallback key). Full Admin-API auto-provisioning of virtual keys is a
later increment — the env map is enough to start isolating tenants.
