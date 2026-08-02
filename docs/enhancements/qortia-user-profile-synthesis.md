---
kind: enhancement
owner: platform
last_reviewed: 2026-05-30
status: post-mvp
---

# Qortia: Stakeholder Profile Synthesis for Dedicated-Mode AI Employees

**Status:** Post-MVP — not yet implemented. See "When to Implement" below.
**Scope:** `platform/app/qortia/remember.py`, `platform/app/qortia/recall.py`,
`platform/app/qortia/models.py`, `the agent runtime/scripts/write_user.py`
**ADR required:** Yes — new LLM call pattern, new API endpoints, profile schema decision
**Depends on:** `docs/enhancements/platform-agent-mode.md` (#70) — `user_profiles`
table and `user_ref` on memories must exist first
**GitHub issue:** #71
**Research source:** Honcho (plastic-labs/honcho) — deriver architecture and
`get_context` pattern. `docs/strategy/00-product-vision.md` — Vidya/Munim/health
navigator scenarios.

---

## When to Implement (Trigger Conditions)

**Classification: Post-MVP.** Do **not** build speculatively. Implement only when
**all** of the following hold:

1. **GTM trigger** — dedicated-mode (employee-tier) agents become a committed
   product motion: at least one pilot/paying deployment of a 1:1 named-human agent
   (e.g. the tutor / support-rep / health-navigator scenarios in
   `docs/strategy/00-product-vision.md`). While every agent is assistant/shared-mode,
   there is no `user_ref` to synthesise against — the feature has no input.
2. **Hard prerequisite** — agent-mode (#70) has shipped to production: the
   `user_profiles` table and the `user_ref` column on `hindsight_memories` exist and
   are being populated. This enhancement cannot start before that.
3. **Telemetry signal** — dedicated-mode agents show the problem this solves:
   unbounded boot-context growth, or repeated re-recall of the same user facts per
   session (observable in recall logs). If boot context stays small, defer.

**Do not implement if:** the platform has no dedicated-mode agents in production, or
#70 has not landed. Until then this is a design on file, not work in flight.

**GitHub issue:** #71 (open, labelled `post-mvp`).

---

## 1. The Problem

For dedicated-mode agents, the value proposition is that the agent *knows you* —
it remembers that Aman struggles with quadratic formulas, that Priya prefers
discounts over refunds, that this patient is on a specific medication regimen.

Without profile synthesis, an dedicated-mode agent accumulates raw episodic memories
tagged with `user_ref` but has no consolidated model of the user. Every session
starts from scratch — the agent must re-read dozens of episodic memories to
reconstruct context. This is:

- **Slow** — boot context grows unboundedly with interaction count
- **Noisy** — raw episodic memories contain redundant and contradictory observations
- **Fragile** — a single missed recall query loses important user context

The Honcho framework (plastic-labs/honcho) solves this with a "deriver" that runs
async after each interaction to extract user model updates and merge them into a
persistent profile. The profile is then injected at session start via `get_context`.

the platform should implement this pattern natively in Qortia rather than adopting
Honcho as a dependency (which would add a third database, cross-service auth
complexity, and application-level-only tenant isolation).

---

## 2. Design

### 2.1 Profile Synthesis Trigger

After every `PROFILE_SYNTHESIS_THRESHOLD` (default: 5) episodic memories tagged
with a given `user_ref` are written for an dedicated-mode agent, a background
synthesis pass runs for that `(agent_id, user_ref)` pair as a fire-and-forget
`asyncio.create_task` from `remember()`.

```python
PROFILE_SYNTHESIS_THRESHOLD = 5


async def _maybe_synthesise_profile(
    agent_id: UUID,
    tenant_id: UUID,
    user_ref: str,
) -> None:
    """
    Fire-and-forget profile synthesis. Called from remember() after
    PROFILE_SYNTHESIS_THRESHOLD user-tagged episodic writes.
    Never raises — synthesis failure is non-fatal.
    """
    try:
        # Fetch last 20 user-tagged episodic memories
        async with tenant_transaction(get_main_pool(), tenant_id, agent_id) as conn:
            memories = await conn.fetch(
                """
                SELECT content, created_at FROM hindsight_memories
                WHERE agent_id = $1 AND user_ref = $2
                  AND type = 'episodic' AND tier = 'active'
                ORDER BY created_at DESC LIMIT 20
                """,
                agent_id,
                user_ref,
            )
            existing = await conn.fetchrow(
                "SELECT profile, summary FROM user_profiles WHERE agent_id = $1 AND user_ref = $2",
                agent_id,
                user_ref,
            )

        existing_profile = existing["profile"] if existing else {}
        existing_summary = existing["summary"] if existing else ""

        # Single LLM call: update profile and summary
        updated_profile, updated_summary = await _run_profile_synthesis(
            memories=[r["content"] for r in memories],
            existing_profile=existing_profile,
            existing_summary=existing_summary,
            tenant_id=tenant_id,
        )

        # Embed the summary
        litellm_key = await get_litellm_key(str(tenant_id))
        embedding = await _get_embedding(updated_summary, litellm_key)

        async with tenant_transaction(get_main_pool(), tenant_id, agent_id) as conn:
            await conn.execute(
                """
                INSERT INTO user_profiles
                    (tenant_id, agent_id, user_ref, profile, summary,
                     summary_embedding, interaction_count, last_interaction_at)
                VALUES ($1, $2, $3, $4, $5, $6, 1, now())
                ON CONFLICT (agent_id, user_ref) DO UPDATE SET
                    profile             = EXCLUDED.profile,
                    summary             = EXCLUDED.summary,
                    summary_embedding   = EXCLUDED.summary_embedding,
                    interaction_count   = user_profiles.interaction_count + 1,
                    last_interaction_at = now(),
                    updated_at          = now()
                """,
                tenant_id,
                agent_id,
                user_ref,
                json.dumps(updated_profile),
                updated_summary,
                str(embedding),
            )
        logger.info(
            {
                "event": "user_profile_synthesised",
                "agent_id": str(agent_id),
                "user_ref": user_ref,
            }
        )
    except Exception as exc:
        logger.warning(
            {
                "event": "user_profile_synthesis_failed",
                "agent_id": str(agent_id),
                "user_ref": user_ref,
                "error": str(exc),
            }
        )
```

### 2.2 Synthesis Prompt

The LLM call produces two outputs: an updated `profile` JSONB and a 3–5 sentence
`summary` in plain prose. The profile schema is agent-defined — the LLM populates
it based on what it observes. No fixed schema is enforced at the platform level.

```python
PROFILE_SYNTHESIS_SYSTEM = """
You are a user modeling assistant. Given recent observations about a user and an
existing user profile, produce an updated profile and a concise summary.

Profile format: JSON object with keys appropriate to what you observe.
Common keys: preferences, learning_profile, context, interaction_patterns.
Only include keys for which you have evidence. Never infer from a single mention.

Summary: 3-5 sentences in third person. Exhaustive within the evidence.
Include temporal qualifiers where relevant. Prefer newer observations over older.
"""
```

### 2.3 Profile Injection at Boot / Session Start

`GET /v1/context` for dedicated-mode agents accepts an optional `user_ref` query
param. When provided and a `user_profiles` row exists:

```python
if agent_mode == "assistant" and user_ref:
    profile_row = await conn.fetchrow(
        "SELECT summary FROM user_profiles WHERE agent_id = $1 AND user_ref = $2",
        agent_id,
        user_ref,
    )
    if profile_row:
        context["user_profile_summary"] = profile_row["summary"]
```

`the agent runtime/scripts/write_user.py` is rewritten for dedicated-mode agents: instead of
writing a placeholder `USER.md`, it fetches the profile summary from the context
endpoint and writes it as structured context.

### 2.4 Recall Boost

When `user_ref` is provided in a `recall` request for an dedicated-mode agent,
the user profile summary is embedded and injected as a synthetic top-ranked result
at the head of the recall response — not as a memory row, but as a context item
with `type: "user_profile"`. This ensures the agent always has the user model
available without relying on the recall pipeline to surface it.

### 2.5 Profile Deletion

`forget()` extended: when called with `user_ref` (no `memory_id`), deletes all
`hindsight_memories` rows tagged with that `user_ref` AND the `user_profiles` row
for `(agent_id, user_ref)`. This is the GDPR-compliant "forget this user" operation.

New endpoint: `DELETE /v1/profiles/{user_ref}` — explicit profile deletion without
touching memories.

---

## 3. Files Affected

| File | Change |
|---|---|
| `platform/app/qortia/remember.py` | `_maybe_synthesise_profile`, synthesis trigger in `remember()`, profile deletion in `forget()` |
| `platform/app/qortia/recall.py` | User profile summary injected as top-ranked context item when `user_ref` provided |
| `platform/app/qortia/remember.py` | `get_context()` returns `user_profile_summary` for dedicated-mode agents |
| `platform/app/qortia/models.py` | `UserProfileResponse`, `ProfileSynthesisConfig` |
| `the agent runtime/scripts/write_user.py` | Rewrite for assistant-mode: fetch and write synthesised profile summary |
| New: `platform/app/qortia/profiles.py` | `GET /v1/profiles/{user_ref}`, `DELETE /v1/profiles/{user_ref}` |

---

## 4. Test Gates

| Gate | What to verify |
|---|---|
| Unit — `test_remember.py` | Synthesis triggered after `PROFILE_SYNTHESIS_THRESHOLD` user-tagged episodic writes |
| Unit — `test_remember.py` | Synthesis NOT triggered for employee-mode agents |
| Unit — `test_remember.py` | Synthesis failure is non-fatal — `remember()` returns success |
| Integration | Full lifecycle: 5 episodic writes with `user_ref` → synthesis runs → `user_profiles` row populated |
| Integration | `GET /v1/context?user_ref=X` returns `user_profile_summary` after synthesis |
| Integration | `DELETE /v1/profiles/{user_ref}` removes profile and all tagged memories |
| Platform unit tests | `cd platform && python3 -m pytest tests/unit/ -q` — 798/798 |
| Full stack health | `python3 scripts/local_agents.py` — 21/21 checks |

---

## 5. Known Constraints

**LiteLLM cost.** Each synthesis call invokes the LiteLLM `/chat/completions`
endpoint. At `PROFILE_SYNTHESIS_THRESHOLD = 5`, a user who sends 50 messages
triggers 10 synthesis calls. At Haiku pricing this is negligible. The threshold
is configurable via `settings`.

**Profile schema is agent-defined.** The platform does not validate the JSONB
structure. Different agent types will produce different profile schemas — a teacher
agent's profile looks different from a health navigator's. This is intentional.

**Synthesis is eventually consistent.** The profile is updated async after writes,
not synchronously. A user who sends 4 messages has no profile yet. The 5th message
triggers synthesis, which completes in the background. The profile is available
from the 6th session onward.

**Privacy.** `user_ref` is an opaque external identifier. The platform never stores
PII directly — the agent's LLM extracts what it observes from conversation content.
Operators are responsible for ensuring their agents do not extract and store
regulated PII in profile JSONB.
