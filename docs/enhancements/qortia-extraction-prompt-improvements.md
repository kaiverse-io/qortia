---
kind: enhancement
owner: platform
last_reviewed: 2026-05-18
status: implemented
---

# Qortia: Extraction Prompt Improvements

**Status:** Implemented — 2026-05-30
**Scope:** `platform/app/qortia/remember.py`, `platform/app/qortia/reflect.py`
**ADR required:** No — prompt engineering change only; no schema, API contract,
or dependency change
**Depends on:** None (but complements `docs/enhancements/qortia-memory-quality.md`
G2/G3 — content floor and dedup)
**GitHub issue:** #75
**Research source:** `docs/research/memory-framework-comparison.md` §3 (Mem0
extraction pipeline) and §12 (Adopt from Mem0 table). `docs/research/zep-graphiti-review.md`
§3 (Graphiti negative-example extraction lists). Mem0 V3 `ADDITIVE_EXTRACTION_PROMPT`
(~800 lines) and Graphiti `extract_nodes.py` (~500 lines).

---

## 1. The Problem

Qortia's per-type extraction prompts produce two categories of noise that degrade
recall quality:

**Temporally ungrounded memories.** An agent that writes "I fixed the bug last
Tuesday" stores that string verbatim. Three weeks later, "last Tuesday" is
meaningless. The agent cannot reason about when this happened relative to other
events. Mem0's extraction prompt resolves relative temporal references against a
`REFERENCE_TIME` (the current timestamp) before storing.

**Unattributed observations.** Qortia's prompts treat all content as first-person
agent observations. In a conversation, "the user said they prefer concise responses"
and "I observed the user struggles with abstract concepts" are stored identically.
Mem0 distinguishes `[User]`, `[Observed]`, and `[Third-party]` attributions —
critical for dedicated-mode agents building stakeholder profiles (#71).

**Noise entities and low-signal memories.** Graphiti's extraction prompts include
extensive "NEVER extract" lists that prevent the most common failure modes: pronouns
without antecedents, abstract concepts without grounding, bare relational terms,
generic action nouns, status-only observations. Qortia has no equivalent, leading
to memories like "done", "ok", "task completed" that pass the 5-word floor (#59
G2) but carry no durable semantic signal.

These are prompt engineering changes — no schema migration, no new dependencies,
no API contract change. Lowest-risk, highest-signal-per-effort item in the batch.

---

## 2. Design

### 2.1 Temporal grounding

Inject `REFERENCE_TIME` into all extraction prompts. The platform passes
`datetime.utcnow().isoformat()` as a formatted string in the prompt context.

```python
# remember.py — in the extraction prompt construction
REFERENCE_TIME = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

TEMPORAL_GROUNDING_INSTRUCTION = f"""
Current date and time: {REFERENCE_TIME}

When extracting memories, resolve all relative temporal references against this
timestamp. Store resolved dates, not relative references.

Examples:
- "last Tuesday" → "on 2026-07-08 (Tuesday)"
- "last week" → "during the week of 2026-07-07"
- "in March" → "in March 2026"
- "yesterday" → "on 2026-07-13"
- "recently" → keep as-is if no specific timeframe can be inferred

If the temporal reference is ambiguous or cannot be resolved, store it as-is.
"""
```

This is a prompt addition — the existing extraction logic is unchanged. The
resolved timestamp becomes part of the memory content string. No schema change.

### 2.2 Multi-speaker attribution

For `episodic` memories extracted from conversation, add attribution prefixes to
the extraction prompt:

```python
ATTRIBUTION_INSTRUCTION = """
When extracting episodic memories from a conversation, prefix each memory with
the appropriate attribution:

[User] — something the user explicitly stated or expressed
[Observed] — something you (the agent) observed about the user or situation
[Third-party] — something mentioned about a person or entity not in the conversation

Examples:
- "[User] Prefers concise responses over detailed explanations."
- "[Observed] Struggles with abstract concepts — responds better to concrete examples."
- "[Third-party] Alice (project lead) approved the architecture change on 2026-07-10."

Use [Observed] when you are inferring from behaviour, not from explicit statements.
Use [User] only for direct statements or clear expressions of preference.
"""
```

This is only added to the `episodic` extraction prompt — not to `mental_model`,
`lesson`, or `decision` prompts, which are agent-internal consolidations.

### 2.3 Negative-example checklist

Add a "DO NOT extract" section to the `episodic` and `experiential` extraction
prompts:

```python
NEGATIVE_EXTRACTION_INSTRUCTION = """
DO NOT extract the following — they produce noise memories with no durable value:

- Pronouns or references without clear antecedents ("he said", "she did", "it worked")
- Abstract concepts without grounding ("success", "progress", "improvement", "things")
- Bare relational terms without qualification ("the manager", "the client", "the team")
  → Use names or specific identifiers instead
- Generic action nouns ("the meeting", "the task", "the thing", "the issue")
  → Only extract if the specific meeting/task/issue is named or described
- Status-only observations with no durable signal:
  ("done", "ok", "noted", "understood", "will do", "sounds good")
- Observations that are true of every interaction and carry no specific information
  ("the user asked a question", "I provided an answer")
- Content that is already captured in a more specific memory type
  (do not duplicate a decision as an episodic memory)
"""
```

This directly addresses the noise entity and low-signal memory problems identified
in `qortia-memory-quality.md` G2/G3, and complements the 5-word content floor.

### 2.4 Reflection prompt improvement

The reflection cycle's consolidation prompt (`reflect.py`) gains the temporal
grounding instruction. When consolidating episodic memories into `mental_model`
or `lesson` types, the LLM should preserve resolved timestamps from the source
memories rather than re-introducing relative references.

```python
REFLECTION_TEMPORAL_INSTRUCTION = f"""
Current date and time: {REFERENCE_TIME}

When synthesising memories, preserve specific dates from source memories.
Do not convert resolved dates back to relative references.
"""
```

---

## 3. Files Affected

| File | Change |
|---|---|
| `platform/app/qortia/remember.py` | Add `TEMPORAL_GROUNDING_INSTRUCTION`, `ATTRIBUTION_INSTRUCTION`, `NEGATIVE_EXTRACTION_INSTRUCTION` to episodic/experiential extraction prompts |
| `platform/app/qortia/reflect.py` | Add `REFLECTION_TEMPORAL_INSTRUCTION` to consolidation prompt |

No migration. No new dependencies. No API contract change.

---

## 4. Test Gates

| Gate | What to verify |
|---|---|
| Eval regression | `evals/run_reh.py` before and after — Recall@5 ≥ 0.95, MRR ≥ 0.86 |
| Eval quality | Noise entity rate (entities extracted per memory) should decrease vs baseline |
| Manual canary | Write 10 episodic memories with relative temporal references ("last Tuesday", "last week") — verify resolved timestamps appear in stored content |
| Manual canary | Write episodic memories from a conversation — verify `[User]`/`[Observed]` prefixes appear |
| Manual canary | Attempt to write status-only content ("done", "ok") — verify not extracted as episodic memory |
| Platform unit tests | `cd platform && python3 -m pytest tests/unit/ -q` — 798/798 |
| Full stack health | `python3 scripts/local_agents.py` — 21/21 checks |

---

## 5. Known Constraints

**Prompt changes affect all agents immediately.** Unlike schema changes, prompt
changes take effect on the next `remember()` call after deployment. There is no
migration window. The eval regression gate is the primary safety check.

**Attribution prefixes are content, not metadata.** `[User]`, `[Observed]`,
`[Third-party]` are stored as part of the memory content string — not as a
separate column. This is intentional: it keeps the schema unchanged and makes
the attribution visible in recall results. A future enhancement could extract
attribution as a structured field if needed.

**Temporal grounding is best-effort.** The LLM may not always resolve ambiguous
temporal references correctly. "Recently" and "a while ago" are stored as-is.
The instruction is additive — it improves grounding for clear cases without
breaking ambiguous ones.

**Negative-example list is not exhaustive.** The list covers the most common
failure modes observed in the eval dataset. New failure modes will be added as
they are observed in production. The list should be treated as a living document
within the prompt string.
