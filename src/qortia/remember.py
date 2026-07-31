from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException

from qortia.auth import AgentIdentity, require_agent
from qortia.common import assert_agent_active
from qortia.db import get_main_pool, tenant_transaction
from qortia.knowledge import extract_entities_with_types
from qortia.models import (
    IMPORTANCE,
    ContextMemories,
    ContextResponse,
    ForgetRequest,
    ForgetResponse,
    MemoryEntry,
    RememberOrgRequest,
    RememberOrgResponse,
    RememberRequest,
    RememberResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# BCP-47 codes supported by xx_ent_wiki_sm routing in knowledge.py
_SUPPORTED_INDIC_LANGS = frozenset({"hi", "bn", "ta", "te", "mr"})


try:
    from langdetect import DetectorFactory as _DetectorFactory
    from langdetect import detect

    _DetectorFactory.seed = 0  # deterministic output across calls
except ImportError:  # pragma: no cover
    detect = None


def _detect_lang(text: str) -> str:
    """Detect BCP-47 language code from text. Returns 'en' on failure or unknown lang."""
    if detect is None:
        return "en"
    try:
        detected = str(detect(text))
        return detected.split("-")[0].lower()
    except Exception:
        return "en"


# ── Extraction Prompt Improvements (#75) ────────────────────────────────────
# These constants are the canonical extraction guidance for any caller
# (agent runtime, reflect cycle) that produces memories for Qortia.
# They are prompt fragments — concatenate into the system/user message
# before asking an LLM to extract structured memories from conversation.


def build_temporal_grounding_instruction(reference_time: datetime | None = None) -> str:
    """Build temporal grounding instruction with the given reference time.

    If reference_time is None, uses current UTC time.
    """
    ts = (reference_time or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    return f"""Current date and time: {ts}

When extracting memories, resolve all relative temporal references against this
timestamp. Store resolved dates, not relative references.

Examples:
- "last Tuesday" → resolve to the actual date (e.g., "on 2026-05-20 (Tuesday)")
- "last week" → resolve to the actual week (e.g., "during the week of 2026-05-12")
- "in March" → "in March 2026" (use current year if ambiguous)
- "yesterday" → resolve to the actual date
- "recently" → keep as-is if no specific timeframe can be inferred

If the temporal reference is ambiguous or cannot be resolved, store it as-is.
Do NOT invent dates — only resolve when the reference is unambiguous."""


ATTRIBUTION_INSTRUCTION = """When extracting episodic memories from a conversation, prefix
each memory with the appropriate attribution:

[User] — something the user explicitly stated or expressed
[Observed] — something you (the agent) observed about the user or situation
[Third-party] — something mentioned about a person or entity not in the conversation

Examples:
- "[User] Prefers concise responses over detailed explanations."
- "[Observed] Struggles with abstract concepts — responds better to concrete examples."
- "[Third-party] Alice (project lead) approved the architecture change on 2026-05-10."

Use [Observed] when you are inferring from behaviour, not from explicit statements.
Use [User] only for direct statements or clear expressions of preference."""


NEGATIVE_EXTRACTION_INSTRUCTION = """DO NOT extract the following — they produce noise
memories with no durable value:

- Pronouns or references without clear antecedents ("he said", "she did", "it worked")
- Abstract concepts without grounding ("success", "progress", "improvement", "things")
- Bare relational terms without qualification ("the manager", "the client", "the team")
  → Use names or specific identifiers instead
- Generic action nouns ("the meeting", "the task", "the thing", "the issue")
  → Only extract if the specific meeting/task/issue is named or described
- Status-only observations with no durable signal:
  ("done", "ok", "noted", "understood", "will do", "sounds good")
- Single-turn questions that reveal no durable preference or fact
  ("what time is it?", "can you help me?", "how do I do X?")
- Greetings and pleasantries ("hello", "thanks", "goodbye", "have a nice day")
- Task instructions that do not reveal a durable fact about the user
  ("please format this as a table", "summarise the above")
- Observations that are true of every interaction and carry no specific information
  ("the user asked a question", "I provided an answer")
- Content that is already captured in a more specific memory type
  (do not duplicate a decision as an episodic memory)"""


def build_extraction_prompt(
    memory_type: str,
    reference_time: datetime | None = None,
) -> str:
    """Build the full extraction prompt for a given memory type.

    Returns the combined prompt instructions for temporal grounding,
    attribution (episodic only), and negative examples (episodic/experiential).
    """
    parts: list[str] = []

    # Temporal grounding applies to all extraction types
    parts.append(build_temporal_grounding_instruction(reference_time))

    # Attribution only for episodic memories
    if memory_type == "episodic":
        parts.append(ATTRIBUTION_INSTRUCTION)

    # Negative examples for episodic and experiential
    if memory_type in ("episodic", "experiential"):
        parts.append(NEGATIVE_EXTRACTION_INSTRUCTION)

    return "\n\n".join(parts)


def _extract_valid_from(metadata: dict | None) -> datetime | None:  # type: ignore[type-arg]
    """Extract valid_from timestamp from memory metadata.

    The caller (agent runtime) may pass valid_from as an ISO-8601 string
    in metadata["valid_from"] to anchor when a fact became true.
    Returns None if not present or unparseable — the DB column default (now()) applies.
    """
    if not metadata:
        return None
    raw = metadata.get("valid_from")
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        logger.warning({"event": "invalid_valid_from", "raw_value": str(raw)[:100]})
        return None


async def _fetch_agent_clearance(agent_id: object, tenant_id: object) -> tuple[int, str]:
    """Fetch clearance_order and division for an agent. Returns (2, 'all') on failure.

    Only used by background tasks (reflect.py's idle-reflection trigger) that
    have raw agent/tenant UUIDs but no AgentIdentity — HTTP endpoints already
    get clearance_order/division resolved once at auth time (see qortia.auth).
    """
    async with get_main_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT a.clearance_level, a.division, cl.level_order
            FROM qortia_agents a
            JOIN qortia_clearance_levels cl ON cl.level_name = a.clearance_level
            WHERE a.id = $1 AND a.tenant_id = $2
            """,
            agent_id,
            tenant_id,
        )
    if row is None:
        return 2, "all"
    return int(row["level_order"]), str(row["division"])


@router.post("/v1/remember", response_model=RememberResponse)
async def remember(
    body: RememberRequest,
    agent: AgentIdentity = Depends(require_agent),  # noqa: B008
    x_work_order_id: str | None = Header(default=None, alias="X-Work-Order-Id"),
) -> RememberResponse:
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)

        ids = []
        episodic_count = 0

        for mem in body.memories:
            # Auto-detect language when agent did not explicitly set it (defaults to "en")
            effective_lang = mem.lang
            if effective_lang == "en" and len(mem.content) >= 20:
                detected = _detect_lang(mem.content)
                if detected != "en":
                    effective_lang = detected
                    logger.info(
                        {
                            "event": "lang_auto_detected",
                            "detected": detected,
                            "qortia.tenant_id": str(agent.tenant_id),
                        }
                    )
            try:
                entities = extract_entities_with_types(mem.content, lang=effective_lang)
            except Exception as exc:
                logger.warning({"event": "ner_extraction_failed", "error": str(exc)})
                entities = []

            expires_at = (
                datetime.now(UTC) + timedelta(seconds=mem.ttl_seconds)
                if mem.type == "short_term" and mem.ttl_seconds
                else None
            )

            content_hash = hashlib.sha256(mem.content.encode()).hexdigest()

            # G4: exact content hash dedup for episodic/experiential within 24h
            if mem.type in ("episodic", "experiential"):
                existing = await conn.fetchval(
                    """
                    SELECT id FROM hindsight_memories
                    WHERE agent_id = $1
                      AND content_hash = $2
                      AND tier = 'active'
                      AND created_at > now() - interval '24 hours'
                    LIMIT 1
                    """,
                    agent.agent_id,
                    content_hash,
                )
                if existing:
                    ids.append(str(existing))
                    continue

            # Extract valid_from from metadata (#75 temporal grounding)
            valid_from = _extract_valid_from(mem.metadata)

            # Build stored metadata — preserve source_message_index for attribution
            stored_metadata: dict = dict(mem.metadata) if mem.metadata else {}  # type: ignore[type-arg]
            # Strip valid_from from stored metadata (it lives in its own column)
            stored_metadata.pop("valid_from", None)

            row_id = await conn.fetchval(
                """
                INSERT INTO hindsight_memories
                    (tenant_id, agent_id, type, content, importance,
                     source_task_id, metadata, entities, expires_at, lang,
                     content_hash, valid_from)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                        COALESCE($12, now()))
                RETURNING id
            """,
                agent.tenant_id,
                agent.agent_id,
                mem.type,
                mem.content,
                IMPORTANCE[mem.type],
                mem.source_task_id,
                json.dumps(stored_metadata),
                json.dumps(entities),
                expires_at,
                effective_lang,
                content_hash,
                valid_from,
            )
            ids.append(str(row_id))

            await conn.execute(
                """
                INSERT INTO memory_history
                    (tenant_id, agent_id, operation, target_table, target_id,
                     content_hash, metadata)
                VALUES ($1, $2, 'remember', 'hindsight_memories', $3, $4, $5)
            """,
                agent.tenant_id,
                agent.agent_id,
                row_id,
                hashlib.sha256(mem.content.encode()).hexdigest(),
                json.dumps({"type": mem.type}),
            )

            if mem.type == "episodic":
                episodic_count += 1

        if episodic_count > 0:
            await conn.execute(
                """
                UPDATE qortia_agents
                SET reflection_counter = reflection_counter + $1, updated_at = now()
                WHERE id = $2
            """,
                episodic_count,
                agent.agent_id,
            )

    logger.info(
        {
            "event": "remember_written",
            "agent_id": str(agent.agent_id),
            "qortia.tenant_id": str(agent.tenant_id),
            "memory_count": len(ids),
            "work_order_id": x_work_order_id,
        }
    )

    return RememberResponse(ids=ids)


@router.post("/v1/remember-org", response_model=RememberOrgResponse)
async def remember_org(
    body: RememberOrgRequest,
    agent: AgentIdentity = Depends(require_agent),  # noqa: B008
) -> RememberOrgResponse:
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)

        # Role check fires second (Q80) — enum check already done by Pydantic
        if body.type in ("process", "decision_log"):
            role = await conn.fetchval(
                "SELECT role FROM qortia_agents WHERE id = $1 AND tenant_id = $2",
                agent.agent_id,
                agent.tenant_id,
            )
            if role != "chief":
                raise HTTPException(403, f"Only chief agent can write type '{body.type}'")

        try:
            entities = extract_entities_with_types(body.content, lang=body.lang)
        except Exception as exc:
            logger.warning({"event": "ner_extraction_failed", "error": str(exc)})
            entities = []

        if body.type == "handoff":
            row_id = await conn.fetchval(
                """
                INSERT INTO org_memory
                    (tenant_id, type, title, content, author_id, entities, lang,
                     min_clearance, audience)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """,
                agent.tenant_id,
                body.type,
                body.title,
                body.content,
                agent.agent_id,
                json.dumps(entities),
                body.lang,
                body.min_clearance,
                body.audience,
            )
        else:
            row_id = await conn.fetchval(
                """
                INSERT INTO org_memory
                    (tenant_id, type, title, content, author_id, entities, lang,
                     min_clearance, audience, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, now())
                ON CONFLICT (tenant_id, type, title)
                WHERE type IN ('process', 'decision_log')
                DO UPDATE SET
                    content      = EXCLUDED.content,
                    author_id    = EXCLUDED.author_id,
                    entities     = EXCLUDED.entities,
                    lang         = EXCLUDED.lang,
                    min_clearance = EXCLUDED.min_clearance,
                    audience     = EXCLUDED.audience,
                    updated_at   = now()
                RETURNING id
            """,
                agent.tenant_id,
                body.type,
                body.title,
                body.content,
                agent.agent_id,
                json.dumps(entities),
                body.lang,
                body.min_clearance,
                body.audience,
            )

        await conn.execute(
            """
            INSERT INTO memory_history
                (tenant_id, agent_id, operation, target_table, target_id, content_hash, metadata)
            VALUES ($1, $2, 'remember_org', 'org_memory', $3, $4, $5)
        """,
            agent.tenant_id,
            agent.agent_id,
            row_id,
            hashlib.sha256(body.content.encode()).hexdigest(),
            json.dumps({"type": body.type, "author_id": str(agent.agent_id)}),
        )

    return RememberOrgResponse(id=str(row_id))


@router.post("/v1/forget", response_model=ForgetResponse)
async def forget(
    body: ForgetRequest,
    agent: AgentIdentity = Depends(require_agent),  # noqa: B008
) -> ForgetResponse:
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)

        hm = await conn.fetchrow(
            "SELECT id, agent_id, type, content FROM hindsight_memories"
            " WHERE id = $1 AND agent_id = $2",
            body.id,
            agent.agent_id,
        )
        om = None
        if hm is None:
            om = await conn.fetchrow(
                "SELECT id, author_id, type, content FROM org_memory"
                " WHERE id = $1 AND tenant_id = $2",
                body.id,
                agent.tenant_id,
            )

        if hm is None and om is None:
            raise HTTPException(404, "Memory not found")

        if hm is not None:
            if hm["agent_id"] != agent.agent_id:
                raise HTTPException(403, "Cannot delete another agent's memory")
            table, row, content = "hindsight_memories", hm, hm["content"]
        else:
            assert (  # noqa: S101
                om is not None
            )  # guarded by `if hm is None and om is None` raise above
            mem_type = om["type"]
            if mem_type in ("org_chart", "weekly_summary"):
                raise HTTPException(403, f"Cannot delete type '{mem_type}'")
            if mem_type == "handoff" and om["author_id"] != agent.agent_id:
                raise HTTPException(403, "Cannot delete another agent's handoff")
            if mem_type in ("process", "decision_log"):
                role = await conn.fetchval(
                    "SELECT role FROM qortia_agents WHERE id = $1 AND tenant_id = $2",
                    agent.agent_id,
                    agent.tenant_id,
                )
                if role != "chief":
                    raise HTTPException(403, f"Only chief can delete type '{mem_type}'")
            table, row, content = "org_memory", om, om["content"]

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        await conn.execute(f"DELETE FROM {table} WHERE id = $1", row["id"])  # noqa: S608

        # Obsidian Layer: Cleanup entity graph
        await conn.execute(
            """
            UPDATE qortia_entities
            SET linked_memory_ids = array_remove(linked_memory_ids, $1),
                updated_at = now()
            WHERE $1 = ANY(linked_memory_ids)
              AND tenant_id = $2
            """,
            row["id"],
            agent.tenant_id,
        )
        # Cross-memory link cleanup (16i)
        await conn.execute(
            """
            DELETE FROM memory_links
            WHERE (source_id = $1 OR target_id = $1)
              AND tenant_id = $2
            """,
            row["id"],
            agent.tenant_id,
        )
        await conn.execute(
            """
            INSERT INTO memory_history
                (tenant_id, agent_id, operation, target_table, target_id, content_hash, metadata)
            VALUES ($1, $2, 'forget', $3, $4, $5, $6)
        """,
            agent.tenant_id,
            agent.agent_id,
            table,
            row["id"],
            content_hash,
            json.dumps({"type": row["type"]}),
        )

    return ForgetResponse(id=str(row["id"]))


@router.get("/v1/context", response_model=ContextResponse)
async def get_context(agent: AgentIdentity = Depends(require_agent)) -> ContextResponse:  # noqa: B008
    clearance_order, agent_division = agent.clearance_order, agent.division
    async with tenant_transaction(
        get_main_pool(),
        agent.tenant_id,
        agent.agent_id,
        memory_clearance_order=clearance_order,
        agent_division=agent_division,
    ) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)

        org_chart = await conn.fetch(
            "SELECT title, content FROM org_memory WHERE type = 'org_chart' ORDER BY created_at ASC"
        )
        processes = await conn.fetch(
            "SELECT title, content FROM org_memory WHERE type = 'process' ORDER BY created_at ASC"
        )
        handoffs = await conn.fetch(
            "SELECT title, content FROM org_memory WHERE type = 'handoff' "
            "ORDER BY created_at DESC LIMIT 5"
        )
        ws = await conn.fetchrow(
            "SELECT title, content FROM org_memory WHERE type = 'weekly_summary' "
            "ORDER BY created_at DESC LIMIT 1"
        )
        mental_models = await conn.fetch(
            """
            SELECT content, importance FROM hindsight_memories
            WHERE agent_id = $1 AND type = 'mental_model' AND is_consolidated = true
            ORDER BY importance DESC LIMIT 20
        """,
            agent.agent_id,
        )
        decisions = await conn.fetch(
            """
            SELECT content FROM hindsight_memories
            WHERE agent_id = $1 AND type = 'decision'
            ORDER BY created_at DESC LIMIT 15
        """,
            agent.agent_id,
        )
        lessons = await conn.fetch(
            """
            SELECT content, importance FROM hindsight_memories
            WHERE agent_id = $1 AND type = 'lesson' AND is_consolidated = true
            ORDER BY importance DESC LIMIT 20
        """,
            agent.agent_id,
        )

    return ContextResponse(
        org_chart=[MemoryEntry(title=r["title"], content=r["content"]) for r in org_chart],
        processes=[MemoryEntry(title=r["title"], content=r["content"]) for r in processes],
        handoffs=[MemoryEntry(title=r["title"], content=r["content"]) for r in handoffs],
        weekly_summary=MemoryEntry(title=ws["title"], content=ws["content"]) if ws else None,
        memories=ContextMemories(
            mental_models=[
                MemoryEntry(content=r["content"], importance=r["importance"]) for r in mental_models
            ],
            decisions=[MemoryEntry(content=r["content"]) for r in decisions],
            lessons=[
                MemoryEntry(content=r["content"], importance=r["importance"]) for r in lessons
            ],
        ),
    )
