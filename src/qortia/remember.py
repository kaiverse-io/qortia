from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.auth.middleware import require_agent
from app.auth.models import AgentIdentity
from app.qortia.knowledge import extract_entities_with_types
from app.qortia.models import (
    ContextResponse,
    ContextMemories,
    ForgetRequest,
    ForgetResponse,
    IMPORTANCE,
    MemoryEntry,
    RememberOrgRequest,
    RememberOrgResponse,
    RememberRequest,
    RememberResponse,
)
from app.db import get_main_pool, tenant_transaction
from app.qortia.common import assert_agent_active

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/remember", response_model=RememberResponse)
async def remember(
    body: RememberRequest,
    agent: AgentIdentity = Depends(require_agent),
) -> RememberResponse:
    async with tenant_transaction(
        get_main_pool(), agent.tenant_id, agent.agent_id
    ) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)

        ids = []
        episodic_count = 0

        for mem in body.memories:
            try:
                entities = extract_entities_with_types(mem.content)
            except Exception as exc:
                logger.warning({"event": "ner_extraction_failed", "error": str(exc)})
                entities = []

            expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=mem.ttl_seconds)
                if mem.type == "short_term" and mem.ttl_seconds
                else None
            )

            row_id = await conn.fetchval(
                """
                INSERT INTO hindsight_memories
                    (tenant_id, agent_id, type, content, importance,
                     source_task_id, metadata, entities, expires_at)
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
                expires_at,
            )
            ids.append(str(row_id))

            await conn.execute(
                """
                INSERT INTO memory_history
                    (tenant_id, agent_id, operation, target_table, target_id, content_hash, metadata)
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
                UPDATE auth.agents
                SET reflection_counter = reflection_counter + $1, updated_at = now()
                WHERE id = $2
            """,
                episodic_count,
                agent.agent_id,
            )

    return RememberResponse(ids=ids)


@router.post("/v1/remember-org", response_model=RememberOrgResponse)
async def remember_org(
    body: RememberOrgRequest,
    agent: AgentIdentity = Depends(require_agent),
) -> RememberOrgResponse:
    async with tenant_transaction(
        get_main_pool(), agent.tenant_id, agent.agent_id
    ) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)

        # Role check fires second (Q80) — enum check already done by Pydantic
        if body.type in ("process", "decision_log"):
            role = await conn.fetchval(
                "SELECT role FROM auth.agents WHERE id = $1 AND tenant_id = $2",
                agent.agent_id,
                agent.tenant_id,
            )
            if role != "chief":
                raise HTTPException(
                    403, f"Only chief agent can write type '{body.type}'"
                )

        try:
            entities = extract_entities_with_types(body.content)
        except Exception as exc:
            logger.warning({"event": "ner_extraction_failed", "error": str(exc)})
            entities = []

        if body.type == "handoff":
            row_id = await conn.fetchval(
                """
                INSERT INTO org_memory (tenant_id, type, title, content, author_id, entities)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """,
                agent.tenant_id,
                body.type,
                body.title,
                body.content,
                agent.agent_id,
                json.dumps(entities),
            )
        else:
            row_id = await conn.fetchval(
                """
                INSERT INTO org_memory
                    (tenant_id, type, title, content, author_id, entities, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, now())
                ON CONFLICT (tenant_id, type, title)
                WHERE type IN ('process', 'decision_log')
                DO UPDATE SET
                    content    = EXCLUDED.content,
                    author_id  = EXCLUDED.author_id,
                    entities   = EXCLUDED.entities,
                    updated_at = now()
                RETURNING id
            """,
                agent.tenant_id,
                body.type,
                body.title,
                body.content,
                agent.agent_id,
                json.dumps(entities),
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
    agent: AgentIdentity = Depends(require_agent),
) -> ForgetResponse:
    async with tenant_transaction(
        get_main_pool(), agent.tenant_id, agent.agent_id
    ) as conn:
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
            assert (
                om is not None
            )  # guarded by `if hm is None and om is None` raise above
            mem_type = om["type"]
            if mem_type in ("org_chart", "weekly_summary"):
                raise HTTPException(403, f"Cannot delete type '{mem_type}'")
            if mem_type == "handoff" and om["author_id"] != agent.agent_id:
                raise HTTPException(403, "Cannot delete another agent's handoff")
            if mem_type in ("process", "decision_log"):
                role = await conn.fetchval(
                    "SELECT role FROM auth.agents WHERE id = $1 AND tenant_id = $2",
                    agent.agent_id,
                    agent.tenant_id,
                )
                if role != "chief":
                    raise HTTPException(403, f"Only chief can delete type '{mem_type}'")
            table, row, content = "org_memory", om, om["content"]

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        await conn.execute(f"DELETE FROM {table} WHERE id = $1", row["id"])

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
async def get_context(agent: AgentIdentity = Depends(require_agent)) -> ContextResponse:
    async with tenant_transaction(
        get_main_pool(), agent.tenant_id, agent.agent_id
    ) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)

        org_chart = await conn.fetch(
            "SELECT title, content FROM org_memory WHERE type = 'org_chart' ORDER BY created_at ASC"
        )
        processes = await conn.fetch(
            "SELECT title, content FROM org_memory WHERE type = 'process' ORDER BY created_at ASC"
        )
        handoffs = await conn.fetch(
            "SELECT title, content FROM org_memory WHERE type = 'handoff' ORDER BY created_at DESC LIMIT 5"
        )
        ws = await conn.fetchrow(
            "SELECT title, content FROM org_memory WHERE type = 'weekly_summary' ORDER BY created_at DESC LIMIT 1"
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
        org_chart=[
            MemoryEntry(title=r["title"], content=r["content"]) for r in org_chart
        ],
        processes=[
            MemoryEntry(title=r["title"], content=r["content"]) for r in processes
        ],
        handoffs=[
            MemoryEntry(title=r["title"], content=r["content"]) for r in handoffs
        ],
        weekly_summary=MemoryEntry(title=ws["title"], content=ws["content"])
        if ws
        else None,
        memories=ContextMemories(
            mental_models=[
                MemoryEntry(content=r["content"], importance=r["importance"])
                for r in mental_models
            ],
            decisions=[MemoryEntry(content=r["content"]) for r in decisions],
            lessons=[
                MemoryEntry(content=r["content"], importance=r["importance"])
                for r in lessons
            ],
        ),
    )
