from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import re
from typing import Any
from uuid import UUID

from app.auth.middleware import require_agent
from app.auth.models import AgentIdentity
from app.background.leader import LOCK_KEY_WEEKLY_SUMMARY, try_acquire_leader
from app.db import get_main_pool, tenant_transaction
from app.qortia.common import assert_agent_active
from app.qortia.models import KnowledgeIngestRequest
from fastapi import APIRouter, Depends, HTTPException

logger = logging.getLogger(__name__)

# ── spaCy singleton ──────────────────────────────────────────

_nlp = None

# Single canonical label set for English NER — used by all extraction functions.
EN_ENTITY_LABELS = frozenset({"ORG", "PERSON", "PRODUCT", "GPE", "NORP", "FAC", "WORK_OF_ART"})
HEADING_PATTERN = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

# ── Indic NER routing (spaCy) ────────────────────────────────

# hi/bn/ta/te/mr -> xx_ent_wiki_sm (multilingual)
_INDIC_MODEL: dict[str, str] = {
    "hi": "xx_ent_wiki_sm",
    "bn": "xx_ent_wiki_sm",
    "ta": "xx_ent_wiki_sm",
    "te": "xx_ent_wiki_sm",
    "mr": "xx_ent_wiki_sm",
}
INDIC_NER_LANGS = frozenset(_INDIC_MODEL)
# Maps xx_ent_wiki_sm labels → qortia_entities_entity_type_check allowed values.
# MISC has no meaningful equivalent — drop it rather than pollute the graph.
_INDIC_LABEL_MAP: dict[str, str] = {"PER": "PERSON", "ORG": "ORG", "LOC": "GPE"}

_SUPPORTED_LANGS = INDIC_NER_LANGS | {"en"}

_indic_pipelines: dict[str, Any] = {}  # lang -> spaCy Language


def _get_indic_pipeline(lang: str) -> Any:
    """Load and cache the spaCy pipeline for an Indic language. Fails fast on missing model."""
    if lang not in _indic_pipelines:
        model = _INDIC_MODEL[lang]
        import spacy

        try:
            _indic_pipelines[lang] = spacy.load(model)
            logger.info({"event": "spacy_model_loaded", "model": model, "lang": lang})
        except OSError as exc:
            logger.error(
                {
                    "event": "spacy_model_load_failed",
                    "model": model,
                    "lang": lang,
                    "error": str(exc),
                }
            )
            raise
    return _indic_pipelines[lang]


def load_spacy_model() -> None:
    global _nlp
    import spacy

    _nlp = spacy.load("en_core_web_sm")
    logger.info({"event": "spacy_model_loaded", "model": "en_core_web_sm"})
    # Warm up Indic pipeline at startup so first memory write doesn't pay the load cost.
    _get_indic_pipeline("hi")


def get_nlp() -> object | None:  # spaCy Language — avoid hard dep on spacy type stubs
    return _nlp


def extract_entities(text: str, lang: str = "en") -> list[str]:
    """
    Extract NER entity texts. Routes to Indic spaCy models or en_core_web_sm.
    Best-effort — caller wraps in try/except and falls back to [].
    Returns text only (label stripped) for backward-compatible callers.
    """
    if lang not in _SUPPORTED_LANGS:
        logger.warning({"event": "ner_lang_unsupported", "lang": lang, "fallback": "en"})
    if lang in INDIC_NER_LANGS:
        doc = _get_indic_pipeline(lang)(text)
        return list(dict.fromkeys(ent.text for ent in doc.ents if ent.label_ in _INDIC_LABEL_MAP))[
            :20
        ]
    nlp = get_nlp()
    if nlp is None:
        logger.warning({"event": "ner_model_not_loaded", "lang": lang})
        return []
    doc = nlp(text)  # type: ignore[operator]
    return list(dict.fromkeys(ent.text for ent in doc.ents if ent.label_ in EN_ENTITY_LABELS))[:20]


def extract_entities_with_types(text: str, lang: str = "en") -> list[tuple[str, str]]:
    """
    Extract NER entities with their label. Routes to Indic spaCy models or en_core_web_sm.
    Returns list of (entity_text, label). Best-effort — caller wraps in try/except.
    """
    if lang not in _SUPPORTED_LANGS:
        logger.warning({"event": "ner_lang_unsupported", "lang": lang, "fallback": "en"})
    if lang in INDIC_NER_LANGS:
        doc = _get_indic_pipeline(lang)(text)
        seen: dict[str, str] = {}
        for ent in doc.ents:
            mapped = _INDIC_LABEL_MAP.get(ent.label_)
            if mapped and ent.text not in seen:
                seen[ent.text] = mapped
        return list(seen.items())[:20]
    doc = get_nlp()(text)  # type: ignore[operator]
    seen_en: dict[str, str] = {}
    for ent in doc.ents:
        if ent.label_ in EN_ENTITY_LABELS and ent.text not in seen_en:
            seen_en[ent.text] = ent.label_
    return list(seen_en.items())[:20]


# ── Section splitting ────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def split_into_sections(content: str) -> list[dict[str, str]]:
    matches = list(HEADING_PATTERN.finditer(content))

    sections: list[dict[str, str]] = []

    if not matches:
        raw = _paragraph_split(content, title="")
        return [s for s in raw if estimate_tokens(s["text"]) >= 50]

    pre = content[: matches[0].start()].strip()
    if pre and estimate_tokens(pre) >= 50:
        sections.append({"heading": "Introduction", "text": pre})

    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_text = content[start:end].strip()

        if estimate_tokens(section_text) > 2000:
            sections.extend(_paragraph_split(section_text, title=heading_text))
        elif estimate_tokens(section_text) >= 50:
            sections.append({"heading": heading_text, "text": section_text})
        elif sections:
            sections[-1]["text"] += "\n\n" + section_text

    return sections


def _paragraph_split(text: str, title: str) -> list[dict[str, str]]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict] = []  # type: ignore[type-arg]
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if estimate_tokens(candidate) > 2000 and current:
            chunks.append({"heading": title, "text": current})
            current = para
        else:
            current = candidate
    if current:
        chunks.append({"heading": title, "text": current})
    return chunks


# ── PageIndex extraction ─────────────────────────────────────


def extract_index_fields(heading: str, text: str, lang: str = "en") -> dict[str, Any]:
    if lang in INDIC_NER_LANGS:
        doc = _get_indic_pipeline(lang)(text)
        entities = list(
            dict.fromkeys(ent.text for ent in doc.ents if ent.label_ in _INDIC_LABEL_MAP)
        )[:10]
        sentences = list(doc.sents)
        noun_chunks: list[str] = []
    else:
        nlp = get_nlp()
        doc = nlp(text)  # type: ignore[operator]
        entities = list(
            dict.fromkeys(ent.text for ent in doc.ents if ent.label_ in EN_ENTITY_LABELS)
        )[:10]
        sentences = list(doc.sents)
        noun_chunks = list(
            dict.fromkeys(
                chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4
            )
        )[:10]

    summary = " ".join(s.text.strip() for s in sentences[:2])
    questions = ([heading] + noun_chunks) if heading else noun_chunks

    return {
        "index_summary": summary,
        "index_entities": json.dumps(entities),
        "index_questions": json.dumps(questions),
    }


# ── Router ───────────────────────────────────────────────────

router = APIRouter()


@router.post("/v1/knowledge")
async def ingest_knowledge(
    body: KnowledgeIngestRequest,
    agent: AgentIdentity = Depends(require_agent),  # noqa: B008
) -> dict:  # type: ignore[type-arg]
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)
        role = await conn.fetchval(
            "SELECT role FROM auth.agents WHERE id = $1 AND tenant_id = $2",
            agent.agent_id,
            agent.tenant_id,
        )
        if role != "chief":
            raise HTTPException(403, "Only chief agent can ingest knowledge")

    sections = split_into_sections(body.content)
    incoming_hashes = [hashlib.sha256(s["text"].encode()).hexdigest() for s in sections]

    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        existing = await conn.fetch(
            """
            SELECT chunk_index, content_hash FROM org_knowledge
            WHERE tenant_id = $1 AND source_path = $2
            ORDER BY chunk_index ASC
        """,
            agent.tenant_id,
            body.source_path,
        )

        existing_hashes = [r["content_hash"] for r in existing]

        if len(existing_hashes) == len(incoming_hashes) and existing_hashes == incoming_hashes:
            return {
                "sections_created": 0,
                "sections_deduped": len(sections),
                "source_path": body.source_path,
            }

        if existing:
            await conn.execute(
                "DELETE FROM org_knowledge WHERE tenant_id = $1 AND source_path = $2",
                agent.tenant_id,
                body.source_path,
            )

        sections_created = 0
        sections_deduped = 0

        for idx, (section, content_hash) in enumerate(zip(sections, incoming_hashes)):  # noqa: B905
            index_fields = extract_index_fields(section["heading"], section["text"], lang=body.lang)

            existing_embedding = await conn.fetchval(
                """
                SELECT embedding FROM org_knowledge
                WHERE tenant_id = $1 AND content_hash = $2
                LIMIT 1
            """,
                agent.tenant_id,
                content_hash,
            )

            await conn.execute(
                """
                INSERT INTO org_knowledge (
                    tenant_id, source_type, source_path, chunk_index,
                    content, content_hash,
                    index_summary, index_questions, index_entities,
                    embedding, author_id, metadata, lang, min_clearance, audience
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """,
                agent.tenant_id,
                body.source_type,
                body.source_path,
                idx,
                section["text"],
                content_hash,
                index_fields["index_summary"],
                index_fields["index_questions"],
                index_fields["index_entities"],
                existing_embedding,
                agent.agent_id,
                json.dumps(body.metadata or {}),
                body.lang,
                body.min_clearance,
                body.audience,
            )

            if existing_embedding is not None:
                sections_deduped += 1
            else:
                sections_created += 1

        await conn.execute(
            """
            INSERT INTO memory_history
                (tenant_id, agent_id, operation, target_table, target_id, metadata)
            VALUES ($1, $2, 'knowledge_ingest', 'org_knowledge', NULL, $3)
        """,
            agent.tenant_id,
            agent.agent_id,
            json.dumps(
                {
                    "source_path": body.source_path,
                    "sections_created": sections_created,
                    "sections_deduped": sections_deduped,
                }
            ),
        )

    return {
        "sections_created": sections_created,
        "sections_deduped": sections_deduped,
        "source_path": body.source_path,
    }


@router.delete("/v1/knowledge/{source_path:path}")
async def delete_knowledge(
    source_path: str,
    agent: AgentIdentity = Depends(require_agent),  # noqa: B008
) -> dict:  # type: ignore[type-arg]
    async with tenant_transaction(get_main_pool(), agent.tenant_id, agent.agent_id) as conn:
        await assert_agent_active(agent.agent_id, agent.tenant_id, conn)
        role = await conn.fetchval(
            "SELECT role FROM auth.agents WHERE id = $1 AND tenant_id = $2",
            agent.agent_id,
            agent.tenant_id,
        )
        if role != "chief":
            raise HTTPException(403, "Only chief agent can delete knowledge")

        result = await conn.execute(
            "DELETE FROM org_knowledge WHERE tenant_id = $1 AND source_path = $2",
            agent.tenant_id,
            source_path,
        )
        chunks_deleted = int(result.split()[-1])

        await conn.execute(
            """
            INSERT INTO memory_history
                (tenant_id, agent_id, operation, target_table, target_id,
                 content_hash, metadata)
            VALUES ($1, $2, 'knowledge_delete', 'org_knowledge', NULL, NULL, $3)
        """,
            agent.tenant_id,
            agent.agent_id,
            json.dumps({"source_path": source_path, "chunks_deleted": chunks_deleted}),
        )

    return {"source_path": source_path, "chunks_deleted": chunks_deleted}


# ── Weekly summary background task ──────────────────────────


def build_weekly_summary(handoffs: list[dict]) -> str:  # type: ignore[type-arg]
    parts = []
    for h in sorted(handoffs, key=lambda x: x["created_at"], reverse=True):
        date = h["created_at"].strftime("%Y-%m-%d")
        agent_name = h.get("agent_name") or "Unknown"
        parts.append(f"[{agent_name} | {date}]\n{h['content'].strip()}")
    return "\n\n---\n\n".join(parts)


async def run_weekly_summary_task() -> None:
    while True:
        await asyncio.sleep(86400)
        async with try_acquire_leader(get_main_pool(), LOCK_KEY_WEEKLY_SUMMARY) as is_leader:
            if is_leader:
                await _run_weekly_summary_cycle()


async def _run_weekly_summary_cycle() -> None:
    async with get_main_pool().acquire() as conn:
        tenants = await conn.fetch(
            "SELECT id, weekly_summary_last_run_at FROM auth.tenants WHERE status = 'active'"
        )

    import hashlib as _hashlib

    for tenant in tenants:
        tenant_id = tenant["id"]
        day_offset = int(_hashlib.md5(str(tenant_id).encode()).hexdigest(), 16) % 7  # noqa: S324
        if datetime.date.today().weekday() != day_offset:
            continue
        await _summarise_tenant(tenant_id, tenant["weekly_summary_last_run_at"])


async def _summarise_tenant(tenant_id: UUID, last_run_at: object) -> None:
    async with get_main_pool().acquire() as conn:
        async with conn.transaction():
            locked = await conn.fetchrow(
                "SELECT id FROM auth.tenants WHERE id = $1 FOR UPDATE SKIP LOCKED",
                tenant_id,
            )
            if not locked:
                return

            if (
                last_run_at
                and (
                    datetime.datetime.now(datetime.UTC) - last_run_at  # type: ignore[operator]
                ).days
                < 7
            ):
                return

            handoffs = await conn.fetch(
                """
                SELECT om.title, om.content, om.created_at, om.lang, a.name AS agent_name
                FROM org_memory om
                LEFT JOIN auth.agents a ON a.id = om.author_id
                WHERE om.tenant_id = $1
                  AND om.type = 'handoff'
                  AND om.created_at > now() - interval '7 days'
                ORDER BY om.created_at DESC
            """,
                tenant_id,
            )

            if len(handoffs) < 3:
                return

            summary_content = build_weekly_summary(list(handoffs))

            from collections import Counter

            lang_counts: Counter[str] = Counter(h["lang"] for h in handoffs if h.get("lang"))
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

            await conn.execute(
                "UPDATE auth.tenants SET weekly_summary_last_run_at = now() WHERE id = $1",
                tenant_id,
            )
