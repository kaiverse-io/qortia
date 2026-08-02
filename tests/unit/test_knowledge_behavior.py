"""Unit tests for knowledge ingestion helpers and section splitting."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from qortia.auth import AgentIdentity


def test_estimate_tokens_scales_word_count() -> None:
    from qortia.knowledge import estimate_tokens

    assert estimate_tokens("one two three four") == int(4 * 1.3)


def test_split_into_sections_without_headings_splits_paragraphs() -> None:
    from qortia.knowledge import split_into_sections

    content = " ".join(["word"] * 60)
    sections = split_into_sections(content)
    assert len(sections) >= 1
    assert sections[0]["heading"] == ""


def test_split_into_sections_with_markdown_headings() -> None:
    from qortia.knowledge import split_into_sections

    content = (
        "## Architecture\n"
        + " ".join(["PostgreSQL stores durable Qortia memories today"] * 12)
        + "\n\n### Retrieval\n"
        + " ".join(["Hybrid recall combines lexical and vector search paths"] * 12)
    )
    sections = split_into_sections(content)
    headings = {section["heading"] for section in sections}
    assert "Architecture" in headings
    assert "Retrieval" in headings


def test_paragraph_split_chunks_large_text() -> None:
    from qortia.knowledge import _paragraph_split

    paragraph = " ".join(["token"] * 800)
    chunks = _paragraph_split(f"{paragraph}\n\n{paragraph}", title="Chunk")
    assert len(chunks) >= 1
    assert all(chunk["heading"] == "Chunk" for chunk in chunks)


def test_extract_index_fields_english(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia import knowledge as kmod

    mock_ent = MagicMock()
    mock_ent.text = "Qortia"
    mock_ent.label_ = "ORG"
    mock_chunk = MagicMock()
    mock_chunk.text = "retrieval pipeline"
    mock_sent = MagicMock()
    mock_sent.text = "Qortia stores memories."
    mock_doc = MagicMock()
    mock_doc.ents = [mock_ent]
    mock_doc.sents = [mock_sent]
    mock_doc.noun_chunks = [mock_chunk]
    monkeypatch.setattr(kmod, "get_nlp", lambda: MagicMock(return_value=mock_doc))

    fields = kmod.extract_index_fields("Architecture", "Qortia stores memories in PostgreSQL")
    assert "Qortia" in fields["index_entities"]
    assert fields["index_summary"]


def test_extract_index_fields_degrades_without_spacy(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia import knowledge as kmod

    monkeypatch.setattr(kmod, "get_nlp", lambda: None)
    fields = kmod.extract_index_fields(
        "Architecture",
        "Qortia stores memories in PostgreSQL. Hybrid recall fuses BM25 and vectors.",
    )
    assert fields["index_summary"]
    assert "PostgreSQL" in fields["index_summary"] or "Qortia" in fields["index_summary"]
    assert fields["index_entities"] == "[]"


def test_extract_entities_with_types_without_spacy(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia import knowledge as kmod

    monkeypatch.setattr(kmod, "get_nlp", lambda: None)
    assert kmod.extract_entities_with_types("Alice reviewed AuthService") == []


def test_load_spacy_model_skips_missing_indic(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia import knowledge as kmod

    fake_spacy = MagicMock()
    fake_spacy.load.return_value = MagicMock(name="en_nlp")
    monkeypatch.setitem(__import__("sys").modules, "spacy", fake_spacy)
    monkeypatch.setattr(
        kmod, "_get_indic_pipeline", MagicMock(side_effect=OSError("no indic model"))
    )
    kmod._nlp = None
    kmod.load_spacy_model()
    assert kmod._nlp is not None
    kmod._get_indic_pipeline.assert_called_once_with("hi")


def test_build_weekly_summary_formats_handoffs() -> None:
    from qortia.knowledge import build_weekly_summary

    handoffs = [
        {
            "created_at": datetime(2026, 8, 1, tzinfo=UTC),
            "agent_name": "Scout",
            "content": "Shipped retrieval improvements",
        },
        {
            "created_at": datetime(2026, 8, 2, tzinfo=UTC),
            "agent_name": "Ops",
            "content": "Scaled PostgreSQL cluster",
        },
    ]
    summary = build_weekly_summary(handoffs)
    assert "Scout" in summary
    assert "PostgreSQL cluster" in summary


@pytest.mark.asyncio
async def test_ingest_knowledge_rejects_non_chief(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia.knowledge import ingest_knowledge
    from qortia.models import KnowledgeIngestRequest

    conn = MagicMock()
    conn.fetchval = AsyncMock(side_effect=["engineer", None])
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("qortia.knowledge.get_main_pool", lambda: MagicMock())
    monkeypatch.setattr("qortia.knowledge.tenant_transaction", lambda *_a, **_k: ctx)
    monkeypatch.setattr("qortia.knowledge.assert_agent_active", AsyncMock())

    agent = AgentIdentity(agent_id=uuid4(), tenant_id=uuid4())
    body = KnowledgeIngestRequest(
        source_type="note",
        source_path="docs/guide.md",
        content=" ".join(["Qortia knowledge ingestion requires chief role today"] * 8),
    )

    with pytest.raises(HTTPException) as exc:
        await ingest_knowledge(body, agent)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_delete_knowledge_returns_deleted_chunk_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.knowledge import delete_knowledge

    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value="chief")
    conn.execute = AsyncMock(return_value="DELETE 3")
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("qortia.knowledge.get_main_pool", lambda: MagicMock())
    monkeypatch.setattr("qortia.knowledge.tenant_transaction", lambda *_a, **_k: ctx)
    monkeypatch.setattr("qortia.knowledge.assert_agent_active", AsyncMock())

    agent = AgentIdentity(agent_id=uuid4(), tenant_id=uuid4())
    result = await delete_knowledge("docs/guide.md", agent)
    assert result["chunks_deleted"] == 3


@pytest.mark.asyncio
async def test_run_weekly_summary_cycle_fetches_active_tenants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qortia.knowledge import _run_weekly_summary_cycle

    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[])
    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire_ctx
    monkeypatch.setattr("qortia.knowledge.get_main_pool", lambda: pool)

    await _run_weekly_summary_cycle()
    conn.fetch.assert_awaited_once()
