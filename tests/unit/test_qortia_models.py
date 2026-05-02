"""
Unit tests — Qortia Pydantic models (Parts 11, 13)
Pure validation, no I/O.
"""

import pytest
from pydantic import ValidationError

from app.qortia.models import (
    IMPORTANCE,
    MemoryItem,
    RecallRequest,
    RememberRequest,
)


# ── IMPORTANCE dict ──────────────────────────────────────────


def test_importance_covers_all_types() -> None:
    assert set(IMPORTANCE.keys()) == {
        "episodic",
        "experiential",
        "mental_model",
        "decision",
        "lesson",
        "short_term",
    }


def test_importance_values_in_range() -> None:
    for t, v in IMPORTANCE.items():
        assert 0.0 <= v <= 1.0, f"{t} importance {v} out of range"


def test_importance_ordering() -> None:
    # lesson > decision > mental_model > experiential > episodic
    assert IMPORTANCE["lesson"] > IMPORTANCE["decision"]
    assert IMPORTANCE["decision"] > IMPORTANCE["mental_model"]
    assert IMPORTANCE["mental_model"] > IMPORTANCE["experiential"]
    assert IMPORTANCE["experiential"] > IMPORTANCE["episodic"]


# ── MemoryItem ───────────────────────────────────────────────


def test_memory_item_valid() -> None:
    item = MemoryItem(type="episodic", content="something happened")
    assert item.type == "episodic"


def test_memory_item_empty_content_rejected() -> None:
    with pytest.raises(ValidationError, match="empty"):
        MemoryItem(type="episodic", content="")


def test_memory_item_whitespace_only_content_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(type="episodic", content="   ")


def test_memory_item_invalid_type_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(type="org_chart", content="test")  # type: ignore[arg-type]


def test_memory_item_metadata_array_rejected() -> None:
    # Pydantic v2 rejects list at type level — error says "valid dictionary"
    with pytest.raises(ValidationError):
        MemoryItem(type="episodic", content="test", metadata=["a", "b"])  # type: ignore[arg-type]


def test_memory_item_metadata_scalar_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(type="episodic", content="test", metadata="string")  # type: ignore[arg-type]


def test_memory_item_metadata_dict_accepted() -> None:
    item = MemoryItem(type="decision", content="chose X", metadata={"reason": "cost"})
    assert item.metadata == {"reason": "cost"}


def test_memory_item_no_extra_fields() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(type="episodic", content="test", importance=0.9)  # type: ignore[call-arg]


# ── RememberRequest ──────────────────────────────────────────


def test_remember_request_empty_array_rejected() -> None:
    with pytest.raises(ValidationError, match="empty"):
        RememberRequest(memories=[])


def test_remember_request_valid_batch() -> None:
    req = RememberRequest(
        memories=[
            MemoryItem(type="episodic", content="a"),
            MemoryItem(type="lesson", content="b"),
        ]
    )
    assert len(req.memories) == 2


# ── RecallRequest ────────────────────────────────────────────


def test_recall_request_defaults() -> None:
    req = RecallRequest(query="what happened")
    assert req.scope == "all"
    assert req.type is None
    assert req.rerank is False
    assert req.entities is None


def test_recall_request_empty_query_rejected() -> None:
    with pytest.raises(ValidationError, match="empty"):
        RecallRequest(query="")


def test_recall_request_whitespace_query_rejected() -> None:
    with pytest.raises(ValidationError):
        RecallRequest(query="   ")


def test_recall_request_invalid_scope_rejected() -> None:
    with pytest.raises(ValidationError):
        RecallRequest(query="test", scope="universe")  # type: ignore[arg-type]


def test_recall_request_invalid_type_rejected() -> None:
    with pytest.raises(ValidationError):
        RecallRequest(query="test", type="org_chart")  # type: ignore[arg-type]


def test_recall_request_empty_entities_list_treated_as_null() -> None:
    req = RecallRequest(query="test", entities=[])
    assert req.entities is None


def test_recall_request_empty_string_entity_rejected() -> None:
    with pytest.raises(ValidationError):
        RecallRequest(query="test", entities=["AuthService", ""])


def test_recall_request_valid_entities() -> None:
    req = RecallRequest(query="test", entities=["AuthService", "Scout"])
    assert req.entities == ["AuthService", "Scout"]


def test_recall_request_all_valid_scopes() -> None:
    for scope in ("private", "org", "knowledge", "all", "archive"):
        req = RecallRequest(query="test", scope=scope)  # type: ignore[arg-type]
        assert req.scope == scope


def test_recall_request_all_valid_types() -> None:
    for t in (
        "episodic",
        "experiential",
        "mental_model",
        "decision",
        "lesson",
        "short_term",
    ):
        req = RecallRequest(query="test", type=t)  # type: ignore[arg-type]
        assert req.type == t


# ── MemoryItem short_term / ttl_seconds ──────────────────────


def test_short_term_memory_requires_ttl_seconds() -> None:
    with pytest.raises(ValidationError, match="ttl_seconds"):
        MemoryItem(type="short_term", content="user is reviewing Q3 report")


def test_short_term_memory_zero_ttl_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(type="short_term", content="context", ttl_seconds=0)


def test_short_term_memory_negative_ttl_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(type="short_term", content="context", ttl_seconds=-60)


def test_short_term_memory_valid() -> None:
    item = MemoryItem(
        type="short_term", content="user is on payment flow", ttl_seconds=3600
    )
    assert item.ttl_seconds == 3600


def test_ttl_seconds_on_non_short_term_rejected() -> None:
    with pytest.raises(ValidationError, match="ttl_seconds"):
        MemoryItem(type="episodic", content="something happened", ttl_seconds=60)


def test_short_term_importance_is_lowest() -> None:
    assert IMPORTANCE["short_term"] < IMPORTANCE["episodic"]


# ── E1: lang field tests ───────────────────────────────────────────


def test_memory_item_lang_defaults_to_en() -> None:
    item = MemoryItem(type="episodic", content="something happened")
    assert item.lang == "en"


def test_memory_item_lang_preserved() -> None:
    item = MemoryItem(
        type="episodic", content="\u0915\u0941\u091b \u0939\u0941\u0906", lang="hi"
    )
    assert item.lang == "hi"


def test_remember_org_request_lang_defaults_to_en() -> None:
    from app.qortia.models import RememberOrgRequest

    req = RememberOrgRequest(type="handoff", title="done", content="finished task")
    assert req.lang == "en"


def test_remember_org_request_lang_preserved() -> None:
    from app.qortia.models import RememberOrgRequest

    req = RememberOrgRequest(
        type="handoff",
        title="done",
        content="\u0915\u093e\u092e \u092a\u0942\u0930\u093e",
        lang="hi",
    )
    assert req.lang == "hi"


def test_recall_request_lang_none_by_default() -> None:
    from app.qortia.models import RecallRequest

    req = RecallRequest(query="what happened")
    assert req.lang is None


def test_recall_request_lang_preserved() -> None:
    from app.qortia.models import RecallRequest

    req = RecallRequest(query="\u0915\u094d\u092f\u093e \u0939\u0941\u0906", lang="hi")
    assert req.lang == "hi"


def test_knowledge_ingest_request_lang_defaults_to_en() -> None:
    from app.qortia.models import KnowledgeIngestRequest

    req = KnowledgeIngestRequest(
        source_type="note", source_path="doc.md", content="some content"
    )
    assert req.lang == "en"


def test_lang_filter_clause_none_returns_empty() -> None:
    from app.qortia.recall import _lang_filter_clause

    clause, params = _lang_filter_clause(None, param=5)
    assert clause == ""
    assert params == []


def test_lang_filter_clause_with_lang_returns_parameterised() -> None:
    from app.qortia.recall import _lang_filter_clause

    clause, params = _lang_filter_clause("hi", param=5)
    assert clause == "AND lang = $5"
    assert params == ["hi"]


def test_weekly_summary_dominant_lang_majority_hindi() -> None:
    from collections import Counter

    handoffs = [{"lang": "hi"}, {"lang": "hi"}, {"lang": "en"}]
    lang_counts: Counter[str] = Counter(h["lang"] for h in handoffs if h.get("lang"))
    dominant = lang_counts.most_common(1)[0][0] if lang_counts else "en"
    assert dominant == "hi"


def test_weekly_summary_dominant_lang_no_lang_field_defaults_en() -> None:
    from collections import Counter

    handoffs = [{"title": "done"}, {"title": "done"}]
    lang_counts: Counter[str] = Counter(h["lang"] for h in handoffs if h.get("lang"))
    dominant = lang_counts.most_common(1)[0][0] if lang_counts else "en"
    assert dominant == "en"


# ── E2: Stanza NER routing tests ───────────────────────────────────────────


def test_extract_entities_english_uses_spacy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """English lang routes to spaCy, not Stanza."""
    from unittest.mock import MagicMock
    import app.qortia.knowledge as kmod

    mock_doc = MagicMock()
    mock_ent = MagicMock()
    mock_ent.text = "OpenAI"
    mock_ent.label_ = "ORG"
    mock_doc.ents = [mock_ent]
    mock_nlp = MagicMock(return_value=mock_doc)
    monkeypatch.setattr(kmod, "_nlp", mock_nlp)

    result = kmod.extract_entities("OpenAI released GPT-4", lang="en")
    assert result == ["OpenAI"]
    mock_nlp.assert_called_once()


def test_extract_entities_hindi_uses_indic_spacy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hindi lang routes to Indic spaCy pipeline."""
    from unittest.mock import MagicMock
    import app.qortia.knowledge as kmod

    mock_ent = MagicMock()
    mock_ent.text = (
        "\u0928\u0930\u0947\u0902\u0926\u094d\u0930 \u092e\u094b\u0926\u0940"
    )
    mock_ent.label_ = "PER"
    mock_doc = MagicMock()
    mock_doc.ents = [mock_ent]
    mock_pipeline = MagicMock(return_value=mock_doc)
    monkeypatch.setitem(kmod._indic_pipelines, "xx_ent_wiki_sm", mock_pipeline)

    result = kmod.extract_entities(
        "\u0928\u0930\u0947\u0902\u0926\u094d\u0930 \u092e\u094b\u0926\u0940 \u092e\u0941\u0902\u092c\u0908 \u0917\u090f",
        lang="hi",
    )
    assert (
        "\u0928\u0930\u0947\u0902\u0926\u094d\u0930 \u092e\u094b\u0926\u0940" in result
    )
    mock_pipeline.assert_called_once()


def test_extract_entities_unsupported_lang_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported lang (kn) returns [] without raising."""
    from unittest.mock import MagicMock
    import app.qortia.knowledge as kmod

    # kn is not in INDIC_NER_LANGS, falls through to spaCy
    mock_doc = MagicMock()
    mock_doc.ents = []  # spaCy finds nothing for Kannada text
    mock_nlp = MagicMock(return_value=mock_doc)
    monkeypatch.setattr(kmod, "_nlp", mock_nlp)

    result = kmod.extract_entities("some kannada text", lang="kn")
    assert result == []
