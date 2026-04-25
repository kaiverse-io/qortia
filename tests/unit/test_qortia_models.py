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
    for scope in ("private", "org", "knowledge", "all"):
        req = RecallRequest(query="test", scope=scope)  # type: ignore[arg-type]
        assert req.scope == scope


def test_recall_request_all_valid_types() -> None:
    for t in ("episodic", "experiential", "mental_model", "decision", "lesson"):
        req = RecallRequest(query="test", type=t)  # type: ignore[arg-type]
        assert req.type == t
