from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from qortia.models import RecallResult
from qortia.recall_helpers import (
    _bm25_normalization,
    _cosine,
    _entity_filter_clause,
    _keyword_boost,
    _lang_filter_clause,
    _mmr,
    _rrf_fuse,
    _sort_by_importance,
    _temporal_filter_clause,
    _to_result,
    _type_filter_clause,
    dynamic_importance,
)


def _result(
    *,
    rid: str | None = None,
    content: str = "memory content about postgres",
    importance: float | None = 0.5,
    score: float = 0.5,
    embedding: list[float] | None = None,
) -> RecallResult:
    result = RecallResult(
        id=rid or str(uuid4()),
        type="episodic",
        scope="private",
        content=content,
        importance=importance,
        created_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
    )
    result._score = score
    result._embedding = embedding or []
    return result


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("short query", 32),
        ("one two three", 32),
        ("one two three four", 0),
    ],
)
def test_bm25_normalization_short_queries_use_length_normalization(
    query: str, expected: int
) -> None:
    assert _bm25_normalization(query) == expected


@pytest.mark.parametrize(
    ("memory_type", "clause", "params"),
    [
        (None, "", []),
        ("decision", "AND type = $3", ["decision"]),
        ("not_valid", "", []),
    ],
)
def test_type_filter_clause_only_accepts_known_memory_types(
    memory_type: str | None, clause: str, params: list[str]
) -> None:
    assert _type_filter_clause(memory_type, 3) == (clause, params)


@pytest.mark.parametrize(
    ("entities", "has_clause"),
    [
        (None, False),
        ([], False),
        (["Postgres", "Qortia"], True),
    ],
)
def test_entity_filter_clause_uses_bound_entity_list(
    entities: list[str] | None, has_clause: bool
) -> None:
    clause, params = _entity_filter_clause(entities, 4)
    assert ("ANY($4::text[])" in clause) is has_clause
    assert params == ([entities] if entities else [])


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        (None, ("", [])),
        ("en", ("AND lang = $5", ["en"])),
        ("hi", ("AND lang = $5", ["hi"])),
    ],
)
def test_lang_filter_clause_is_parameterised(
    lang: str | None, expected: tuple[str, list[str]]
) -> None:
    assert _lang_filter_clause(lang, 5) == expected


def test_temporal_filter_clause_defaults_to_current_facts_only() -> None:
    assert _temporal_filter_clause(None, 6) == ("AND valid_until IS NULL", [])


def test_temporal_filter_clause_builds_point_in_time_bounds() -> None:
    as_of = datetime(2025, 1, 2, tzinfo=UTC)
    clause, params = _temporal_filter_clause(as_of, 6)
    assert "valid_from <= $6" in clause
    assert "(valid_until IS NULL OR valid_until > $6)" in clause
    assert params == [as_of]


@pytest.mark.parametrize(
    ("base", "recalls", "last_seen", "confidence", "expected_range"),
    [
        (0.5, 0, None, 1.0, (0.5, 0.5)),
        (0.5, 10, None, 1.0, (0.7, 0.8)),
        (0.5, 0, datetime.now(UTC) - timedelta(days=1), 1.0, (0.6, 0.8)),
        (0.8, 0, None, 0.5, (0.4, 0.4)),
        (2.0, 99, datetime.now(UTC), 2.0, (1.0, 1.0)),
    ],
)
def test_dynamic_importance_combines_frequency_recency_and_confidence(
    base: float,
    recalls: int,
    last_seen: datetime | None,
    confidence: float,
    expected_range: tuple[float, float],
) -> None:
    score = dynamic_importance(base, recalls, last_seen, confidence)
    assert expected_range[0] <= score <= expected_range[1]


def test_to_result_preserves_temporal_and_ranking_private_attrs() -> None:
    row = {
        "id": uuid4(),
        "type": "decision",
        "content": "team chose pgvector for portable memory",
        "importance": 0.9,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "valid_from": datetime(2025, 12, 1, tzinfo=UTC),
        "valid_until": None,
        "recall_count": 7,
        "last_recalled_at": datetime(2026, 1, 2, tzinfo=UTC),
        "confidence_multiplier": 0.75,
        "rank": 0.42,
    }

    result = _to_result(row, "private")

    assert result.id == str(row["id"])
    assert result.valid_from == row["valid_from"].isoformat()
    assert result.valid_until is None
    assert result._recall_count == 7
    assert result._confidence_multiplier == 0.75
    assert result._score == 0.42


def test_rrf_fuse_deduplicates_and_entity_boosts_linked_results() -> None:
    linked = _result(rid="linked", importance=0.5)
    normal = _result(rid="normal", importance=0.5)

    fused = _rrf_fuse([normal, linked, normal], entity_links={"linked"})
    ids = [r.id for r in fused]
    assert set(ids) == {"linked", "normal"}
    assert len(ids) == 2

    # With equal single occurrences, entity boost should prefer linked.
    fused_once = _rrf_fuse([normal, linked], entity_links={"linked"})
    assert fused_once[0].id == "linked"


def test_sort_by_importance_ranks_frequently_recalled_memories_first() -> None:
    stale = _result(rid="stale", importance=0.5)
    popular = _result(rid="popular", importance=0.5)
    popular._recall_count = 30

    assert _sort_by_importance([stale, popular])[0].id == "popular"


@pytest.mark.parametrize(
    ("query", "content", "score"),
    [
        ("postgres tenant memory", "Postgres stores tenant scoped memory rows", 1.0),
        ("to be", "tiny words are ignored", 0.0),
        ("missing token", "nothing overlaps here", 0.0),
    ],
)
def test_keyword_boost_scores_unique_query_token_overlap(
    query: str, content: str, score: float
) -> None:
    assert _keyword_boost(query, content) == score


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([0.0, 0.0], [1.0, 1.0], 0.0),
    ],
)
def test_cosine_handles_parallel_orthogonal_and_zero_vectors(
    a: list[float], b: list[float], expected: float
) -> None:
    assert _cosine(a, b) == pytest.approx(expected)


def test_mmr_returns_empty_when_no_candidate_reaches_min_score() -> None:
    assert _mmr([1.0, 0.0], [_result(score=0.1, embedding=[1.0, 0.0])]) == []


def test_mmr_prefers_relevant_non_redundant_candidates() -> None:
    first = _result(rid="first", score=0.9, embedding=[1.0, 0.0])
    duplicate = _result(rid="duplicate", score=0.8, embedding=[0.99, 0.01])
    diverse = _result(rid="diverse", score=0.7, embedding=[0.0, 1.0])

    selected = _mmr([1.0, 0.0], [first, duplicate, diverse], dedup_threshold=0.95, k=3)

    assert [r.id for r in selected] == ["first", "diverse"]
