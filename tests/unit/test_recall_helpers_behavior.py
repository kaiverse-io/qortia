from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from qortia import config
from qortia.models import RecallResult
from qortia.recall_helpers import (
    _apply_char_budget,
    _bm25_normalization,
    _cosine,
    _entity_filter_clause,
    _keyword_boost,
    _lang_filter_clause,
    _mmr,
    _resolve_max_chars,
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


def test_apply_char_budget_drops_whole_results_past_the_cap() -> None:
    """/v1/recall had no response-size cap at all before this — measured at
    38,961 chars/call for 5.5% precision (agnova's scale eval against real
    FiQA data). Same policy as _budget_memories: rank order, drop whole
    entries, never slice one."""
    first = _result(rid="first", content="x" * 100)
    second = _result(rid="second", content="y" * 100)
    third = _result(rid="third", content="z" * 100)

    out = _apply_char_budget([first, second, third], max_chars=150)

    assert [r.id for r in out] == ["first"]


def test_apply_char_budget_keeps_the_top_result_even_over_budget() -> None:
    """An empty list is indistinguishable from 'no relevant memories' — a
    single result longer than max_chars is still returned rather than
    dropped, unlike _budget_memories/_fill_budget which can legitimately
    return nothing (they render one flat string; this returns a list)."""
    only = _result(rid="only", content="x" * 500)

    out = _apply_char_budget([only], max_chars=10)

    assert [r.id for r in out] == ["only"]


def test_apply_char_budget_stops_at_first_non_fit_not_a_smaller_later_one() -> None:
    """Matches _budget_memories' `used > 0` guard exactly: once a result
    doesn't fit, budgeting stops there — it does not skip ahead looking for
    a smaller later result that would fit, since results arrive ranked and
    skipping ahead would return a less relevant result in place of a more
    relevant one just because it happens to be shorter."""
    big = _result(rid="big", content="x" * 100)
    small = _result(rid="small", content="y" * 10)

    out = _apply_char_budget([big, small], max_chars=50)

    assert [r.id for r in out] == ["big"]


def test_apply_char_budget_empty_input_returns_empty() -> None:
    assert _apply_char_budget([], max_chars=100) == []


def test_resolve_max_chars_none_falls_back_to_configured_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None is the common case — no caller passes max_chars — and must
    resolve to the server default, not 'no cap': recall() having no size cap
    by default at all is the gap this whole change closes."""
    monkeypatch.setattr(config.settings, "recall_default_max_chars", 8000)

    assert _resolve_max_chars(None) == 8000


def test_resolve_max_chars_explicit_value_overrides_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, "recall_default_max_chars", 8000)

    assert _resolve_max_chars(500) == 500


def test_resolve_max_chars_explicit_non_positive_is_returned_as_is() -> None:
    """A caller passing 0 (or negative) is opting out of any cap — the same
    'non-positive means unbounded' convention _apply_char_budget's own
    caller in recall.py checks for; _resolve_max_chars doesn't coerce this to
    the default, it's a distinct, deliberate caller choice from omitting the
    field entirely."""
    assert _resolve_max_chars(0) == 0
    assert _resolve_max_chars(-1) == -1


def test_rrf_fuse_uses_configured_k(monkeypatch: pytest.MonkeyPatch) -> None:
    """RRF's k constant moved from a hardcoded module constant to
    config.settings.recall_rrf_k — confirm _rrf_fuse actually reads it live
    (not a stale value captured at import time) by checking it changes the
    ranking, not just that it runs.

    _rrf_fuse scores each id by 1/(k+pos) summed over how many times *that
    id* repeats in the input (not classic positional RRF across separate
    ranked lists — pos here is a per-id repeat counter). final_score then
    multiplies that raw score by importance, so a frequent-but-lower-
    importance id and a single-appearance-but-higher-importance id trade off
    against each other as k changes: small k makes raw RRF magnitude (and so
    frequency) dominate; as k grows, every raw score shrinks toward zero at
    a rate set by count (~count/k), so the importance multiplier decides
    instead. frequent (5 appearances, importance 0.3) vs single (1
    appearance, importance 0.95) — worked by hand: k=1 gives raw·imp of
    0.435 vs 0.475 (single wins); k=1000 gives 0.00150 vs 0.00095 (frequent
    wins, since 5×0.3 > 1×0.95 dominates once position differences vanish).
    """
    frequent = _result(rid="frequent", importance=0.3)
    single = _result(rid="single", importance=0.95)
    fused = [frequent, frequent, frequent, frequent, frequent, single]

    monkeypatch.setattr(config.settings, "recall_rrf_k", 1)
    assert _rrf_fuse(fused)[0].id == "single"

    monkeypatch.setattr(config.settings, "recall_rrf_k", 1000)
    assert _rrf_fuse(fused)[0].id == "frequent"


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
