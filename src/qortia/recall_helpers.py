"""Pure helper functions for the recall pipeline. No I/O or database access."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from app.qortia.models import RecallResult

RRF_K = 60
SEARCH_FETCH_MULTIPLIER = 2
PRIVATE_RESULT_LIMIT = 20
ORG_RESULT_LIMIT = 10
KNOWLEDGE_RESULT_LIMIT = 16

_VALID_MEMORY_TYPES: frozenset[str] = frozenset(
    {"episodic", "experiential", "mental_model", "decision", "lesson", "short_term"}
)

def _bm25_normalization(query: str) -> int:
    """ts_rank_cd normalization: 32 for short queries (≤3 tokens), 0 otherwise."""
    return 32 if len(query.split()) <= 3 else 0

def dynamic_importance(
    base_importance: float,
    recall_count: int,
    last_recalled_at: datetime | None,
    confidence_multiplier: float = 1.0,
) -> float:
    frequency_boost = math.log1p(recall_count) / 10.0
    recency_boost = 0.0
    if last_recalled_at:
        days_since = (datetime.now(timezone.utc) - last_recalled_at).days
        recency_boost = max(0.0, 1.0 - (days_since / 30.0)) * 0.2
    raw = min(1.0, base_importance + frequency_boost + recency_boost)
    return max(0.0, min(1.0, raw * confidence_multiplier))

def _entity_filter_clause(
    entities: list[str] | None,
    base_param: int,
) -> tuple[str, list]:  # type: ignore[type-arg]
    if not entities:
        return "", []
    # Use a subquery to check for entity text match in nested [text, type] pairs.
    # This correctly handles the the platform entity structure.
    clause = f"""
        AND EXISTS (
            SELECT 1 FROM jsonb_array_elements(entities) AS e
            WHERE e->>0 = ANY(${base_param}::text[])
        )
    """
    return clause, [entities]

def _type_filter_clause(
    memory_type: str | None,
    param: int,
) -> tuple[str, list]:  # type: ignore[type-arg]
    """Return a parameterised type filter clause.

    body.type is a Pydantic Literal — already validated — but we never
    interpolate user-supplied strings into SQL. Defence-in-depth whitelist
    ensures the pattern is safe even if validation is relaxed upstream.
    """
    if not memory_type or memory_type not in _VALID_MEMORY_TYPES:
        return "", []
    return f"AND type = ${param}", [memory_type]

def _temporal_filter_clause(
    as_of: object,
    param: int,
) -> tuple[str, list]:  # type: ignore[type-arg]
    """Return a parameterised temporal filter clause.

    No as_of: exclude superseded rows (valid_until IS NULL = currently valid).
    With as_of: point-in-time range — fact was valid at that timestamp.
    """
    if as_of is None:
        return "AND valid_until IS NULL", []
    return (
        f"AND valid_from <= ${param} AND (valid_until IS NULL OR valid_until > ${param})",
        [as_of],
    )

def _lang_filter_clause(
    lang: str | None,
    param: int,
) -> tuple[str, list]:  # type: ignore[type-arg]
    """Return a parameterised lang filter clause. None = search all languages."""
    if not lang:
        return "", []
    return f"AND lang = ${param}", [lang]

def _to_result(row: dict, scope: str) -> RecallResult:  # type: ignore[type-arg]
    r = RecallResult(
        id=str(row["id"]),
        type=row.get("type", "knowledge"),
        scope=scope,  # type: ignore[arg-type]
        content=row["content"],
        importance=row.get("importance"),
        created_at=row["created_at"].isoformat(),
        valid_from=row["valid_from"].isoformat() if row.get("valid_from") else None,
        valid_until=row["valid_until"].isoformat() if row.get("valid_until") else None,
    )
    r._recall_count = row.get("recall_count", 0) or 0
    r._last_recalled_at = row.get("last_recalled_at")
    r._confidence_multiplier = float(row.get("confidence_multiplier", 1.0) or 1.0)
    r._score = float(row.get("score") or row.get("rank") or 0.0)
    return r

def _rrf_fuse(
    results: list[RecallResult], entity_links: set[str] | None = None
) -> list[RecallResult]:
    if not results:
        return []
    scores: dict[str, float] = {}
    by_id: dict[str, RecallResult] = {}
    seen_positions: dict[str, int] = {}

    for result in results:
        key = result.id
        pos = seen_positions.get(key, 0) + 1
        seen_positions[key] = pos
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + pos)
        by_id[key] = result

    def final_score(rid: str) -> float:
        r = by_id[rid]
        imp = dynamic_importance(
            base_importance=r.importance if r.importance is not None else 0.5,
            recall_count=r._recall_count,
            last_recalled_at=r._last_recalled_at,
            confidence_multiplier=r._confidence_multiplier,
        )
        boost = 1.0
        if entity_links and rid in entity_links:
            boost = 1.5  # 50% boost for entity adjacency
        return scores[rid] * imp * boost

    return [by_id[rid] for rid in sorted(scores.keys(), key=final_score, reverse=True)]


def _sort_by_importance(results: list[RecallResult]) -> list[RecallResult]:
    """Re-rank a flat result list by dynamic_importance (frequency + recency boost).

    Applied at the return site of type-routed strategies so that frequently-recalled
    and recently-accessed memories surface above raw BM25/cosine rank.
    """
    return sorted(
        results,
        key=lambda r: dynamic_importance(
            base_importance=r.importance if r.importance is not None else 0.5,
            recall_count=getattr(r, "_recall_count", 0),
            last_recalled_at=getattr(r, "_last_recalled_at", None),
            confidence_multiplier=getattr(r, "_confidence_multiplier", 1.0),
        ),
        reverse=True,
    )


def _keyword_boost(query: str, content: str) -> float:
    """Normalised token overlap between query and content (case-insensitive).

    Returns a value in [0, 1] representing the fraction of unique query tokens
    that appear in the content. Used to re-score knowledge candidates before MMR
    so that paraphrased queries that share key terms with the ground truth chunk
    are not buried by episodic memories with higher raw vector scores.
    """
    query_tokens = {t.lower() for t in query.split() if len(t) > 2}
    if not query_tokens:
        return 0.0
    content_lower = content.lower()
    matched = sum(1 for t in query_tokens if t in content_lower)
    return matched / len(query_tokens)

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)  # type: ignore[no-any-return]

def _mmr(
    query_embedding: list[float],
    candidates: list[RecallResult],
    lambda_: float = 0.5,
    dedup_threshold: float = 0.85,
    min_score: float = 0.35,
    k: int = 4,
) -> list[RecallResult]:
    eligible = [c for c in candidates if c._score >= min_score]
    if not eligible:
        return []

    selected: list[RecallResult] = []
    remaining = list(eligible)

    while remaining and len(selected) < k:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        best_candidate = None
        best_mmr_score = -1.0

        for candidate in remaining:
            if not candidate._embedding:
                continue
            relevance = _cosine(query_embedding, candidate._embedding)
            redundancy = (
                max(
                    _cosine(candidate._embedding, s._embedding)
                    for s in selected
                    if s._embedding
                )
                if selected
                else 0.0
            )
            if redundancy >= dedup_threshold:
                continue
            score = lambda_ * relevance - (1 - lambda_) * redundancy
            if score > best_mmr_score:
                best_mmr_score = score
                best_candidate = candidate

        if best_candidate is None:
            break
        remaining.remove(best_candidate)
        selected.append(best_candidate)

    return selected

