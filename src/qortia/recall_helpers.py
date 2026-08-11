"""Pure helper functions for the recall pipeline. No I/O or database access.

Tuning constants that used to live here as module-level literals (RRF k, the
result-count limits, the search fetch multiplier) are now read from
`qortia.config.settings` at point of use instead of cached at import time —
reading a process-wide, already-loaded settings singleton isn't I/O in the
sense this docstring means (nothing is touched per-call; that happened once
at process startup), and caching them as bare module constants would freeze
whatever value existed at first import, invisible to any later config change
or to a test that monkeypatches `config.settings` expecting it to take
effect (same reasoning `recall_rerank._llm_rerank` already applies to
`rerank_model`). See qortia.config's own docstring for why these became
file-configurable: recall()'s response size was never capped at all before,
a real gap this module now closes.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from qortia import config
from qortia.models import RecallResult

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
        days_since = (datetime.now(UTC) - last_recalled_at).days
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
    clause = f"""
        AND EXISTS (
            SELECT 1 FROM jsonb_array_elements(entities) AS e
            WHERE e->>0 = ANY(${base_param}::text[])
        )
    """  # noqa: S608
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
    rrf_k = config.settings.recall_rrf_k
    scores: dict[str, float] = {}
    by_id: dict[str, RecallResult] = {}
    seen_positions: dict[str, int] = {}

    for result in results:
        key = result.id
        pos = seen_positions.get(key, 0) + 1
        seen_positions[key] = pos
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + pos)
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


def _resolve_max_chars(body_max_chars: int | None) -> int:
    """What /v1/recall's char budget actually is for one request.

    `None` — the common case, no caller passes `max_chars` — resolves to the
    server-configured default, not "no cap": recall() having no size cap by
    default at all is the exact gap this whole change closes (see
    `_apply_char_budget`). An explicit non-positive value is the caller's own
    choice to opt out and get everything; a positive value is used as-is.
    Pulled out of the endpoint as its own pure function so it's unit-testable
    without standing up `/v1/recall`'s full DB-backed pipeline.
    """
    if body_max_chars is not None:
        return body_max_chars
    return config.settings.recall_default_max_chars


def _apply_char_budget(results: list[RecallResult], max_chars: int) -> list[RecallResult]:
    """Keep results, already ranked, until their combined `content` would cross
    `max_chars`. Drops whole results rather than truncating any one — a sliced
    memory reads as a complete, wrong answer, not a signal something was cut.

    /v1/recall had no cap on total response size at all: the private/org/
    knowledge result-count limits (`config.settings.recall_*_result_limit`,
    20/10/16 by default) bound *result count*, not content volume, and full
    `RecallResult.content` is unbounded per row. Measured against a real
    corpus (agnova's evals/run_scale_eval_qortia.py,
    276 FiQA documents, 100 queries): recall() averaged 38,961 characters
    returned per call — roughly 80x what a comparable capped lexical backend
    returns for the same queries — for only 5.5% of those characters belonging
    to a query's actual relevant document.

    Same policy `_budget_memories` (qortia.remember, `/v1/context`) already
    established for this codebase — ranked-order, drop whole entries, stop at
    the first one that doesn't fit rather than skipping ahead to a smaller
    later one (its `used > 0` guard), keep at least one even over budget.
    `/v1/recall` had never gotten the same treatment; this is that policy
    applied to the other read path, not a new invention.
    (`agnova.memory.qortia_backend._fill_budget` independently applies a
    *client-side* version of the same idea to `/v1/context`'s response, for
    the same reason.) An empty list here would be indistinguishable from "no
    relevant memories," and Qortia content has no snippet-window precedent
    capping a single result the way agnova's `_snippet()` does, so one memory
    can legitimately exceed any budget on its own.
    """
    kept: list[RecallResult] = []
    used = 0
    for result in results:
        cost = len(result.content)
        if kept and used + cost > max_chars:
            break
        kept.append(result)
        used += cost
        if used >= max_chars:
            break
    return kept


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
    dot = sum(x * y for x, y in zip(a, b))  # noqa: B905
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
            selected_embeddings = [s._embedding for s in selected if s._embedding]
            redundancy = (
                max(
                    _cosine(candidate._embedding, selected_embedding)
                    for selected_embedding in selected_embeddings
                )
                if selected_embeddings
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
