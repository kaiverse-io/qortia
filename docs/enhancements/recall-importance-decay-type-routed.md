---
kind: enhancement
owner: platform
last_reviewed: 2026-05-30
status: implemented
---

# Recall: Apply Dynamic Importance Decay to Type-Routed Strategies

**Status:** Implemented — 2026-05-30
**Scope:** `platform/app/qortia/recall.py`
**ADR required:** No — no schema change, no API contract change
**Depends on:** None
**Research source:** "Managing Memory for AI Agents" (O'Reilly/Redis, 2026) — Ch. 1,
"Importance scoring: recency, frequency of reference, user engagement metrics"

---

## 1. The Problem

`dynamic_importance` is defined in `recall.py` and applied correctly in `_rrf_fuse`
for the full hybrid pipeline. It boosts results based on `recall_count` and
`last_recalled_at`, implementing the industry-standard importance scoring pattern.

However, the four type-routed strategies bypass `_rrf_fuse` entirely and return
results ordered by raw BM25 rank or cosine score:

| Function | Sort order | `dynamic_importance` applied? |
|---|---|---|
| `_recall_decisions` | `rank DESC, created_at DESC` | No |
| `_recall_lessons` | `embedding <=> query ASC` (cosine) | No |
| `_recall_episodic` | `CASE tsv match THEN 0 ELSE 1, rank DESC, created_at DESC` | No |
| `_recall_short_term` | `rank DESC, created_at DESC` | No |

A decision recalled 50 times last week ranks identically to one never recalled.
A lesson that has been consistently surfaced as relevant ranks the same as one
that has never been retrieved. The `recall_count` and `last_recalled_at` columns
are written on every access (via `_record_recall_access`) but never read back into
the type-routed ranking.

This is inconsistent with the full hybrid pipeline and with the design intent of
`dynamic_importance` (Q95 in `docs/archive/design-clarity-monolith.md`).

---

## 2. Root Cause

The type-routed strategies were added as fast-path shortcuts for specific memory
types. They return DB-ordered results directly without a post-sort step. The
`dynamic_importance` function was added later as part of the full hybrid pipeline
and was not retrofitted to the type-routed paths.

---

## 3. Fix

Add a post-sort step to each type-routed function that re-orders results by
`dynamic_importance` before returning. The DB query still runs with its existing
`ORDER BY` (for efficient index use and LIMIT), but the Python-side sort applies
the importance signal on the returned rows.

```python
# Shared helper — add once, call from all four functions
def _sort_by_importance(results: list[RecallResult]) -> list[RecallResult]:
    return sorted(
        results,
        key=lambda r: dynamic_importance(
            base_importance=r.importance if r.importance is not None else 0.5,
            recall_count=r._recall_count,
            last_recalled_at=r._last_recalled_at,
        ),
        reverse=True,
    )
```

Apply at the return site of each function:

```python
# _recall_decisions
return _sort_by_importance([_to_result(dict(r), "private") for r in rows])

# _recall_lessons (already filtered by score >= 0.35)
return _sort_by_importance([
    _to_result(dict(r), "private") for r in rows if (r.get("score") or 0) >= 0.35
])

# _recall_episodic
return _sort_by_importance([_to_result(dict(r), "private") for r in rows])

# _recall_short_term — importance decay less relevant for TTL-scoped memories,
# but recency signal (last_recalled_at) is still useful for deduplication
return _sort_by_importance([_to_result(dict(r), "private") for r in rows])
```

---

## 4. Why This Is Safe

- The DB query still uses the index-friendly `ORDER BY rank DESC` / cosine sort
  with `LIMIT 10`. The Python sort operates on at most 10 rows — negligible cost.
- `dynamic_importance` is a pure function with no I/O. It cannot fail.
- The sort is stable: ties in `dynamic_importance` preserve the original DB order.
- `_recall_short_term` memories have `importance = 0.1` (the lowest tier in
  `IMPORTANCE`). The `dynamic_importance` boost from `recall_count` and
  `last_recalled_at` is small relative to the base. The sort has minimal effect
  on short-term results but is included for consistency.
- No schema change. No migration. No API contract change.

---

## 5. Files Affected

| File | Change |
|---|---|
| `platform/app/qortia/recall.py` | Add `_sort_by_importance` helper; apply at return site of `_recall_decisions`, `_recall_lessons`, `_recall_episodic`, `_recall_short_term` |

---

## 6. Test Gates

| Gate | What to verify |
|---|---|
| Unit test — `test_recall_pipeline.py` | `_recall_decisions` returns results sorted by `dynamic_importance`, not raw BM25 rank |
| Unit test — `test_recall_pipeline.py` | A result with `recall_count=10` outranks one with `recall_count=0` when BM25 scores are equal |
| Recall eval | `evals/run_reh.py` — Recall@5 ≥ 0.95, MRR ≥ 0.86 (must not regress) |
| Platform unit tests | `cd platform && python3 -m pytest tests/unit/ -q` — 296/296 |

---

## 7. Related

- `dynamic_importance` definition: `recall.py` lines ~40–50
- `_rrf_fuse` (applies importance correctly): `recall.py`
- `_record_recall_access` (writes `recall_count`, `last_recalled_at`): `recall.py`
- Q95 in `docs/archive/design-clarity-monolith.md` (archived) — dynamic importance design decision
