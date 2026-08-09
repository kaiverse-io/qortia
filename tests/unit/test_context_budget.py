"""Unit tests for `_budget_memories` — the pure cross-bucket truncation logic
behind `GET /v1/context`'s `budget` parameter — and for `get_context()`'s own
wiring (mocked connection, no live DB; see tests/integration/test_qortia.py
for the real-DB round trip)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from qortia.auth import AgentIdentity
from qortia.models import MemoryEntry
from qortia.remember import _budget_memories, get_context

TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
AGENT_ID = UUID("00000000-0000-0000-0000-000000000002")


class _AcquireContext:
    def __init__(self, conn: MagicMock) -> None:
        self.conn = conn

    async def __aenter__(self) -> MagicMock:
        return self.conn

    async def __aexit__(self, *_exc: object) -> None:
        return None


def _entry(content: str, importance: float) -> MemoryEntry:
    return MemoryEntry(content=content, importance=importance)


def test_no_budget_passes_through_unchanged() -> None:
    mm = [_entry("a", 0.8)]
    dec = [_entry("b", 0.9)]
    les = [_entry("c", 0.95)]

    out = _budget_memories(mm, dec, les, budget=None)

    assert out.mental_models == mm
    assert out.decisions == dec
    assert out.lessons == les


def test_zero_or_negative_budget_treated_as_no_budget() -> None:
    mm = [_entry("a", 0.8)]

    out = _budget_memories(mm, [], [], budget=0)

    assert out.mental_models == mm


def test_lessons_survive_over_lower_importance_mental_models() -> None:
    """The failure this replaces: render-order truncation drops 0.95 lessons
    before 0.3-ish content. Here the pool is small enough that only the
    lesson should survive a tight budget."""
    mm = [_entry("low importance filler " * 5, 0.3)]
    lesson = _entry("the important lesson", 0.95)

    out = _budget_memories(mm, [], [lesson], budget=len(lesson.content))

    assert out.lessons == [lesson]
    assert out.mental_models == []


def test_cross_bucket_ordering_not_bucket_order() -> None:
    """A high-importance decision must survive over a low-importance mental
    model even though mental_models would render first."""
    weak_model = _entry("weak model", 0.2)
    strong_decision = _entry("strong decision", 0.9)

    out = _budget_memories([weak_model], [strong_decision], [], budget=len(strong_decision.content))

    assert out.decisions == [strong_decision]
    assert out.mental_models == []


def test_at_least_one_entry_survives_even_if_it_exceeds_budget() -> None:
    """Dropping to zero entries is worse than one entry over budget — the
    `used > 0` guard means the first (highest-importance) entry always gets
    in, and only subsequent entries are budget-checked."""
    only = _entry("x" * 500, 0.95)

    out = _budget_memories([], [], [only], budget=10)

    assert out.lessons == [only]


def test_within_bucket_order_is_preserved_for_survivors() -> None:
    """Truncation decides *which* entries survive by importance; it must not
    reorder the ones that do survive within their own bucket."""
    first = _entry("first, higher importance", 0.9)
    second = _entry("second, still kept", 0.85)

    out = _budget_memories([], [], [first, second], budget=len(first.content) + len(second.content))

    assert out.lessons == [first, second]


# ── get_context() wiring — mocked conn, no live DB ──────────────────────────


def _row(**fields: object) -> MagicMock:
    row = MagicMock()
    row.__getitem__.side_effect = fields.__getitem__
    return row


@pytest.mark.asyncio
async def test_get_context_passes_budget_through_and_reports_reflection_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = MagicMock()
    conn.fetch = AsyncMock(
        side_effect=[
            [_row(title="Org", content="org chart")],  # org_chart
            [_row(title="Proc", content="process")],  # processes
            [_row(title="Handoff", content="handoff")],  # handoffs
            [_row(content="model", importance=0.8)],  # mental_models
            [_row(content="decision", importance=0.9)],  # decisions
            [_row(content="lesson", importance=0.95)],  # lessons
        ]
    )
    conn.fetchrow = AsyncMock(return_value=_row(title="Week", content="weekly"))
    conn.fetchval = AsyncMock(return_value=7)

    monkeypatch.setattr("qortia.remember.get_main_pool", lambda: MagicMock())
    monkeypatch.setattr(
        "qortia.remember.tenant_transaction", lambda *_a, **_k: _AcquireContext(conn)
    )
    monkeypatch.setattr("qortia.remember.assert_agent_active", AsyncMock())

    agent = AgentIdentity(agent_id=AGENT_ID, tenant_id=TENANT_ID)
    result = await get_context(agent, budget=5)

    assert result.reflection_counter == 7
    # budget=5 is smaller than any single entry's content, so only the
    # highest-importance survivor (the lesson) should remain — the same
    # "at least one entry survives" guarantee _budget_memories tests above.
    assert [e.content for e in result.memories.lessons] == ["lesson"]
    assert result.memories.mental_models == []
    assert result.memories.decisions == []


@pytest.mark.asyncio
async def test_get_context_defaults_reflection_counter_to_zero_when_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`qortia_agents.reflection_counter` is NOT NULL in the schema, but the
    fetchval() call is still a nullable Python type — guard the None case
    explicitly rather than let a schema assumption silently drift."""
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=None)

    monkeypatch.setattr("qortia.remember.get_main_pool", lambda: MagicMock())
    monkeypatch.setattr(
        "qortia.remember.tenant_transaction", lambda *_a, **_k: _AcquireContext(conn)
    )
    monkeypatch.setattr("qortia.remember.assert_agent_active", AsyncMock())

    agent = AgentIdentity(agent_id=AGENT_ID, tenant_id=TENANT_ID)
    result = await get_context(agent, budget=None)

    assert result.reflection_counter == 0
    assert result.weekly_summary is None
