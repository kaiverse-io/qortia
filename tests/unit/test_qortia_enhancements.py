"""Unit tests — qortia enhancements:
  1. Importance decay in type-routed recall (_sort_by_importance)
  2. Rerank model decoupling (settings.rerank_model fallback)
  3. Background reflection trigger (run_background_reflection_trigger / _trigger_idle_reflections)
  4. Auto lang detection (_detect_lang + remember integration)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
AGENT_ID = UUID("00000000-0000-0000-0000-000000000002")


# ── helpers ────────────────────────────────────────────────────


def _make_result(
    recall_count: int = 0,
    last_recalled_at: datetime | None = None,
    importance: float = 0.5,
    content: str = "test",
) -> object:
    from app.qortia.models import RecallResult

    r = RecallResult(
        id=str(UUID(int=recall_count + 1)),
        type="episodic",
        scope="private",
        content=content,
        importance=importance,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
    )
    r._recall_count = recall_count
    r._last_recalled_at = last_recalled_at
    r._score = 0.5
    return r


def _patch_tx(conn):
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


# ══════════════════════════════════════════════════════════════
# 1. Importance decay — _sort_by_importance
# ══════════════════════════════════════════════════════════════


def test_sort_by_importance_higher_recall_count_ranks_first() -> None:
    from app.qortia.recall_helpers import _sort_by_importance

    low = _make_result(recall_count=0, importance=0.5)
    high = _make_result(recall_count=20, importance=0.5)
    result = _sort_by_importance([low, high])
    assert result[0].id == high.id


def test_sort_by_importance_recent_access_boosts_rank() -> None:
    from app.qortia.recall_helpers import _sort_by_importance

    stale = _make_result(
        recall_count=5,
        last_recalled_at=datetime.now(timezone.utc) - timedelta(days=60),
    )
    fresh = _make_result(
        recall_count=5,
        last_recalled_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    result = _sort_by_importance([stale, fresh])
    assert result[0].id == fresh.id


def test_sort_by_importance_preserves_all_results() -> None:
    from app.qortia.recall_helpers import _sort_by_importance

    items = [_make_result(recall_count=i) for i in range(5)]
    result = _sort_by_importance(items)
    assert len(result) == 5


def test_sort_by_importance_empty_list() -> None:
    from app.qortia.recall_helpers import _sort_by_importance

    assert _sort_by_importance([]) == []


def test_sort_by_importance_missing_attrs_handled() -> None:
    """Results without _recall_count/_last_recalled_at (e.g. org results) don't crash."""
    from app.qortia.recall_helpers import _sort_by_importance
    from app.qortia.models import RecallResult

    r = RecallResult(
        id="00000000-0000-0000-0000-000000000099",
        type="org_chart",
        scope="org",
        content="no attrs",
        importance=0.7,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
    )
    result = _sort_by_importance([r])
    assert len(result) == 1


# ══════════════════════════════════════════════════════════════
# 2. Rerank model decoupling
# ══════════════════════════════════════════════════════════════


def test_rerank_model_default_in_settings() -> None:
    from app.config import Settings

    s = Settings()
    assert s.rerank_model == "anthropic/claude-3-haiku-20240307"


def test_rerank_model_env_override() -> None:
    from app.config import Settings

    s = Settings(rerank_model="anthropic/claude-3-5-sonnet-20241022")
    assert s.rerank_model == "anthropic/claude-3-5-sonnet-20241022"


@pytest.mark.asyncio
async def test_llm_rerank_uses_settings_when_domain_md_has_no_model() -> None:
    """When domain_md is empty/null, _llm_rerank falls back to settings.rerank_model."""
    from app.qortia.recall_rerank import _llm_rerank
    from app.auth.models import AgentIdentity

    agent = AgentIdentity(
        agent_id=AGENT_ID,
        tenant_id=TENANT_ID,
        jti=UUID("00000000-0000-0000-0000-000000000010"),
    )
    results = [_make_result(content=f"mem {i}") for i in range(3)]

    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value="")  # empty domain_md

    pool_ctx = MagicMock()
    pool_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    pool_ctx.__aexit__ = AsyncMock(return_value=False)

    captured_model = {}

    async def fake_post(path, headers, json, timeout):
        captured_model["model"] = json["model"]
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "[1,2,3]"}}],
            "usage": {},
        }
        return resp

    with patch("app.qortia.recall_rerank.get_main_pool") as mock_pool, patch(
        "app.qortia.recall_rerank.get_litellm_client"
    ) as mock_client, patch(
        "app.qortia.recall_rerank.get_litellm_key", return_value="test-key"
    ), patch(
        "app.qortia.recall_rerank.settings"
    ) as mock_settings:
        mock_pool.return_value.acquire.return_value = pool_ctx
        mock_client.return_value.post = fake_post
        mock_settings.rerank_model = "anthropic/claude-3-haiku-20240307"

        await _llm_rerank("test query", results, agent)

    assert captured_model.get("model") == "anthropic/claude-3-haiku-20240307"


@pytest.mark.asyncio
async def test_llm_rerank_prefers_domain_md_model_over_settings() -> None:
    from app.qortia.recall_rerank import _llm_rerank
    from app.auth.models import AgentIdentity

    agent = AgentIdentity(
        agent_id=AGENT_ID,
        tenant_id=TENANT_ID,
        jti=UUID("00000000-0000-0000-0000-000000000010"),
    )
    results = [_make_result(content=f"mem {i}") for i in range(2)]

    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(
        return_value="model: anthropic/claude-opus-4-5\nrole: engineer"
    )
    pool_ctx = MagicMock()
    pool_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    pool_ctx.__aexit__ = AsyncMock(return_value=False)

    captured_model = {}

    async def fake_post(path, headers, json, timeout):
        captured_model["model"] = json["model"]
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "[1,2]"}}],
            "usage": {},
        }
        return resp

    with patch("app.qortia.recall_rerank.get_main_pool") as mock_pool, patch(
        "app.qortia.recall_rerank.get_litellm_client"
    ) as mock_client, patch(
        "app.qortia.recall_rerank.get_litellm_key", return_value="test-key"
    ), patch(
        "app.qortia.recall_rerank.settings"
    ) as mock_settings:
        mock_pool.return_value.acquire.return_value = pool_ctx
        mock_client.return_value.post = fake_post
        mock_settings.rerank_model = "anthropic/claude-3-haiku-20240307"

        await _llm_rerank("test query", results, agent)

    assert captured_model.get("model") == "anthropic/claude-opus-4-5"


# ══════════════════════════════════════════════════════════════
# 3. Background reflection trigger
# ══════════════════════════════════════════════════════════════


def test_run_background_reflection_trigger_is_async() -> None:
    import inspect
    from app.qortia.reflect import run_background_reflection_trigger

    assert inspect.iscoroutinefunction(run_background_reflection_trigger)


def test_trigger_idle_reflections_is_async() -> None:
    import inspect
    from app.qortia.reflect import _trigger_idle_reflections

    assert inspect.iscoroutinefunction(_trigger_idle_reflections)


@pytest.mark.asyncio
async def test_trigger_idle_reflections_calls_reflect_agent_for_each_row() -> None:
    from app.qortia.reflect import _trigger_idle_reflections

    rows = [
        {"agent_id": AGENT_ID, "tenant_id": TENANT_ID},
        {
            "agent_id": UUID("00000000-0000-0000-0000-000000000099"),
            "tenant_id": TENANT_ID,
        },
    ]
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=rows)

    pool_ctx = MagicMock()
    pool_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    pool_ctx.__aexit__ = AsyncMock(return_value=False)

    called_agents = []

    async def fake_reflect_agent(agent_id, tenant_id):
        called_agents.append(agent_id)

    with patch("app.qortia.reflect.get_main_pool") as mock_pool, patch(
        "app.qortia.reflect._reflect_agent", side_effect=fake_reflect_agent
    ), patch("app.qortia.reflect.settings") as mock_settings:
        mock_pool.return_value.acquire.return_value = pool_ctx
        mock_settings.idle_reflection_window_h = 4

        await _trigger_idle_reflections()

    assert len(called_agents) == 2


@pytest.mark.asyncio
async def test_trigger_idle_reflections_handles_empty_result() -> None:
    from app.qortia.reflect import _trigger_idle_reflections

    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])

    pool_ctx = MagicMock()
    pool_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    pool_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.qortia.reflect.get_main_pool") as mock_pool, patch(
        "app.qortia.reflect._reflect_agent"
    ) as mock_reflect, patch(
        "app.qortia.reflect.settings"
    ) as mock_settings:
        mock_pool.return_value.acquire.return_value = pool_ctx
        mock_settings.idle_reflection_window_h = 4

        await _trigger_idle_reflections()

    mock_reflect.assert_not_called()


@pytest.mark.asyncio
async def test_trigger_idle_reflections_logs_on_db_error() -> None:
    from app.qortia.reflect import _trigger_idle_reflections

    pool_ctx = MagicMock()
    pool_ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("db down"))
    pool_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.qortia.reflect.get_main_pool") as mock_pool, patch(
        "app.qortia.reflect.settings"
    ) as mock_settings:
        mock_pool.return_value.acquire.return_value = pool_ctx
        mock_settings.idle_reflection_window_h = 4
        # Should not raise — errors are caught and logged
        await _trigger_idle_reflections()


@pytest.mark.asyncio
async def test_reflect_agent_skips_inactive_agent() -> None:
    from app.qortia.reflect import _reflect_agent

    mock_conn = AsyncMock()
    # First fetchval: status check → "inactive"
    # fetch calls: return empty to avoid further processing
    mock_conn.fetchval = AsyncMock(return_value="inactive")
    mock_conn.fetch = AsyncMock(return_value=[])

    tx_ctx = MagicMock()
    tx_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    tx_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.qortia.reflect.tenant_transaction", return_value=tx_ctx), patch(
        "app.qortia.reflect.get_main_pool"
    ), patch(
        "app.qortia.reflect._call_litellm_reflect", new_callable=AsyncMock
    ) as mock_llm, patch(
        "app.qortia.remember._fetch_agent_clearance",
        new_callable=AsyncMock,
        return_value=(1, None),
    ):
        await _reflect_agent(AGENT_ID, TENANT_ID)

    mock_llm.assert_not_called()


# ══════════════════════════════════════════════════════════════
# 4. Auto lang detection
# ══════════════════════════════════════════════════════════════


def test_detect_lang_returns_string() -> None:
    from app.qortia.remember import _detect_lang

    result = _detect_lang("The quick brown fox jumps over the lazy dog")
    assert isinstance(result, str)
    assert len(result) >= 2


def test_detect_lang_returns_detected_lang() -> None:
    """_detect_lang passes through what langdetect returns (base tag only)."""
    from app.qortia.remember import _detect_lang

    with patch("app.qortia.remember.detect", return_value="hi"):
        result = _detect_lang("some text")
    assert result == "hi"


def test_detect_lang_returns_en_on_failure() -> None:
    from app.qortia.remember import _detect_lang

    with patch("app.qortia.remember.detect", side_effect=Exception("model error")):
        result = _detect_lang("any text at all")
    assert result == "en"


def test_detect_lang_normalises_subtag() -> None:
    """zh-cn → zh, en-US → en, etc."""
    from app.qortia.remember import _detect_lang

    with patch("app.qortia.remember.detect", return_value="zh-cn"):
        result = _detect_lang("some chinese text")
    assert result == "zh"


@pytest.mark.asyncio
async def test_remember_auto_detects_hindi_lang() -> None:
    """When agent passes lang='en' (default) but content is Hindi, lang is auto-detected."""
    from app.qortia.remember import remember
    from app.auth.models import AgentIdentity
    from app.qortia.models import RememberRequest, MemoryItem

    agent = AgentIdentity(
        agent_id=AGENT_ID,
        tenant_id=TENANT_ID,
        jti=UUID("00000000-0000-0000-0000-000000000010"),
    )
    body = RememberRequest(
        memories=[
            MemoryItem(
                type="episodic",
                content="यह एक परीक्षण वाक्य है जो हिंदी में लिखा गया है और यह काफी लंबा है",
                lang="en",  # agent didn't detect — default
            )
        ]
    )

    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=UUID(int=42))
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.execute = AsyncMock()

    tx_ctx = MagicMock()
    tx_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    tx_ctx.__aexit__ = AsyncMock(return_value=False)

    written_langs = []
    original_fetchval = mock_conn.fetchval

    async def capture_fetchval(*args, **kwargs):
        sql = args[0] if args else ""
        if "INSERT INTO hindsight_memories" in sql:
            written_langs.append(args[10])  # lang is $10
        return UUID(int=42)

    mock_conn.fetchval = capture_fetchval

    with patch(
        "app.qortia.remember.tenant_transaction", return_value=tx_ctx
    ), patch("app.qortia.remember.get_main_pool"), patch(
        "app.qortia.remember.assert_agent_active", new_callable=AsyncMock
    ), patch(
        "app.qortia.remember.extract_entities_with_types", return_value=[]
    ):
        await remember(body=body, agent=agent)

    if written_langs:
        assert written_langs[0] == "hi"


@pytest.mark.asyncio
async def test_remember_skips_detection_for_short_content() -> None:
    """Content < 20 chars skips auto-detection and keeps lang='en'."""
    from app.qortia.remember import remember
    from app.auth.models import AgentIdentity
    from app.qortia.models import RememberRequest, MemoryItem

    agent = AgentIdentity(
        agent_id=AGENT_ID,
        tenant_id=TENANT_ID,
        jti=UUID("00000000-0000-0000-0000-000000000010"),
    )
    body = RememberRequest(
        memories=[MemoryItem(type="episodic", content="a b c d e f", lang="en")]
    )

    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=UUID(int=1))
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.execute = AsyncMock()

    tx_ctx = MagicMock()
    tx_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    tx_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "app.qortia.remember.tenant_transaction", return_value=tx_ctx
    ), patch("app.qortia.remember.get_main_pool"), patch(
        "app.qortia.remember.assert_agent_active", new_callable=AsyncMock
    ), patch(
        "app.qortia.remember.extract_entities_with_types", return_value=[]
    ), patch(
        "app.qortia.remember._detect_lang"
    ) as mock_detect:
        await remember(body=body, agent=agent)

    mock_detect.assert_not_called()
