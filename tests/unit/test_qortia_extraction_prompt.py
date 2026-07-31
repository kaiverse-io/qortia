"""
Unit tests — qortia extraction prompt improvements (#75)
Covers: temporal grounding, attribution instruction, negative-example checklist,
        valid_from extraction from metadata, build_extraction_prompt composition.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from qortia.models import MemoryItem, RememberRequest
from qortia.remember import (
    ATTRIBUTION_INSTRUCTION,
    NEGATIVE_EXTRACTION_INSTRUCTION,
    _extract_valid_from,
    build_extraction_prompt,
    build_temporal_grounding_instruction,
)

# ── Temporal grounding instruction ───────────────────────��──────


class TestTemporalGroundingInstruction:
    def test_contains_reference_time(self) -> None:
        ts = datetime(2026, 5, 20, 14, 30, tzinfo=UTC)
        result = build_temporal_grounding_instruction(ts)
        assert "2026-05-20 14:30 UTC" in result

    def test_default_uses_current_time(self) -> None:
        result = build_temporal_grounding_instruction()
        # Should contain a date string in the current year
        assert "UTC" in result
        assert "Current date and time:" in result

    def test_contains_resolution_examples(self) -> None:
        result = build_temporal_grounding_instruction()
        assert "last Tuesday" in result
        assert "last week" in result
        assert "recently" in result

    def test_instructs_not_to_invent_dates(self) -> None:
        result = build_temporal_grounding_instruction()
        assert "Do NOT invent dates" in result


# ── Attribution instruction ─────────────────────────────────────


class TestAttributionInstruction:
    def test_contains_user_prefix(self) -> None:
        assert "[User]" in ATTRIBUTION_INSTRUCTION

    def test_contains_observed_prefix(self) -> None:
        assert "[Observed]" in ATTRIBUTION_INSTRUCTION

    def test_contains_third_party_prefix(self) -> None:
        assert "[Third-party]" in ATTRIBUTION_INSTRUCTION

    def test_distinguishes_observed_from_user(self) -> None:
        assert "inferring from behaviour" in ATTRIBUTION_INSTRUCTION
        assert "direct statements" in ATTRIBUTION_INSTRUCTION


# ── Negative extraction instruction ────────────────────────────


class TestNegativeExtractionInstruction:
    def test_rejects_pronouns_without_antecedents(self) -> None:
        assert "Pronouns or references without clear antecedents" in NEGATIVE_EXTRACTION_INSTRUCTION

    def test_rejects_abstract_concepts(self) -> None:
        assert "Abstract concepts without grounding" in NEGATIVE_EXTRACTION_INSTRUCTION

    def test_rejects_status_only_observations(self) -> None:
        assert "Status-only observations" in NEGATIVE_EXTRACTION_INSTRUCTION
        assert '"done"' in NEGATIVE_EXTRACTION_INSTRUCTION
        assert '"ok"' in NEGATIVE_EXTRACTION_INSTRUCTION

    def test_rejects_greetings(self) -> None:
        assert "Greetings and pleasantries" in NEGATIVE_EXTRACTION_INSTRUCTION

    def test_rejects_single_turn_questions(self) -> None:
        assert "Single-turn questions" in NEGATIVE_EXTRACTION_INSTRUCTION

    def test_rejects_task_instructions(self) -> None:
        assert (
            "Task instructions that do not reveal a durable fact" in NEGATIVE_EXTRACTION_INSTRUCTION
        )

    def test_rejects_generic_action_nouns(self) -> None:
        assert "Generic action nouns" in NEGATIVE_EXTRACTION_INSTRUCTION


# ── build_extraction_prompt composition ─────────────────────────


class TestBuildExtractionPrompt:
    def test_episodic_includes_all_three_sections(self) -> None:
        ts = datetime(2026, 5, 20, 10, 0, tzinfo=UTC)
        result = build_extraction_prompt("episodic", reference_time=ts)
        assert "2026-05-20 10:00 UTC" in result
        assert "[User]" in result
        assert "[Observed]" in result
        assert "DO NOT extract" in result

    def test_experiential_includes_temporal_and_negative_but_not_attribution(
        self,
    ) -> None:
        result = build_extraction_prompt("experiential")
        assert "Current date and time:" in result
        assert "DO NOT extract" in result
        # Attribution is episodic-only
        assert "[User] —" not in result

    def test_mental_model_includes_only_temporal(self) -> None:
        result = build_extraction_prompt("mental_model")
        assert "Current date and time:" in result
        assert "[User] —" not in result
        assert "DO NOT extract" not in result

    def test_decision_includes_only_temporal(self) -> None:
        result = build_extraction_prompt("decision")
        assert "Current date and time:" in result
        assert "DO NOT extract" not in result

    def test_lesson_includes_only_temporal(self) -> None:
        result = build_extraction_prompt("lesson")
        assert "Current date and time:" in result
        assert "DO NOT extract" not in result


# ── _extract_valid_from ─────────────────────────────────────────


class TestExtractValidFrom:
    def test_none_metadata_returns_none(self) -> None:
        assert _extract_valid_from(None) is None

    def test_empty_metadata_returns_none(self) -> None:
        assert _extract_valid_from({}) is None

    def test_missing_key_returns_none(self) -> None:
        assert _extract_valid_from({"other_key": "value"}) is None

    def test_iso_string_parses_correctly(self) -> None:
        result = _extract_valid_from({"valid_from": "2026-05-20T14:30:00+00:00"})
        assert result is not None
        assert result.year == 2026
        assert result.month == 5
        assert result.day == 20
        assert result.hour == 14
        assert result.minute == 30

    def test_iso_string_with_z_suffix(self) -> None:
        result = _extract_valid_from({"valid_from": "2026-05-20T14:30:00Z"})
        assert result is not None
        assert result.year == 2026

    def test_datetime_object_passes_through(self) -> None:
        dt = datetime(2026, 3, 15, 9, 0, tzinfo=UTC)
        result = _extract_valid_from({"valid_from": dt})
        assert result is dt

    def test_invalid_string_returns_none(self) -> None:
        result = _extract_valid_from({"valid_from": "not-a-date"})
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = _extract_valid_from({"valid_from": ""})
        assert result is None

    def test_none_value_returns_none(self) -> None:
        result = _extract_valid_from({"valid_from": None})
        assert result is None


# ── remember() with valid_from and source_message_index ─────────


TENANT_ID = uuid4()
AGENT_ID = uuid4()


def _make_identity():
    identity = MagicMock()
    identity.agent_id = AGENT_ID
    identity.tenant_id = TENANT_ID
    return identity


@pytest.mark.asyncio
async def test_remember_passes_valid_from_to_insert() -> None:
    """When metadata contains valid_from, it is passed to the INSERT."""
    from qortia.auth import AgentIdentity
    from qortia.remember import remember

    new_id = uuid4()
    agent = AgentIdentity(agent_id=AGENT_ID, tenant_id=TENANT_ID)
    body = RememberRequest(
        memories=[
            MemoryItem(
                type="episodic",
                content="[User] Started working on the new authentication module on 2026-05-15",
                metadata={
                    "valid_from": "2026-05-15T09:00:00+00:00",
                    "source_message_index": 3,
                },
            )
        ]
    )

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"status": "active"})
    # First fetchval: dedup check returns None; second: INSERT returns new_id
    mock_conn.fetchval = AsyncMock(side_effect=[None, new_id])
    mock_conn.execute = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("qortia.remember.tenant_transaction", return_value=mock_ctx),
        patch("qortia.remember.get_main_pool"),
        patch("qortia.remember.extract_entities_with_types", return_value=[]),
    ):
        result = await remember(body, agent)

    assert str(new_id) in result.ids

    # Verify the INSERT was called with valid_from as the 12th parameter
    insert_call = mock_conn.fetchval.call_args_list[1]
    insert_args = insert_call[0]
    # The SQL should contain valid_from
    assert "valid_from" in insert_args[0]
    # 12th positional arg (index 12) is the valid_from datetime
    valid_from_arg = insert_args[12]
    assert valid_from_arg is not None
    assert valid_from_arg.year == 2026
    assert valid_from_arg.month == 5
    assert valid_from_arg.day == 15


@pytest.mark.asyncio
async def test_remember_strips_valid_from_from_stored_metadata() -> None:
    """valid_from should be in its own column, not duplicated in metadata JSON."""
    from qortia.auth import AgentIdentity
    from qortia.remember import remember

    new_id = uuid4()
    agent = AgentIdentity(agent_id=AGENT_ID, tenant_id=TENANT_ID)
    body = RememberRequest(
        memories=[
            MemoryItem(
                type="episodic",
                content="[User] Prefers Python over JavaScript for backend development",
                metadata={
                    "valid_from": "2026-05-15T09:00:00+00:00",
                    "source_message_index": 7,
                },
            )
        ]
    )

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"status": "active"})
    mock_conn.fetchval = AsyncMock(side_effect=[None, new_id])
    mock_conn.execute = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("qortia.remember.tenant_transaction", return_value=mock_ctx),
        patch("qortia.remember.get_main_pool"),
        patch("qortia.remember.extract_entities_with_types", return_value=[]),
    ):
        await remember(body, agent)

    # The 7th positional arg (index 7) in the INSERT is the metadata JSON
    insert_call = mock_conn.fetchval.call_args_list[1]
    metadata_json = insert_call[0][7]
    metadata_dict = json.loads(metadata_json)
    # valid_from should NOT be in metadata (it's in its own column)
    assert "valid_from" not in metadata_dict
    # source_message_index should still be preserved
    assert metadata_dict["source_message_index"] == 7


@pytest.mark.asyncio
async def test_remember_without_valid_from_uses_db_default() -> None:
    """When no valid_from in metadata, pass None so DB default (now()) applies."""
    from qortia.auth import AgentIdentity
    from qortia.remember import remember

    new_id = uuid4()
    agent = AgentIdentity(agent_id=AGENT_ID, tenant_id=TENANT_ID)
    body = RememberRequest(
        memories=[
            MemoryItem(
                type="episodic",
                content="The user mentioned they enjoy hiking on weekends regularly",
            )
        ]
    )

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"status": "active"})
    mock_conn.fetchval = AsyncMock(side_effect=[None, new_id])
    mock_conn.execute = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("qortia.remember.tenant_transaction", return_value=mock_ctx),
        patch("qortia.remember.get_main_pool"),
        patch("qortia.remember.extract_entities_with_types", return_value=[]),
    ):
        result = await remember(body, agent)

    assert str(new_id) in result.ids
    # 12th arg (valid_from) should be None — COALESCE in SQL will use now()
    insert_call = mock_conn.fetchval.call_args_list[1]
    valid_from_arg = insert_call[0][12]
    assert valid_from_arg is None


# ── Reflect prompt temporal grounding ───────────────────────────


class TestReflectPromptTemporalGrounding:
    def test_reflect_prompt_contains_temporal_instruction(self) -> None:
        from qortia.reflect import _build_reflect_prompt

        result = _build_reflect_prompt(
            recent=["memory one about last Tuesday"],
            existing=[],
        )
        assert "Current date and time:" in result
        assert "UTC" in result

    def test_reflect_prompt_preserves_resolved_dates_rule(self) -> None:
        from qortia.reflect import _build_reflect_prompt

        result = _build_reflect_prompt(recent=[], existing=[])
        assert "Preserve specific resolved dates from source memories" in result

    def test_reflect_prompt_preserves_attribution_prefixes(self) -> None:
        from qortia.reflect import _build_reflect_prompt

        result = _build_reflect_prompt(recent=[], existing=[])
        assert "[User]" in result
        assert "[Observed]" in result
        assert "[Third-party]" in result
