"""Unit tests for provisioning CLI and eval router helpers."""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from qortia.eval_router import _parse_dt


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("2026-08-02T12:00:00+00:00", "2026-08-02"),
        ("2026-08-02T12:00:00", "2026-08-02"),
        ("not-a-date", None),
    ],
)
def test_parse_dt_handles_iso_and_invalid(raw: str | None, expected: str | None) -> None:
    parsed = _parse_dt(raw)
    if expected is None:
        assert parsed is None
    else:
        assert parsed is not None
        assert parsed.date().isoformat() == expected


@pytest.mark.asyncio
async def test_eval_seed_agent_returns_404_when_eval_mode_disabled() -> None:
    from qortia import config
    from qortia.eval_router import seed_eval_agent

    old = config.settings.eval_mode
    config.settings.eval_mode = False
    try:
        with pytest.raises(HTTPException) as exc:
            await seed_eval_agent(uuid4(), uuid4())
        assert exc.value.status_code == 404
    finally:
        config.settings.eval_mode = old


@pytest.mark.asyncio
async def test_eval_recall_returns_404_when_eval_mode_disabled() -> None:
    from qortia import config
    from qortia.eval_router import eval_recall

    old = config.settings.eval_mode
    config.settings.eval_mode = False
    try:
        with pytest.raises(HTTPException) as exc:
            await eval_recall("query", uuid4(), uuid4())
        assert exc.value.status_code == 404
    finally:
        config.settings.eval_mode = old


@pytest.mark.asyncio
async def test_cli_create_tenant_prints_id(capsys: pytest.CaptureFixture[str]) -> None:
    from qortia.provisioning import _cli_create_tenant

    tenant_id = uuid4()
    pool = MagicMock()
    pool.close = AsyncMock()
    with (
        patch("qortia.provisioning.asyncpg.create_pool", AsyncMock(return_value=pool)),
        patch("qortia.provisioning.create_tenant", AsyncMock(return_value=tenant_id)),
    ):
        await _cli_create_tenant(argparse.Namespace(name="Acme"))

    assert str(tenant_id) in capsys.readouterr().out


@pytest.mark.asyncio
async def test_cli_create_agent_prints_id(capsys: pytest.CaptureFixture[str]) -> None:
    from qortia.provisioning import _cli_create_agent

    agent_id = uuid4()
    tenant_id = uuid4()
    pool = MagicMock()
    pool.close = AsyncMock()
    with (
        patch("qortia.provisioning.asyncpg.create_pool", AsyncMock(return_value=pool)),
        patch("qortia.provisioning.create_agent", AsyncMock(return_value=agent_id)),
    ):
        await _cli_create_agent(
            argparse.Namespace(tenant=str(tenant_id), clearance="internal", division="all")
        )

    assert str(agent_id) in capsys.readouterr().out


@pytest.mark.asyncio
async def test_cli_issue_key_prints_plaintext_once(capsys: pytest.CaptureFixture[str]) -> None:
    from qortia.provisioning import _cli_issue_key

    tenant_id = uuid4()
    pool = MagicMock()
    pool.close = AsyncMock()
    with (
        patch("qortia.provisioning.asyncpg.create_pool", AsyncMock(return_value=pool)),
        patch("qortia.provisioning.issue_api_key", AsyncMock(return_value="qortia_sk_test")),
    ):
        await _cli_issue_key(argparse.Namespace(tenant=str(tenant_id)))

    output = capsys.readouterr().out
    assert "qortia_sk_test" in output
    assert "Store this now" in output


def test_provisioning_main_dispatches_subcommand() -> None:
    from qortia.provisioning import main

    handler = MagicMock()
    namespace = argparse.Namespace(func=handler, command="create-tenant")
    with (
        patch("qortia.provisioning.asyncio.run") as run_mock,
        patch.object(argparse.ArgumentParser, "parse_args", return_value=namespace),
    ):
        main()
    run_mock.assert_called_once()
    handler.assert_called_once_with(namespace)
