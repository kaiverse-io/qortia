"""Tests for qortia-worker CLI entrypoint."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_run_starts_selected_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia import workers

    started: list[str] = []

    async def fake_embed() -> None:
        started.append("embed")

    async def fake_archive() -> None:
        started.append("archive")

    monkeypatch.setattr(workers, "init_litellm_client", lambda: None)
    monkeypatch.setattr(workers, "init_main_pool", AsyncMock())
    monkeypatch.setattr(workers, "validate_embedding_config", AsyncMock())
    monkeypatch.setattr(workers, "close_litellm_client", AsyncMock())
    monkeypatch.setattr(workers, "close_main_pool", AsyncMock())
    monkeypatch.setattr(workers, "_WORKERS", {"embed": fake_embed, "archive": fake_archive})

    await workers._run(["embed", "archive"])
    assert set(started) == {"embed", "archive"}


@pytest.mark.asyncio
async def test_run_rejects_unknown_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia import workers

    monkeypatch.setattr(workers, "init_litellm_client", lambda: None)
    monkeypatch.setattr(workers, "init_main_pool", AsyncMock())
    monkeypatch.setattr(workers, "validate_embedding_config", AsyncMock())
    with pytest.raises(SystemExit, match="unknown worker"):
        await workers._run(["nope"])


def test_main_invokes_asyncio_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia import workers

    called: list[object] = []

    def fake_run(coro: object) -> None:
        called.append(coro)
        # close the coroutine to avoid "never awaited" warnings
        if asyncio.iscoroutine(coro):
            coro.close()

    monkeypatch.setattr(workers.asyncio, "run", fake_run)
    workers.main([])
    assert called
