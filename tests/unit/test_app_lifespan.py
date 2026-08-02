"""App lifespan wires LiteLLM, pool, embedding validate, spaCy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_lifespan_validates_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia import app as app_mod

    monkeypatch.setattr(app_mod, "init_litellm_client", lambda: None)
    monkeypatch.setattr(app_mod, "init_main_pool", AsyncMock())
    validate = AsyncMock()
    monkeypatch.setattr(app_mod, "validate_embedding_config", validate)
    monkeypatch.setattr(app_mod, "load_spacy_model", lambda: None)
    monkeypatch.setattr(app_mod, "close_litellm_client", AsyncMock())
    monkeypatch.setattr(app_mod, "close_main_pool", AsyncMock())

    async with app_mod.lifespan(MagicMock()):
        pass

    validate.assert_awaited_once()
