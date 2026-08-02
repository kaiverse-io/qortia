"""Unit tests for consolidated embedding client + config surface."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_embed_text_posts_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import qortia.config as config
    from qortia.embeddings import embed_text

    monkeypatch.setattr(config.settings, "embedding_model", "bge-m3")
    monkeypatch.setattr(config.settings, "embedding_dimension", 1024)
    response = MagicMock()
    response.json.return_value = {"data": [{"embedding": [0.1] * 1024}]}
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("qortia.embeddings.get_litellm_client", lambda: client)

    out = await embed_text("hello", "key")
    assert len(out) == 1024
    kwargs = client.post.await_args.kwargs
    assert kwargs["json"]["model"] == "bge-m3"


@pytest.mark.asyncio
async def test_validate_rejects_dimension_not_matching_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qortia.config as config
    from qortia.embeddings import validate_embedding_config

    monkeypatch.setattr(config.settings, "embedding_dimension", 768)
    with pytest.raises(RuntimeError, match="does not match schema"):
        await validate_embedding_config()


@pytest.mark.asyncio
async def test_validate_skips_live_probe_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qortia.config as config
    from qortia.embeddings import validate_embedding_config

    monkeypatch.setattr(config.settings, "embedding_dimension", 1024)
    monkeypatch.setattr(config.settings, "litellm_api_key", "")
    await validate_embedding_config()  # must not raise


@pytest.mark.asyncio
async def test_cache_key_includes_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import qortia.config as config
    from qortia.embedding_cache import _make_cache_key, clear_all_caches

    clear_all_caches()
    monkeypatch.setattr(config.settings, "embedding_model", "bge-m3")
    a = _make_cache_key("q", "t1", "en")
    monkeypatch.setattr(config.settings, "embedding_model", "other-model")
    b = _make_cache_key("q", "t1", "en")
    assert a != b


@pytest.mark.asyncio
async def test_embed_query_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia.embeddings import embed_query

    tenant = uuid4()
    cached = [0.3] * 1024
    monkeypatch.setattr("qortia.embeddings.get_cached_embedding", lambda *_a, **_k: cached)
    post = AsyncMock()
    monkeypatch.setattr("qortia.embeddings.get_litellm_client", lambda: MagicMock(post=post))
    assert await embed_query("x", tenant) == cached
    post.assert_not_called()
