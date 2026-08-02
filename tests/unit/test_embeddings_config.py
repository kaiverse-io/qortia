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

    out = await embed_text("hello", "key", tenant_id="tid-1")
    assert len(out) == 1024
    kwargs = client.post.await_args.kwargs
    assert kwargs["json"]["model"] == "bge-m3"
    assert kwargs["json"]["user"] == "tid-1"
    assert kwargs["json"]["metadata"] == {"qortia.tenant_id": "tid-1"}


@pytest.mark.asyncio
async def test_get_litellm_key_prefers_tenant_map(monkeypatch: pytest.MonkeyPatch) -> None:
    import qortia.config as config
    from qortia.auth import get_litellm_key

    tid = str(uuid4())
    monkeypatch.setattr(config.settings, "litellm_api_key", "sk-shared")
    monkeypatch.setattr(config.settings, "litellm_tenant_keys", {tid: "sk-tenant"})
    assert await get_litellm_key(tid) == "sk-tenant"
    assert await get_litellm_key(str(uuid4())) == "sk-shared"


def test_env_tenant_keys_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia.config import _env_tenant_keys

    monkeypatch.setenv("QORTIA_LITELLM_TENANT_KEYS", '{"a":"sk-a","b":"sk-b"}')
    assert _env_tenant_keys("QORTIA_LITELLM_TENANT_KEYS") == {"a": "sk-a", "b": "sk-b"}
    monkeypatch.setenv("QORTIA_LITELLM_TENANT_KEYS", "not-json")
    assert _env_tenant_keys("QORTIA_LITELLM_TENANT_KEYS") == {}


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
