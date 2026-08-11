"""Unit tests for the consolidated /chat/completions client (qortia.chat)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from qortia.chat import ChatCompletionError, chat_completion


def _response(*, status_code: int = 200, body: object | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body if body is not None else {}
    return response


@pytest.mark.asyncio
async def test_chat_completion_returns_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _response(body={"choices": [{"message": {"content": "hello there"}}]})
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("qortia.chat.get_litellm_client", lambda: client)

    content = await chat_completion(model="test-model", prompt="hi", litellm_key="key", timeout=5.0)

    assert content == "hello there"
    kwargs = client.post.await_args.kwargs
    assert kwargs["json"]["model"] == "test-model"
    assert kwargs["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert "response_format" not in kwargs["json"]
    assert "max_tokens" not in kwargs["json"]


@pytest.mark.asyncio
async def test_chat_completion_json_mode_and_max_tokens_set_body_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(body={"choices": [{"message": {"content": "{}"}}]})
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("qortia.chat.get_litellm_client", lambda: client)

    await chat_completion(
        model="m", prompt="p", litellm_key="key", timeout=5.0, json_mode=True, max_tokens=200
    )

    body = client.post.await_args.kwargs["json"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 200


@pytest.mark.asyncio
async def test_chat_completion_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _response(status_code=503)
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("qortia.chat.get_litellm_client", lambda: client)

    with pytest.raises(ChatCompletionError, match="LiteLLM error: 503"):
        await chat_completion(model="m", prompt="p", litellm_key="key", timeout=5.0)


@pytest.mark.asyncio
async def test_chat_completion_wraps_bare_timeout_error_with_a_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """asyncio.timeout() raises a bare TimeoutError() with no args — found
    live under load (entity_summary_update_failed/rerank_failed logging
    error: '' for every timeout, indistinguishable from any other silent
    failure). ChatCompletionError is the one error type this module
    promises callers; it should say what happened, not repeat the same
    emptiness."""
    client = MagicMock()
    client.post = AsyncMock(side_effect=TimeoutError())
    monkeypatch.setattr("qortia.chat.get_litellm_client", lambda: client)

    with pytest.raises(ChatCompletionError, match="timed out after 5.0s"):
        await chat_completion(model="m", prompt="p", litellm_key="key", timeout=5.0)


@pytest.mark.asyncio
async def test_chat_completion_wraps_httpx_timeout_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
    monkeypatch.setattr("qortia.chat.get_litellm_client", lambda: client)

    with pytest.raises(ChatCompletionError, match="timed out after 5.0s"):
        await chat_completion(model="m", prompt="p", litellm_key="key", timeout=5.0)


@pytest.mark.asyncio
async def test_chat_completion_raises_on_missing_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _response(body={"usage": {}})
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("qortia.chat.get_litellm_client", lambda: client)

    with pytest.raises(ChatCompletionError, match="malformed LiteLLM response"):
        await chat_completion(model="m", prompt="p", litellm_key="key", timeout=5.0)


@pytest.mark.asyncio
async def test_chat_completion_logs_usage_only_when_log_event_given(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _response(
        body={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
    )
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("qortia.chat.get_litellm_client", lambda: client)

    with caplog.at_level("INFO"):
        await chat_completion(model="m", prompt="p", litellm_key="key", timeout=5.0)
    assert not any("qortia_chat_test_event" in str(r.msg) for r in caplog.records)

    caplog.clear()
    with caplog.at_level("INFO"):
        await chat_completion(
            model="m",
            prompt="p",
            litellm_key="key",
            timeout=5.0,
            log_event="qortia_chat_test_event",
            tenant_id="tid-1",
        )
    logged = [r.msg for r in caplog.records if isinstance(r.msg, dict)]
    assert any(
        entry.get("event") == "qortia_chat_test_event"
        and entry.get("prompt_tokens") == 7
        and entry.get("completion_tokens") == 3
        and entry.get("qortia.tenant_id") == "tid-1"
        for entry in logged
    )
