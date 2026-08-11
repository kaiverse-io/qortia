"""Consolidated chat-completion client for Qortia.

Same shape as qortia.embeddings for the `/embeddings` endpoint: one module
owns the LiteLLM/OpenAI-compatible request/response mechanics for
`/chat/completions` (build the body, POST with auth headers, pull
`choices[0].message.content` back out, log token usage) so the three
call sites that need an LLM completion — recall_rerank._llm_rerank,
entity_graph._update_entity_summary, reflect._call_litellm_reflect —
don't each re-implement it slightly differently. What differs per call
site (prompt content, response_format, timeout budget, and — critically —
whether a failure should be swallowed-with-fallback or propagated as an
HTTP error) stays at the call site; only the mechanical request/response
shape is shared here.

Do not POST `/chat/completions` from other modules — call `chat_completion`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from qortia.auth import litellm_auth_headers
from qortia.common import get_litellm_client

logger = logging.getLogger(__name__)


class ChatCompletionError(RuntimeError):
    """Non-2xx response or a body missing the expected `choices[0].message.content` shape.

    Deliberately not a subclass of httpx's own errors — callers already
    branch on catching this broadly (fall back) vs. letting it propagate
    (surface as an HTTP 5xx), and that decision belongs to the caller, not
    to this module guessing which policy fits.
    """


async def chat_completion(
    *,
    model: str,
    prompt: str,
    litellm_key: str,
    timeout: float,
    json_mode: bool = False,
    max_tokens: int | None = None,
    log_event: str | None = None,
    tenant_id: str | None = None,
) -> str:
    """POST /chat/completions with a single user-role prompt; return the message content.

    Raises ChatCompletionError on a non-200 response or an unparseable body —
    callers decide whether to catch it (fallback) or let it propagate.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    # +5s outer bound beyond httpx's own timeout — matches the buffer each
    # call site used before consolidation (35/30, 20/15, 125/120).
    async with asyncio.timeout(timeout + 5.0):
        resp = await get_litellm_client().post(
            "/chat/completions",
            headers=litellm_auth_headers(litellm_key),
            json=body,
            timeout=timeout,
        )

    if resp.status_code != 200:
        raise ChatCompletionError(f"LiteLLM error: {resp.status_code}")

    try:
        raw_resp = resp.json()
        content: str = raw_resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise ChatCompletionError(f"malformed LiteLLM response: {exc}") from exc

    if log_event:
        usage = raw_resp.get("usage", {})
        logger.info(
            {
                "event": log_event,
                "qortia.tenant_id": tenant_id,
                "model": model,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        )
    return content
