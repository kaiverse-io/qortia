#!/usr/bin/env python3
"""Minimal OpenAI-compatible /embeddings server for local E2E tests.

Returns deterministic 1024-dim unit vectors derived from input text so the
qortia worker + recall path can be exercised without Ollama/BGE-M3.
Not for production quality measurement.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

DIM = 1024
app = FastAPI(title="mock-litellm-embeddings")


class EmbedRequest(BaseModel):
    model: str = "bge-m3"
    input: str | list[str]


class EmbedData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbedResponse(BaseModel):
    object: str = "list"
    data: list[EmbedData]
    model: str
    usage: dict[str, int] = Field(default_factory=dict)


def _embed(text: str, dim: int = DIM) -> list[float]:
    tokens = re.findall(r"[a-z0-9]+", text.lower()) or ["empty"]
    vec = [0.0] * dim
    for tok in tokens:
        h = hashlib.sha256(tok.encode()).digest()
        for i in range(0, 32, 4):
            idx = int.from_bytes(h[i : i + 4], "big") % dim
            sign = 1.0 if h[i] % 2 == 0 else -1.0
            vec[idx] += sign
    # mix in full-text hash so unique strings differ
    full = hashlib.sha256(text.encode()).digest()
    for i in range(dim):
        vec[i] += (full[i % 32] / 255.0 - 0.5) * 0.01
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/embeddings", response_model=EmbedResponse)
def embeddings(body: EmbedRequest) -> EmbedResponse:
    inputs = [body.input] if isinstance(body.input, str) else list(body.input)
    data = [
        EmbedData(index=i, embedding=_embed(text))
        for i, text in enumerate(inputs)
    ]
    return EmbedResponse(
        data=data,
        model=body.model,
        usage={"prompt_tokens": sum(len(t.split()) for t in inputs), "total_tokens": 0},
    )


@app.post("/chat/completions")
def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
    """Stub for reflect/rerank paths that accidentally hit the mock."""
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "[]"},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, log_level="info")
