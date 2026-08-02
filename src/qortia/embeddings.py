"""Consolidated embedding client for Qortia.

Industry-standard shape for an OSS memory layer:
- One module owns LiteLLM `/embeddings` calls (write path + query path).
- Model and dimension are env-configurable; defaults match the shipped schema
  (BGE-M3 @ 1024 — multilingual, works with spaCy NER which is separate).
- Query path uses `embedding_cache`; write/worker path does not (unique texts).
- Schema dimension is fixed by migration V1 (`vector(1024)`). Changing dim
  requires a new migration + full re-embed — see docs/how-to/embeddings.md.

Do not POST `/embeddings` from other modules — call `embed_text` / `embed_query`.
"""

from __future__ import annotations

import logging
from uuid import UUID

from qortia import config
from qortia.auth import get_litellm_key
from qortia.common import get_litellm_client
from qortia.embedding_cache import get_cached_embedding, put_cached_embedding

logger = logging.getLogger(__name__)

# Must match migrations/V1__initial_schema.sql `vector(1024)` columns.
# Runtime settings.embedding_dimension must equal this unless you ship a
# dimension-change migration and re-embed every row.
SCHEMA_EMBEDDING_DIMENSION = 1024


def embedding_model() -> str:
    return config.settings.embedding_model


def embedding_dimension() -> int:
    return config.settings.embedding_dimension


async def embed_text(
    text: str,
    litellm_key: str,
    *,
    tenant_id: str | None = None,
    timeout: float = 30.0,
) -> list[float]:
    """Embed a single string via the configured LiteLLM model. No cache.

    When ``tenant_id`` is set, the OpenAI-compatible ``user`` field and
    ``metadata.qortia.tenant_id`` are sent so a LiteLLM gateway can attribute
    cost/traces per tenant (ADR-003). Engines that ignore these fields are fine.
    """
    body: dict[str, object] = {"model": embedding_model(), "input": text}
    if tenant_id:
        body["user"] = tenant_id
        body["metadata"] = {"qortia.tenant_id": tenant_id}
    resp = await get_litellm_client().post(
        "/embeddings",
        headers={"Authorization": f"Bearer {litellm_key}"},
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    embedding: list[float] = resp.json()["data"][0]["embedding"]
    if len(embedding) != embedding_dimension():
        raise RuntimeError(
            f"Embedding dimension mismatch for {embedding_model()}: "
            f"configured {embedding_dimension()}, got {len(embedding)}."
        )
    return embedding


async def embed_query(query: str, tenant_id: UUID, lang: str = "en") -> list[float] | None:
    """Embed a recall query (cached per tenant/lang/model). Returns None on failure."""
    tid = str(tenant_id)
    effective_lang = lang or "en"

    cached = get_cached_embedding(query, tid, effective_lang)
    if cached is not None:
        return cached

    try:
        litellm_key = await get_litellm_key(tid)
        embedding = await embed_text(query, litellm_key, tenant_id=tid, timeout=10.0)
        put_cached_embedding(query, tid, effective_lang, embedding)
        return embedding
    except Exception as exc:
        logger.warning({"event": "recall_embed_failed", "error": str(exc)})
        try:
            from qortia.telemetry import qortia_recall_degraded

            qortia_recall_degraded.add(1, {"reason": "embed_failed", "qortia.tenant_id": tid})
        except Exception:  # noqa: S110
            pass
        return None


async def validate_embedding_config() -> None:
    """Fail fast if settings disagree with schema or the live model.

    Called from app lifespan when a LiteLLM API key is configured. Local boots
    without a key skip the live probe (OSS / unit-test friendly) but still
    enforce settings.dimension == SCHEMA_EMBEDDING_DIMENSION.
    """
    configured = embedding_dimension()
    if configured != SCHEMA_EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"QORTIA_EMBEDDING_DIMENSION={configured} does not match schema "
            f"vector({SCHEMA_EMBEDDING_DIMENSION}) from migrations/V1. "
            "Change the model only at the same dimension, or ship a migration that "
            "alters all embedding columns, drops/rebuilds HNSW indexes, NULLs existing "
            "vectors, and re-runs the embedding worker. See docs/how-to/embeddings.md."
        )

    if not config.settings.litellm_api_key:
        logger.info(
            {
                "event": "embedding_validate_skipped",
                "reason": "no_litellm_api_key",
                "model": embedding_model(),
                "dimension": configured,
            }
        )
        return

    from qortia.auth import get_platform_embed_key

    embed_key = get_platform_embed_key()
    try:
        embedding = await embed_text(
            "dimension check",
            embed_key,
            tenant_id="platform",
            timeout=60.0,
        )
    except Exception as exc:
        raise RuntimeError(f"Embedding model unavailable ({embedding_model()}): {exc}") from exc

    logger.info(
        {
            "event": "embedding_validated",
            "model": embedding_model(),
            "dimension": len(embedding),
        }
    )
