"""Semantic embedding cache for the recall pipeline.

Option A implementation: in-process per-tenant TTL+LRU cache.
Eliminates redundant embedding calls for identical queries within
the same tenant within the TTL window.

Thread safety: all cache operations are protected by a global lock.
The cache is per-process; each platform replica maintains its own.

Cache key includes tenant_id to prevent cross-tenant cache hits —
different tenants may use different embedding models via LiteLLM routing.
"""

from __future__ import annotations

import hashlib
import logging
import threading

from cachetools import TTLCache

from app.config import settings

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()
_tenant_caches: dict[str, TTLCache[str, list[float]]] = {}


def _get_tenant_cache(tenant_id: str) -> TTLCache[str, list[float]]:
    """Return or create a per-tenant TTL cache instance.

    Must be called under _cache_lock or by functions that acquire it.
    """
    if tenant_id not in _tenant_caches:
        _tenant_caches[tenant_id] = TTLCache(
            maxsize=settings.embedding_cache_max_size,
            ttl=settings.embedding_cache_ttl_seconds,
        )
    return _tenant_caches[tenant_id]


def _make_cache_key(query: str, tenant_id: str, lang: str) -> str:
    """Produce a deterministic cache key from normalised query text.

    Key structure: SHA-256 of "{tenant_id}:{lang}:{normalised_query}"
    Tenant ID is included in the hash to make collisions across tenants
    impossible even if an attacker crafts inputs.
    """
    normalised = query.lower().strip()
    raw = f"{tenant_id}:{lang}:{normalised}".encode()
    return hashlib.sha256(raw).hexdigest()


def get_cached_embedding(query: str, tenant_id: str, lang: str) -> list[float] | None:
    """Look up a cached embedding. Returns None on miss."""
    key = _make_cache_key(query, tenant_id, lang)
    with _cache_lock:
        cache = _get_tenant_cache(tenant_id)
        result: list[float] | None = cache.get(key)
    if result is not None:
        logger.debug({"event": "embedding_cache_hit", "tenant_id": tenant_id})
    return result


def put_cached_embedding(
    query: str, tenant_id: str, lang: str, embedding: list[float]
) -> None:
    """Store an embedding in the tenant cache."""
    key = _make_cache_key(query, tenant_id, lang)
    with _cache_lock:
        cache = _get_tenant_cache(tenant_id)
        cache[key] = embedding


def clear_all_caches() -> None:
    """Clear all tenant caches. Used in tests."""
    with _cache_lock:
        _tenant_caches.clear()


def get_cache_stats() -> dict[str, int]:
    """Return cache statistics for observability.

    Cardinality budget: O(tenants) which is bounded.
    """
    with _cache_lock:
        return {
            "tenant_count": len(_tenant_caches),
            "total_entries": sum(len(c) for c in _tenant_caches.values()),
        }
