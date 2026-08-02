"""Standalone runtime settings — env-driven, no external secrets manager.

Replaces the old host-platform `app.config.settings` (which pulled some values
from Vault). Qortia is meant to be dropped into any Postgres+pgvector instance,
so everything here is a plain environment variable with a sane local default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    database_url: str = "postgresql://localhost:5432/qortia"
    litellm_url: str = "http://localhost:4000"
    litellm_api_key: str = ""
    # Defaults match migrations/V1 vector(1024) + multilingual BGE-M3 (ADR-002).
    embedding_model: str = "bge-m3"
    embedding_dimension: int = 1024
    qortia_dedup_similarity_threshold: float = 0.95  # calibrated for BGE-M3 1024-dim, ADR-105
    embedding_cache_max_size: int = 10_000
    embedding_cache_ttl_seconds: int = 3600
    eval_mode: bool = False
    rerank_model: str = "anthropic/claude-3-haiku-20240307"
    reflection_threshold: int = 10
    idle_reflection_interval_s: float = 300.0
    idle_reflection_window_h: float = 1.0


def load_settings() -> Settings:
    d = Settings()
    return Settings(
        database_url=_env_str("QORTIA_DATABASE_URL", d.database_url),
        litellm_url=_env_str("QORTIA_LITELLM_URL", d.litellm_url),
        litellm_api_key=_env_str("QORTIA_LITELLM_API_KEY", d.litellm_api_key),
        embedding_model=_env_str("QORTIA_EMBEDDING_MODEL", d.embedding_model),
        embedding_dimension=_env_int("QORTIA_EMBEDDING_DIMENSION", d.embedding_dimension),
        qortia_dedup_similarity_threshold=_env_float(
            "QORTIA_DEDUP_SIMILARITY_THRESHOLD", d.qortia_dedup_similarity_threshold
        ),
        embedding_cache_max_size=_env_int(
            "QORTIA_EMBEDDING_CACHE_MAX_SIZE", d.embedding_cache_max_size
        ),
        embedding_cache_ttl_seconds=_env_int(
            "QORTIA_EMBEDDING_CACHE_TTL_SECONDS", d.embedding_cache_ttl_seconds
        ),
        eval_mode=_env_bool("QORTIA_EVAL_MODE", d.eval_mode),
        rerank_model=_env_str("QORTIA_RERANK_MODEL", d.rerank_model),
        reflection_threshold=_env_int("QORTIA_REFLECTION_THRESHOLD", d.reflection_threshold),
        idle_reflection_interval_s=_env_float(
            "QORTIA_IDLE_REFLECTION_INTERVAL_S", d.idle_reflection_interval_s
        ),
        idle_reflection_window_h=_env_float(
            "QORTIA_IDLE_REFLECTION_WINDOW_H", d.idle_reflection_window_h
        ),
    )


settings = load_settings()
