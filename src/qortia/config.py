"""Standalone runtime settings — env-driven, no external secrets manager.

Replaces the old host-platform `app.config.settings` (which pulled some values
from Vault). Qortia is meant to be dropped into any Postgres+pgvector instance,
so everything here is a plain environment variable with a sane local default.

Tunables that were previously hardcoded module constants (recall's RRF k,
result-count limits, the response char budget — `recall_helpers.py`, found
while adding a cap /v1/recall never had) are now readable from an optional
TOML file too, since "change the env var and restart the container" is a
heavier operation than editing a versioned, reviewable config file for values
that are tuning knobs, not secrets or per-deployment topology (those stay
env-only — `database_url`, `litellm_api_key`, `qortia_admin_token`, etc. are
deliberately not exposed to the file layer at all, so a config file can never
become a second place a credential leaks into git).

Precedence: env var > TOML file > code default. The file is optional — no
`QORTIA_CONFIG_FILE` and no `qortia.toml` in the working directory reproduces
today's pure-env behaviour exactly. Env vars stay the final override, matching
every other setting here (the operator layer for deployment-specific tuning),
not superseded by a versioned file a deploy might not have picked up yet.
"""

from __future__ import annotations

import json
import logging
import os
import tomllib
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Not `.toml`'s pyproject.toml convention (that file is packaging metadata,
# not runtime config) — a name that reads as "runtime tuning knobs" on sight.
_DEFAULT_CONFIG_FILE = "qortia.toml"


def _load_toml_file(path: str) -> dict[str, Any]:
    """Best-effort: a missing file is normal (env-only deployments, the
    common case), a malformed one logs and is treated as absent rather than
    crashing startup over a config typo."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning({"event": "config_file_invalid", "path": path, "error": str(exc)})
        return {}


def _env_str(name: str, default: Any) -> str:
    raw = os.environ.get(name)
    return raw if raw is not None else str(default)


def _env_tenant_keys(name: str) -> dict[str, str]:
    """Parse QORTIA_LITELLM_TENANT_KEYS JSON object {tenant_id: virtual_key}."""
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning({"event": "litellm_tenant_keys_invalid_json", "error": str(exc)})
        return {}
    if not isinstance(parsed, dict):
        logger.warning({"event": "litellm_tenant_keys_not_object"})
        return {}
    out: dict[str, str] = {}
    for k, v in parsed.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def _env_float(name: str, default: Any) -> float:
    # `default` may be a TOML-file value now, not always an already-`float`
    # code default — coerce it through the same `float()` the env-var path
    # uses rather than trusting its type, so `rrf_k = "60"` (string) in the
    # file still becomes a real float/int and only genuinely unparseable
    # file content (`rrf_k = "sixty"`) fails loudly at startup.
    raw = os.environ.get(name)
    return float(raw) if raw is not None else float(default)


def _env_int(name: str, default: Any) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else int(default)


def _env_bool(name: str, default: Any) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    database_url: str = "postgresql://localhost:5432/qortia"
    litellm_url: str = "http://localhost:4000"
    litellm_api_key: str = ""
    # Optional JSON map tenant_id → LiteLLM virtual key (ADR-003). Empty = shared key.
    litellm_tenant_keys: dict[str, str] = field(default_factory=dict)
    # Defaults match migrations/V1 vector(1024) + multilingual BGE-M3 (ADR-002).
    embedding_model: str = "bge-m3"
    embedding_dimension: int = 1024
    qortia_dedup_similarity_threshold: float = 0.95  # calibrated for BGE-M3 1024-dim, ADR-105
    embedding_cache_max_size: int = 10_000
    embedding_cache_ttl_seconds: int = 3600
    eval_mode: bool = False
    # Platform-admin bearer token gating qortia.admin_router (/v1/admin/*), ADR-004.
    # Empty (default) = router stays unmounted and its auth dependency 404s outright.
    qortia_admin_token: str = ""
    # Empty (default) = reranking is skipped, not attempted-then-failed — see
    # recall_rerank._llm_rerank's guard. Deliberately no vendor default here:
    # unlike embedding_model (structurally pinned to migrations/V1's
    # vector(1024), ADR-002), nothing about this value is load-bearing —
    # it was a specific commercial model (anthropic/claude-3-haiku) baked
    # into an open, "drop into any Postgres+pgvector instance" project with
    # no corresponding credential ever configured by default, so every
    # rerank=True call paid a network round-trip guaranteed to fail and
    # logged a misleading rerank_failed warning for what both is intentional
    # ("no rerank model configured") and had nothing to do with a failure.
    rerank_model: str = ""
    reflection_threshold: int = 10
    idle_reflection_interval_s: float = 300.0
    idle_reflection_window_h: float = 1.0
    # ── recall tuning — was hardcoded in recall_helpers.py, file-configurable
    # from here down (`[recall]` table); see the module docstring above for why
    # these and not database_url/litellm_api_key/qortia_admin_token.
    recall_rrf_k: int = 60  # reciprocal-rank-fusion constant, _rrf_fuse
    recall_search_fetch_multiplier: int = 2  # over-fetch factor before RRF/MMR trims to the limit
    recall_private_result_limit: int = 20
    recall_org_result_limit: int = 10
    recall_knowledge_result_limit: int = 16
    # Applied when a /v1/recall caller doesn't pass max_chars explicitly — 0 or
    # negative means no default cap (today's original, unbounded behaviour).
    # Measured unbounded: 38,961 chars/call average against a real 276-document
    # corpus (agnova's evals/run_scale_eval_qortia.py) for 5.5% precision — an
    # explicit per-request max_chars was not enough by itself, since a caller
    # has to already know to ask for it; this is the fix applying by default.
    recall_default_max_chars: int = 8_000


def load_settings() -> Settings:
    d = Settings()
    file_path = os.environ.get("QORTIA_CONFIG_FILE", _DEFAULT_CONFIG_FILE)
    file_values = _load_toml_file(file_path)
    recall_file = file_values.get("recall", {})
    if not isinstance(recall_file, dict):
        logger.warning({"event": "config_file_recall_table_not_object", "path": file_path})
        recall_file = {}

    def _file_or(key: str, fallback: object) -> Any:
        return file_values.get(key, fallback)

    def _recall_file_or(key: str, fallback: object) -> Any:
        return recall_file.get(key, fallback)

    return Settings(
        database_url=_env_str("QORTIA_DATABASE_URL", d.database_url),
        litellm_url=_env_str("QORTIA_LITELLM_URL", d.litellm_url),
        litellm_api_key=_env_str("QORTIA_LITELLM_API_KEY", d.litellm_api_key),
        litellm_tenant_keys=_env_tenant_keys("QORTIA_LITELLM_TENANT_KEYS"),
        embedding_model=_env_str(
            "QORTIA_EMBEDDING_MODEL", _file_or("embedding_model", d.embedding_model)
        ),
        embedding_dimension=_env_int(
            "QORTIA_EMBEDDING_DIMENSION", _file_or("embedding_dimension", d.embedding_dimension)
        ),
        qortia_dedup_similarity_threshold=_env_float(
            "QORTIA_DEDUP_SIMILARITY_THRESHOLD",
            _file_or("dedup_similarity_threshold", d.qortia_dedup_similarity_threshold),
        ),
        embedding_cache_max_size=_env_int(
            "QORTIA_EMBEDDING_CACHE_MAX_SIZE",
            _file_or("embedding_cache_max_size", d.embedding_cache_max_size),
        ),
        embedding_cache_ttl_seconds=_env_int(
            "QORTIA_EMBEDDING_CACHE_TTL_SECONDS",
            _file_or("embedding_cache_ttl_seconds", d.embedding_cache_ttl_seconds),
        ),
        eval_mode=_env_bool("QORTIA_EVAL_MODE", d.eval_mode),
        qortia_admin_token=_env_str("QORTIA_ADMIN_TOKEN", d.qortia_admin_token),
        rerank_model=_env_str("QORTIA_RERANK_MODEL", _file_or("rerank_model", d.rerank_model)),
        reflection_threshold=_env_int(
            "QORTIA_REFLECTION_THRESHOLD", _file_or("reflection_threshold", d.reflection_threshold)
        ),
        idle_reflection_interval_s=_env_float(
            "QORTIA_IDLE_REFLECTION_INTERVAL_S",
            _file_or("idle_reflection_interval_s", d.idle_reflection_interval_s),
        ),
        idle_reflection_window_h=_env_float(
            "QORTIA_IDLE_REFLECTION_WINDOW_H",
            _file_or("idle_reflection_window_h", d.idle_reflection_window_h),
        ),
        recall_rrf_k=_env_int("QORTIA_RECALL_RRF_K", _recall_file_or("rrf_k", d.recall_rrf_k)),
        recall_search_fetch_multiplier=_env_int(
            "QORTIA_RECALL_SEARCH_FETCH_MULTIPLIER",
            _recall_file_or("search_fetch_multiplier", d.recall_search_fetch_multiplier),
        ),
        recall_private_result_limit=_env_int(
            "QORTIA_RECALL_PRIVATE_RESULT_LIMIT",
            _recall_file_or("private_result_limit", d.recall_private_result_limit),
        ),
        recall_org_result_limit=_env_int(
            "QORTIA_RECALL_ORG_RESULT_LIMIT",
            _recall_file_or("org_result_limit", d.recall_org_result_limit),
        ),
        recall_knowledge_result_limit=_env_int(
            "QORTIA_RECALL_KNOWLEDGE_RESULT_LIMIT",
            _recall_file_or("knowledge_result_limit", d.recall_knowledge_result_limit),
        ),
        recall_default_max_chars=_env_int(
            "QORTIA_RECALL_DEFAULT_MAX_CHARS",
            _recall_file_or("default_max_chars", d.recall_default_max_chars),
        ),
    )


settings = load_settings()
