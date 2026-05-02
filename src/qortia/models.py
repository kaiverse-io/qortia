from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_validator,
    model_validator,
)


# ── remember ────────────────────────────────────────────────

IMPORTANCE: dict[str, float] = {
    "episodic": 0.3,
    "experiential": 0.6,
    "mental_model": 0.8,
    "decision": 0.9,
    "lesson": 0.95,
    "short_term": 0.1,
}


class MemoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "episodic", "experiential", "mental_model", "decision", "lesson", "short_term"
    ]
    content: str
    source_task_id: UUID | None = None
    metadata: dict | None = None  # type: ignore[type-arg]
    ttl_seconds: int | None = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v

    @field_validator("metadata")
    @classmethod
    def metadata_is_object(cls, v: object) -> object:
        if v is not None and not isinstance(v, dict):
            raise ValueError("metadata must be a JSON object")
        return v

    @model_validator(mode="after")
    def validate_ttl(self) -> "MemoryItem":
        if self.ttl_seconds is not None and self.type != "short_term":
            raise ValueError("ttl_seconds is only valid for short_term memories")
        if self.type == "short_term" and (
            self.ttl_seconds is None or self.ttl_seconds <= 0
        ):
            raise ValueError("short_term memories require a positive ttl_seconds")
        return self


class RememberRequest(BaseModel):
    memories: list[MemoryItem]

    @field_validator("memories")
    @classmethod
    def memories_not_empty(cls, v: list) -> list:  # type: ignore[type-arg]
        if not v:
            raise ValueError("memories array must not be empty")
        return v


class RememberResponse(BaseModel):
    ids: list[str]


# ── remember-org ─────────────────────────────────────────────


class RememberOrgRequest(BaseModel):
    type: Literal["handoff", "process", "decision_log"]
    title: str
    content: str


class RememberOrgResponse(BaseModel):
    id: str


# ── forget ───────────────────────────────────────────────────


class ForgetRequest(BaseModel):
    id: UUID


class ForgetResponse(BaseModel):
    id: str


# ── context ──────────────────────────────────────────────────


class MemoryEntry(BaseModel):
    title: str | None = None
    content: str
    importance: float | None = None


class ContextMemories(BaseModel):
    mental_models: list[MemoryEntry]
    decisions: list[MemoryEntry]
    lessons: list[MemoryEntry]


class ContextResponse(BaseModel):
    org_chart: list[MemoryEntry]
    processes: list[MemoryEntry]
    handoffs: list[MemoryEntry]
    weekly_summary: MemoryEntry | None
    memories: ContextMemories


# ── reflect ──────────────────────────────────────────────────


class ReflectResponse(BaseModel):
    memories_written: int
    reflection_counter: int


# ── recall ───────────────────────────────────────────────────


class RecallRequest(BaseModel):
    query: str
    scope: Literal["private", "org", "knowledge", "all", "archive"] = "all"
    type: (
        Literal[
            "episodic",
            "experiential",
            "mental_model",
            "decision",
            "lesson",
            "short_term",
        ]
        | None
    ) = None
    rerank: bool = False
    entities: list[str] | None = None
    as_of: datetime | None = None  # point-in-time recall (16j)

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v

    @field_validator("entities")
    @classmethod
    def entities_valid(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            if not v:
                return None  # empty list treated as null
            for el in v:
                if not isinstance(el, str) or not el.strip():
                    raise ValueError("each entity must be a non-empty string")
        return v


class RecallResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    type: str
    scope: Literal["private", "org", "knowledge"]
    content: str
    importance: float | None = None
    created_at: str
    entity_summary: str | None = None
    linked_via: str | None = (
        None  # ID of the result that surfaced this via cross-link (16i)
    )
    valid_from: str | None = None  # when this fact became true (16j)
    valid_until: str | None = (
        None  # when this fact was superseded; None = currently valid (16j)
    )

    # Internal ranking signals — never serialised (Q95)
    _recall_count: int = PrivateAttr(default=0)
    _last_recalled_at: datetime | None = PrivateAttr(default=None)
    _score: float = PrivateAttr(default=0.0)
    _embedding: list[float] = PrivateAttr(default_factory=list)


class RecallResponse(BaseModel):
    results: list[RecallResult]


# ── knowledge ────────────────────────────────────────────────


class KnowledgeIngestRequest(BaseModel):
    source_type: Literal["file", "url", "transcript", "note"]
    source_path: str
    content: str
    metadata: dict | None = None  # type: ignore[type-arg]
