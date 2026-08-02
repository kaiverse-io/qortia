"""Shared helpers for qortia integration tests."""

from __future__ import annotations

import pytest

from tests.integration.conftest import MOCK_EMBEDDING

VECTOR_LITERAL = "[" + ",".join(str(v) for v in MOCK_EMBEDDING) + "]"


def memory_payload(memory_type: str, content: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {"type": memory_type, "content": content}
    payload.update(extra)
    return payload


def patch_entity_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    entities = [("Qortia", "ORG"), ("PostgreSQL", "PRODUCT")]

    import qortia.knowledge as knowledge_mod
    import qortia.remember as remember_mod

    monkeypatch.setattr(knowledge_mod, "extract_entities_with_types", lambda *_a, **_k: entities)
    monkeypatch.setattr(knowledge_mod, "extract_entities", lambda *_a, **_k: ["Qortia"])
    monkeypatch.setattr(remember_mod, "extract_entities_with_types", lambda *_a, **_k: entities)


def patch_knowledge_index(monkeypatch: pytest.MonkeyPatch) -> None:
    import qortia.knowledge as knowledge_mod

    monkeypatch.setattr(
        knowledge_mod,
        "extract_index_fields",
        lambda heading, text, lang="en": {
            "index_summary": text[:200],
            "index_entities": '["Qortia", "PostgreSQL"]',
            "index_questions": f'["{heading or "Qortia retrieval"}"]',
        },
    )
