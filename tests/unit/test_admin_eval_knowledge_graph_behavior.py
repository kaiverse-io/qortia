from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from qortia import config
from qortia.auth import AgentIdentity
from qortia.models import KnowledgeIngestRequest

TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
AGENT_ID = UUID("00000000-0000-0000-0000-000000000002")


class _AsyncContext:
    def __init__(self, value: object = None) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _FakePool:
    def __init__(self, conn: object | None = None) -> None:
        self.conn = conn or MagicMock()
        self.closed = False
        self.execute = AsyncMock()
        self.fetchval = AsyncMock()

    def acquire(self) -> _AsyncContext:
        return _AsyncContext(self.conn)

    async def close(self) -> None:
        self.closed = True


class _Ent:
    def __init__(self, text: str, label: str) -> None:
        self.text = text
        self.label_ = label


class _Span:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeDoc:
    def __init__(self) -> None:
        self.ents = [
            _Ent("Qortia", "ORG"),
            _Ent("PostgreSQL", "PRODUCT"),
            _Ent("Ignored", "DATE"),
        ]
        self.sents = [_Span("Qortia stores memory."), _Span("PostgreSQL backs recall.")]
        self.noun_chunks = [_Span("portable memory"), _Span("very long noun chunk ignored")]


class _FakeNLP:
    def __call__(self, _text: str) -> _FakeDoc:
        return _FakeDoc()


def _tenant_tx(conn: MagicMock) -> _AsyncContext:
    return _AsyncContext(conn)


def test_provisioning_main_dispatches_admin_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from qortia import provisioning

    pool = _FakePool()
    monkeypatch.setattr(provisioning.asyncpg, "create_pool", AsyncMock(return_value=pool))

    for argv in (
        ["qortia-admin", "create-tenant", "--name", "Acme"],
        [
            "qortia-admin",
            "create-agent",
            "--tenant",
            str(TENANT_ID),
            "--clearance",
            "external",
            "--division",
            "sales",
        ],
        ["qortia-admin", "issue-key", "--tenant", str(TENANT_ID)],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        provisioning.main()

    out = capsys.readouterr().out
    assert "tenant_id:" in out
    assert "agent_id:" in out
    assert "api_key: qortia_sk_" in out
    assert pool.closed is True


@pytest.mark.asyncio
async def test_provisioning_cli_helpers_close_pools(monkeypatch: pytest.MonkeyPatch) -> None:
    from qortia import provisioning

    pools = [_FakePool(), _FakePool(), _FakePool()]
    monkeypatch.setattr(provisioning.asyncpg, "create_pool", AsyncMock(side_effect=pools))

    await provisioning._cli_create_tenant(argparse.Namespace(name="Tenant"))
    await provisioning._cli_create_agent(
        argparse.Namespace(tenant=str(TENANT_ID), name=None, clearance="internal", division="all")
    )
    await provisioning._cli_issue_key(argparse.Namespace(tenant=str(TENANT_ID)))

    assert all(pool.closed for pool in pools)


def test_knowledge_spacy_entity_extraction_and_index_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qortia.knowledge as knowledge

    fake_spacy = types.SimpleNamespace(load=MagicMock(return_value=_FakeNLP()))
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    monkeypatch.setattr(knowledge, "_indic_pipelines", {})

    knowledge.load_spacy_model()

    assert knowledge.extract_entities("Qortia uses PostgreSQL") == ["Qortia", "PostgreSQL"]
    assert knowledge.extract_entities("unsupported", lang="xx") == ["Qortia", "PostgreSQL"]
    assert knowledge.extract_entities_with_types("Qortia") == [
        ("Qortia", "ORG"),
        ("PostgreSQL", "PRODUCT"),
    ]
    fields = knowledge.extract_index_fields("Memory", "Qortia stores memory")
    assert fields["index_summary"] == "Qortia stores memory. PostgreSQL backs recall."
    assert "Qortia" in fields["index_entities"]
    assert "portable memory" in fields["index_questions"]


def test_knowledge_indic_entity_routing_and_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qortia.knowledge as knowledge

    class IndicDoc:
        ents = [_Ent("राम", "PER"), _Ent("दिल्ली", "LOC"), _Ent("noise", "MISC")]
        sents = [_Span("राम दिल्ली गया।")]

    class IndicNLP:
        def __call__(self, _text: str) -> IndicDoc:
            return IndicDoc()

    monkeypatch.setattr(knowledge, "_indic_pipelines", {})
    monkeypatch.setitem(sys.modules, "spacy", types.SimpleNamespace(load=lambda _model: IndicNLP()))

    assert knowledge.extract_entities("राम दिल्ली गया", lang="hi") == ["राम", "दिल्ली"]
    assert knowledge.extract_entities_with_types("राम दिल्ली गया", lang="hi") == [
        ("राम", "PERSON"),
        ("दिल्ली", "GPE"),
    ]
    indic_fields = knowledge.extract_index_fields("याद", "राम दिल्ली गया", lang="hi")
    assert "राम" in json.loads(indic_fields["index_entities"])

    monkeypatch.setattr(knowledge, "_indic_pipelines", {})

    def fail_load(_model: str) -> object:
        raise OSError("missing model")

    monkeypatch.setitem(sys.modules, "spacy", types.SimpleNamespace(load=fail_load))
    with pytest.raises(OSError):
        knowledge._get_indic_pipeline("hi")


def test_knowledge_section_splitting_covers_heading_and_paragraph_branches() -> None:
    from qortia.knowledge import split_into_sections

    intro = " ".join(["intro"] * 50)
    big = "\n\n".join(" ".join([f"paragraph{i}"] * 800) for i in range(4))
    content = (
        f"{intro}\n\n"
        "## Large Section\n"
        f"{big}\n\n"
        "## Small Section\n"
        "too short\n\n"
        "## Final Section\n" + " ".join(["final"] * 50)
    )

    sections = split_into_sections(content)

    assert sections[0]["heading"] == "Introduction"
    assert any(section["heading"] == "Large Section" for section in sections)
    assert "too short" in sections[-2]["text"]
    assert sections[-1]["heading"] == "Final Section"
    assert split_into_sections("too short") == []


@pytest.mark.asyncio
async def test_ingest_knowledge_creates_replaces_and_dedupes_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qortia.knowledge as knowledge

    agent = AgentIdentity(agent_id=AGENT_ID, tenant_id=TENANT_ID)
    body = KnowledgeIngestRequest(
        source_type="note",
        source_path="docs/qortia.md",
        content="Qortia knowledge section has enough durable recall words for indexing.",
    )

    section = {
        "heading": "Memory",
        "text": "Qortia knowledge section has enough durable recall words for indexing.",
    }
    conn = MagicMock()
    conn.fetchval = AsyncMock(side_effect=["chief", None])
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock()
    monkeypatch.setattr(knowledge, "tenant_transaction", lambda *_a, **_k: _tenant_tx(conn))
    monkeypatch.setattr(knowledge, "get_main_pool", lambda: object())
    monkeypatch.setattr(knowledge, "assert_agent_active", AsyncMock())
    monkeypatch.setattr(knowledge, "split_into_sections", lambda _content: [section])
    monkeypatch.setattr(
        knowledge,
        "extract_index_fields",
        lambda *_a, **_k: {
            "index_summary": "summary",
            "index_entities": '["Qortia"]',
            "index_questions": '["Memory"]',
        },
    )

    created = await knowledge.ingest_knowledge(body, agent)
    assert created["sections_created"] == 1

    existing_hash = conn.execute.await_args_list[0].args[6]
    conn.fetchval = AsyncMock(side_effect=["chief"])
    conn.fetch = AsyncMock(return_value=[{"chunk_index": 0, "content_hash": existing_hash}])
    deduped = await knowledge.ingest_knowledge(body, agent)
    assert deduped["sections_deduped"] == 1

    conn.fetchval = AsyncMock(side_effect=["chief", "cached-embedding"])
    conn.fetch = AsyncMock(return_value=[{"chunk_index": 0, "content_hash": "old"}])
    replaced = await knowledge.ingest_knowledge(body, agent)
    assert replaced["sections_deduped"] == 1
    assert any(
        "DELETE FROM org_knowledge" in call.args[0]
        for call in conn.execute.await_args_list
        if call.args
    )


@pytest.mark.asyncio
async def test_knowledge_ingest_and_delete_require_chief_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qortia.knowledge as knowledge

    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value="engineer")
    conn.execute = AsyncMock(return_value="DELETE 2")
    monkeypatch.setattr(knowledge, "tenant_transaction", lambda *_a, **_k: _tenant_tx(conn))
    monkeypatch.setattr(knowledge, "get_main_pool", lambda: object())
    monkeypatch.setattr(knowledge, "assert_agent_active", AsyncMock())
    agent = AgentIdentity(agent_id=AGENT_ID, tenant_id=TENANT_ID)

    with pytest.raises(HTTPException) as ingest_error:
        await knowledge.ingest_knowledge(
            KnowledgeIngestRequest(
                source_type="note",
                source_path="x",
                content="enough words for the pydantic request body to be accepted",
            ),
            agent,
        )
    assert ingest_error.value.status_code == 403

    with pytest.raises(HTTPException) as delete_error:
        await knowledge.delete_knowledge("x", agent)
    assert delete_error.value.status_code == 403

    conn.fetchval = AsyncMock(return_value="chief")
    deleted = await knowledge.delete_knowledge("x", agent)
    assert deleted == {"source_path": "x", "chunks_deleted": 2}


@pytest.mark.asyncio
async def test_weekly_summary_cycle_and_tenant_summary_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qortia.knowledge as knowledge

    real_summarise_tenant = knowledge._summarise_tenant
    today = datetime.now(UTC).date().weekday()
    tenant_for_today = UUID(int=0)
    tenant_other_day = UUID(int=0)
    while True:
        digest_day = (
            int(__import__("hashlib").md5(str(tenant_for_today).encode()).hexdigest(), 16) % 7
        )
        if digest_day == today:
            break
        tenant_for_today = UUID(int=tenant_for_today.int + 1)
    while True:
        digest_day = (
            int(__import__("hashlib").md5(str(tenant_other_day).encode()).hexdigest(), 16) % 7
        )
        if digest_day != today:
            break
        tenant_other_day = UUID(int=tenant_other_day.int + 1)

    summarise = AsyncMock()
    conn = MagicMock()
    conn.fetch = AsyncMock(
        return_value=[
            {"id": tenant_for_today, "weekly_summary_last_run_at": None},
            {"id": tenant_other_day, "weekly_summary_last_run_at": None},
        ]
    )
    monkeypatch.setattr(knowledge, "get_main_pool", lambda: _FakePool(conn))
    monkeypatch.setattr(knowledge, "_summarise_tenant", summarise)

    await knowledge._run_weekly_summary_cycle()
    summarise.assert_awaited_once()

    handoffs = [
        {
            "title": "handoff",
            "content": f"content {idx}",
            "created_at": datetime(2026, 1, idx + 1, tzinfo=UTC),
            "lang": "en",
            "agent_name": "Kai",
        }
        for idx in range(3)
    ]
    summary = knowledge.build_weekly_summary(handoffs)
    assert "[Kai | 2026-01-03]" in summary

    monkeypatch.setattr(knowledge, "_summarise_tenant", real_summarise_tenant)
    summary_conn = MagicMock()
    summary_conn.transaction.return_value = _AsyncContext()
    summary_conn.fetchrow = AsyncMock(return_value={"id": TENANT_ID})
    summary_conn.fetch = AsyncMock(return_value=handoffs)
    summary_conn.execute = AsyncMock()
    monkeypatch.setattr(knowledge, "get_main_pool", lambda: _FakePool(summary_conn))

    await knowledge._summarise_tenant(TENANT_ID, None)
    assert summary_conn.execute.await_count == 2

    summary_conn.fetchrow = AsyncMock(return_value=None)
    await knowledge._summarise_tenant(TENANT_ID, None)
    summary_conn.fetchrow = AsyncMock(return_value={"id": TENANT_ID})
    await knowledge._summarise_tenant(TENANT_ID, datetime.now(UTC) - timedelta(days=1))
    summary_conn.fetch = AsyncMock(return_value=handoffs[:2])
    await knowledge._summarise_tenant(TENANT_ID, None)


@pytest.mark.asyncio
async def test_eval_router_guards_and_temporal_seed_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qortia.config as config
    import qortia.eval_router as eval_router

    assert eval_router._parse_dt(None) is None
    assert eval_router._parse_dt("not-a-date") is None
    assert eval_router._parse_dt("2026-01-01T00:00:00").tzinfo is not None

    old_eval_mode = config.settings.eval_mode
    config.settings.eval_mode = False
    with pytest.raises(HTTPException):
        await eval_router.seed_eval_agent(AGENT_ID, TENANT_ID)
    with pytest.raises(HTTPException):
        await eval_router.seed_eval_memory(
            eval_router.SeedMemoryRequest(
                agent_id=AGENT_ID,
                tenant_id=TENANT_ID,
                content="Qortia seed memory",
            )
        )
    with pytest.raises(HTTPException):
        await eval_router.eval_recall("Qortia", TENANT_ID, AGENT_ID)
    with pytest.raises(HTTPException):
        await eval_router.eval_reflect(TENANT_ID, AGENT_ID)

    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.fetchval = AsyncMock(return_value=uuid4())
    monkeypatch.setattr(eval_router, "get_main_pool", lambda: _FakePool(conn))
    monkeypatch.setattr(eval_router, "extract_entities_with_types", lambda *_a, **_k: [])
    monkeypatch.setattr(eval_router, "provision_eval_litellm_key", AsyncMock())

    config.settings.eval_mode = True
    try:
        seeded = await eval_router.seed_eval_agent(AGENT_ID, TENANT_ID, role="chief")
        assert seeded["status"] == "seeded"
        memory = await eval_router.seed_eval_memory(
            eval_router.SeedMemoryRequest(
                agent_id=AGENT_ID,
                tenant_id=TENANT_ID,
                content="Qortia seeded memory",
                valid_from="2026-01-01T00:00:00+00:00",
                valid_until="2026-02-01T00:00:00+00:00",
            )
        )
        assert memory["status"] == "seeded"
    finally:
        config.settings.eval_mode = old_eval_mode


@pytest.mark.asyncio
async def test_eval_router_wrappers_call_underlying_qortia_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qortia.config as config
    import qortia.eval_router as eval_router

    old_eval_mode = config.settings.eval_mode
    config.settings.eval_mode = True
    try:
        recall_response = MagicMock()
        recall_response.model_dump.return_value = {"results": []}
        reflect_response = MagicMock()
        reflect_response.model_dump.return_value = {"memories_written": 0}
        remember_org_response = MagicMock(id=str(uuid4()))

        monkeypatch.setattr("qortia.recall.recall", AsyncMock(return_value=recall_response))
        monkeypatch.setattr("qortia.reflect.reflect", AsyncMock(return_value=reflect_response))
        monkeypatch.setattr(
            "qortia.remember.remember_org", AsyncMock(return_value=remember_org_response)
        )
        monkeypatch.setattr(
            "qortia.knowledge.ingest_knowledge", AsyncMock(return_value={"ok": True})
        )
        monkeypatch.setattr(
            eval_router, "get_main_pool", lambda: _FakePool(MagicMock(execute=AsyncMock()))
        )

        assert await eval_router.eval_recall("Qortia", TENANT_ID, AGENT_ID) == {"results": []}
        assert await eval_router.eval_recall_full(
            eval_router.RecallRequestFull(query="Qortia"), TENANT_ID, AGENT_ID
        ) == {"results": []}
        assert await eval_router.eval_reflect(TENANT_ID, AGENT_ID) == {"memories_written": 0}
        assert "id" in await eval_router.eval_remember_org(
            eval_router.RememberOrgRequestBody(
                type="process",
                title="Process",
                content="Qortia process content has enough words to pass validation today",
                valid_from="2026-01-01T00:00:00+00:00",
                valid_until="2026-02-01T00:00:00+00:00",
            ),
            TENANT_ID,
            AGENT_ID,
        )
        assert await eval_router.eval_ingest_knowledge(
            eval_router.KnowledgeIngestBody(
                source_type="note",
                source_path="qortia.md",
                content="Qortia content",
            ),
            TENANT_ID,
            AGENT_ID,
        ) == {"ok": True}
    finally:
        config.settings.eval_mode = old_eval_mode


@pytest.mark.asyncio
async def test_entity_summary_dedup_and_update_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qortia.entity_graph as graph

    assert (
        await graph._update_entity_summary(None, "new memory content", "key")
        == "new memory content"
    )

    # rerank_model defaults to "" (not configured, skip) — must configure one
    # explicitly here or the empty-model guard returns "old" unchanged
    # without ever reaching the mocked LLM call this line asserts against.
    monkeypatch.setattr(config.settings, "rerank_model", "test-model")
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"choices": [{"message": {"content": "updated summary"}}]}
    client = MagicMock(post=AsyncMock(return_value=response))
    monkeypatch.setattr("qortia.chat.get_litellm_client", lambda: client)
    assert await graph._update_entity_summary("old", "new", "key") == "updated summary"
    client.post = AsyncMock(side_effect=RuntimeError("down"))
    assert await graph._update_entity_summary("old", "new", "key") == "old"

    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value={"id": uuid4(), "similarity": 0.99})
    conn.execute = AsyncMock()
    monkeypatch.setattr(graph, "get_main_pool", lambda: _FakePool(conn))
    await graph._maybe_dedup_memory(uuid4(), [0.1], TENANT_ID, AGENT_ID, "episodic")
    conn.execute.assert_awaited()

    conn.fetchrow = AsyncMock(return_value={"link_count": 1, "summary": None})
    await graph._maybe_update_entity_summary(conn, TENANT_ID, AGENT_ID, "Qortia", "content", False)
    await graph._maybe_update_entity_summary(conn, TENANT_ID, None, "Qortia", "content", True)

    conn.fetchrow = AsyncMock(return_value={"link_count": 3, "summary": "old"})
    monkeypatch.setattr(graph, "get_litellm_key", AsyncMock(return_value="key"))
    monkeypatch.setattr(graph, "_update_entity_summary", AsyncMock(return_value="merged"))
    await graph._maybe_update_entity_summary(conn, TENANT_ID, AGENT_ID, "Qortia", "content", False)

    conn.fetchrow = AsyncMock(side_effect=RuntimeError("db down"))
    await graph._maybe_update_entity_summary(conn, TENANT_ID, AGENT_ID, "Qortia", "content", False)
