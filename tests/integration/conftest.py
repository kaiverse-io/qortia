"""Integration test fixtures.

Qortia's own FastAPI app runs in-process via httpx ASGITransport. A real
Postgres+pgvector container is spun up automatically via testcontainers — no
external docker-compose or just target required, no Vault, no JWT keypairs.

If these tests fail with a connection error on a nested/sandboxed Docker
host (not a normal machine or CI), see "Running tests" in CONTRIBUTING.md.

Event loop design
-----------------
The FastAPI app runs on a dedicated background thread (loop.run_forever).
Sync fixtures call the background loop via run_coroutine_threadsafe().
pytest-asyncio (mode=auto) manages its own per-test loop; the two never mix.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

# qortia.* modules are deliberately NOT imported at module level here: the
# first import of qortia.config binds `settings` (env-var-driven) into every
# module that does `from qortia.config import settings` — including
# qortia.db, qortia.common, qortia.auth. Postgres's actual host/port is only
# known once _postgres_container has started, so every qortia.* import must
# happen after QORTIA_DATABASE_URL is set in _app_and_loop, not before.

MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "migrations"
MOCK_EMBEDDING = [0.1] * 1024
# Fixed test value for QORTIA_ADMIN_TOKEN — set for the whole session so
# qortia.admin_router (ADR-004) is mounted on the shared app_client, same as
# QORTIA_EVAL_MODE gates qortia.eval_router below. Not a real secret.
ADMIN_TOKEN = "qortia-admin-test-token"  # noqa: S105

POSTGRES_URL: str = ""
POSTGRES_SUPERUSER_URL: str = ""

# ── Container fixtures (session-scoped) ───────────────────────────────────────


@pytest.fixture(scope="session")
def _postgres_container():
    container = (
        DockerContainer("pgvector/pgvector:pg16")
        .with_env("POSTGRES_DB", "qortia_test")
        .with_env("POSTGRES_USER", "postgres")
        .with_env("POSTGRES_PASSWORD", "test")
        .with_exposed_ports(5432)
        .with_tmpfs_mount("/var/lib/postgresql/data")
        .waiting_for(
            LogMessageWaitStrategy(
                "database system is ready to accept connections"
            ).with_startup_timeout(60)
        )
    )
    container.start()
    try:
        yield container
    finally:
        container.stop(delete_volume=True, force=True)


@pytest.fixture(scope="session")
def _pg_urls(_postgres_container: DockerContainer):
    host = _postgres_container.get_container_host_ip()
    port = _postgres_container.get_exposed_port(5432)
    superuser = f"postgresql://postgres:test@{host}:{port}/qortia_test"
    app_user = f"postgresql://qortia_platform:qortia_platform@{host}:{port}/qortia_test"
    return superuser, app_user


# ── Session setup (runs once) ─────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def setup_infrastructure(_pg_urls: tuple[str, str]) -> None:
    """Apply the standalone migration as superuser, then point the app role at it."""
    superuser_url, app_url = _pg_urls

    global POSTGRES_URL, POSTGRES_SUPERUSER_URL
    POSTGRES_URL = app_url
    POSTGRES_SUPERUSER_URL = superuser_url

    async def _run() -> None:
        su = await asyncpg.connect(superuser_url)
        try:
            for migration in sorted(MIGRATIONS_DIR.glob("V*.sql")):
                await su.execute(migration.read_text())
        finally:
            await su.close()

    asyncio.run(_run())


# ── App client (session-scoped) ───────────────────────────────────────────────


def _mock_litellm(path: str, **_: Any) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    if "/embeddings" in path:
        resp.json.return_value = {"data": [{"embedding": MOCK_EMBEDDING}]}
    elif "/chat/completions" in path:
        resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"reflections": ['
                            '{"action": "CREATE", "type": "mental_model", '
                            '"content": "Test insight", "importance": 0.8}, '
                            '{"action": "CREATE", "type": "lesson", '
                            '"content": "Test lesson", "importance": 0.9}, '
                            '{"action": "RETAIN", "id": "00000000-0000-0000-0000-000000000001"}'
                            "]}"
                        )
                    }
                }
            ]
        }
    return resp


@pytest.fixture(scope="session")
def _app_and_loop(setup_infrastructure: None, _pg_urls: tuple[str, str]):  # type: ignore[return]
    """Start qortia's own FastAPI app once for the entire test session.

    The app loop runs on a dedicated background thread (loop.run_forever), the
    same pattern used across this project's other session-scoped async setup
    (mirrors evals/ harness conventions) — sync fixtures submit work via
    run_coroutine_threadsafe().
    """
    _, app_url = _pg_urls
    os.environ["QORTIA_DATABASE_URL"] = app_url
    os.environ["QORTIA_LITELLM_URL"] = "http://mock-litellm:4000"
    os.environ["QORTIA_EVAL_MODE"] = "false"
    os.environ["QORTIA_ADMIN_TOKEN"] = ADMIN_TOKEN

    # qortia.* modules read settings via `config.settings.<field>` (a live
    # attribute lookup on the qortia.config module), not a frozen
    # `from qortia.config import settings` binding — so reassigning
    # qortia.config.settings here correctly reaches every already-imported
    # consumer too, regardless of whether some other test module (e.g. a
    # tests/unit/*.py file collected earlier in the same pytest session)
    # imported a qortia.* module before these env vars were set.
    import qortia.config as config_mod

    config_mod.settings = config_mod.load_settings()

    _ready = threading.Event()
    _shutdown = threading.Event()
    started: dict[str, Any] = {}

    def _thread_main() -> None:
        async def _run() -> None:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=lambda p, **kw: _mock_litellm(p, **kw))
            mock_http.get = AsyncMock(side_effect=lambda p, **kw: _mock_litellm(p, **kw))
            mock_http.aclose = AsyncMock()

            import qortia.app as app_mod
            import qortia.common as common_mod

            async with app_mod.lifespan(app_mod.app):
                # Swap in the mock LiteLLM client after real startup has run.
                common_mod._litellm_client = mock_http
                ac = AsyncClient(
                    transport=ASGITransport(app=app_mod.app),
                    base_url="http://test",
                    timeout=10.0,
                )
                await ac.__aenter__()
                started["ac"] = ac
                _ready.set()

                while not _shutdown.is_set():
                    await asyncio.sleep(0.05)

                await ac.__aexit__(None, None, None)

            for task in asyncio.all_tasks():
                task.cancel()
            await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        started["loop"] = loop
        try:
            loop.run_until_complete(_run())
        except Exception as exc:  # noqa: BLE001
            started["startup_error"] = exc
            _ready.set()
        loop.close()

    thread = threading.Thread(target=_thread_main, daemon=True)
    thread.start()
    if not _ready.wait(timeout=60):
        raise RuntimeError("App startup timed out after 60s")
    if "startup_error" in started:
        raise started["startup_error"]

    loop = started["loop"]

    async def _ping() -> str:
        return "pong"

    ping_result = asyncio.run_coroutine_threadsafe(_ping(), loop).result(timeout=10)
    assert ping_result == "pong", f"Loop not processing tasks: {ping_result}"
    yield loop, started["ac"]

    _shutdown.set()
    thread.join(timeout=20)


@pytest.fixture(scope="session")
def app_client(_app_and_loop) -> AsyncClient:  # type: ignore[return]
    _, ac = _app_and_loop
    return ac


@pytest.fixture(scope="session")
def _session_loop(_app_and_loop) -> asyncio.AbstractEventLoop:
    loop, _ = _app_and_loop
    return loop


def _call(loop: asyncio.AbstractEventLoop, coro: Any) -> Any:
    """Submit a coroutine to the app's background loop and block until done."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=60)


# ── Auth header helpers ────────────────────────────────────────────────────────
# API keys authenticate a tenant; the caller names which agent it's acting as
# via X-Agent-Id (see qortia.auth.require_agent). No JWT, no Vault — a key is
# issued on demand against the real DB and cached per tenant for the session.
#
# This runs on its own throwaway event loop rather than the app's shared
# _session_loop: it only ever touches its own independent asyncpg pool, never
# app state, so there's no reason to route it through the background thread.

_key_cache: dict[str, str] = {}


def _get_or_issue_key(tenant_id: str, *, ensure_tenant: bool = False) -> str:
    if tenant_id in _key_cache:
        return _key_cache[tenant_id]

    async def _issue() -> str:
        from qortia.provisioning import issue_api_key

        pool = await asyncpg.create_pool(POSTGRES_SUPERUSER_URL)
        try:
            if ensure_tenant:
                await pool.execute(
                    "INSERT INTO qortia_tenants (id) VALUES ($1) ON CONFLICT DO NOTHING",
                    tenant_id,
                )
            return await issue_api_key(pool, tenant_id)
        finally:
            await pool.close()

    key = asyncio.run(_issue())
    _key_cache[tenant_id] = key
    return key


def fresh_agent_headers() -> dict[str, str]:
    """Headers for a tenant/agent pair that has never been provisioned —
    the tenant (and its key) exist, but the agent row does not, so
    require_agent's tenant/agent binding check rejects it."""
    tenant_id = str(uuid4())
    agent_id = str(uuid4())
    api_key = _get_or_issue_key(tenant_id, ensure_tenant=True)
    return {"Authorization": f"Bearer {api_key}", "X-Agent-Id": agent_id}


def make_agent_headers(agent_id: str, tenant_id: str) -> dict[str, str]:
    """Headers for an already-provisioned agent under an already-provisioned tenant."""
    api_key = _get_or_issue_key(str(tenant_id))
    return {"Authorization": f"Bearer {api_key}", "X-Agent-Id": str(agent_id)}


def create_active_agent(committed_conn: Any, tenant_id: str, role: str = "engineer") -> str:
    """Insert an active agent into the DB and track it for cleanup."""
    aid = str(uuid4())
    committed_conn.execute(
        "INSERT INTO qortia_agents (id, tenant_id, name, role, status) VALUES ($1, $2, $3, $4, $5)",
        aid,
        tenant_id,
        f"agent-{aid[:8]}",
        role,
        "active",
    )
    committed_conn.track("qortia_agents", aid)
    return aid


# ── DB fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def pg_url(setup_infrastructure: None) -> str:
    return POSTGRES_URL


@pytest.fixture(scope="session")
def pg_superuser_url(setup_infrastructure: None) -> str:
    return POSTGRES_SUPERUSER_URL


@pytest.fixture
def conn(setup_infrastructure: None, _session_loop: asyncio.AbstractEventLoop):  # type: ignore[return]
    """Per-test DB connection with rollback."""

    async def _acquire():
        c = await asyncpg.connect(POSTGRES_URL)
        tr = c.transaction()
        await tr.start()
        return c, tr

    c, tr = _call(_session_loop, _acquire())
    yield c
    _call(_session_loop, tr.rollback())
    _call(_session_loop, c.close())


@pytest.fixture
def committed_conn(setup_infrastructure: None, _session_loop: asyncio.AbstractEventLoop):  # type: ignore[return]
    """Per-test DB connection that commits immediately. Tracks rows for cleanup."""
    inserted: list[tuple[str, str]] = []

    c = _call(_session_loop, asyncpg.connect(POSTGRES_URL))

    class _Conn:
        def execute(self, q, *a):
            return _call(_session_loop, c.execute(q, *a))

        def fetchval(self, q, *a):
            return _call(_session_loop, c.fetchval(q, *a))

        def fetch(self, q, *a):
            return _call(_session_loop, c.fetch(q, *a))

        def track(self, table: str, row_id: str) -> str:
            inserted.append((table, row_id))
            return row_id

    proxy = _Conn()
    yield proxy

    async def _cleanup():
        for table, row_id in reversed(inserted):
            try:
                await c.execute(f"DELETE FROM {table} WHERE id = $1", row_id)  # noqa: S608
            except Exception:
                pass
        await c.close()

    _call(_session_loop, _cleanup())


@pytest.fixture
def tenant_id(committed_conn, _session_loop: asyncio.AbstractEventLoop) -> str:
    tid = str(uuid4())
    committed_conn.execute(
        "INSERT INTO qortia_tenants (id, name) VALUES ($1, $2)",
        tid,
        f"test-{tid[:8]}",
    )
    committed_conn.track("qortia_tenants", tid)
    return tid


@pytest.fixture
def agent_id(committed_conn, tenant_id: str, _session_loop: asyncio.AbstractEventLoop) -> str:
    return create_active_agent(committed_conn, tenant_id)
