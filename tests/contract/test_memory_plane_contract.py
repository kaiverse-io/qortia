"""Contract tests for Qortia's side of the agnova memory-plane seam.

Each asserts a property a *client* (agnova's QortiaMemoryBackend, or any
future one) depends on. None require a live database — they read the app's
own route table and response-construction source. F3/F4 were additionally
verified live: a real remember() -> recall() round trip against this app on
qortia_net, and a direct SQL read of hindsight_memories/qortia_outcome_records
after 12 recalls under distinct X-Work-Order-Id headers, confirmed
confidence_multiplier frozen at 1 and qortia_outcome_records empty
database-wide. See the memory-plane proposal for the full experiment logs.

All five findings here are now fixed (each docstring says how) — this file
is the regression suite pinning them, not the open list anymore. See the
sibling agnova and aither repos for the parts of the seam that live there.
"""

from __future__ import annotations

import inspect

# ── F1 · /v1/context cannot be asked for a budget ───────────────────────────


def test_context_endpoint_accepts_a_budget() -> None:
    """A client with a bounded context window must be able to ask for a bundle
    that fits, instead of fetching everything and truncating blind.

    Fixed: GET /v1/context now takes `budget` (chars) and keeps the highest-
    importance entries across mental_models/decisions/lessons combined, in
    importance order, dropping whole entries rather than slicing any one of
    them. See _budget_memories() in qortia/remember.py.
    """
    from qortia.remember import get_context

    params = set(inspect.signature(get_context).parameters) - {"agent"}
    assert params & {"budget", "limit", "max_tokens"}, (
        "GET /v1/context takes no size parameter at all "
        f"(params={sorted(inspect.signature(get_context).parameters)}); "
        "the client must fetch the whole bundle and byte-slice it"
    )


# ── F2 · decisions ship without the importance the ranking depends on ──────


def test_every_context_entry_carries_importance() -> None:
    """ContextResponse.MemoryEntry has an `importance` field. It must be
    populated for decisions too, not just mental_models and lessons, or a
    client cannot sort the bundle by importance even if it wants to.

    Fixed: the decisions query now selects `importance` (it was always
    populated at write time via IMPORTANCE["decision"] — the query just
    never selected it).
    """
    from qortia import remember as remember_mod

    src = inspect.getsource(remember_mod.get_context)
    built = [ln.strip() for ln in src.splitlines() if "MemoryEntry(content=" in ln]
    without_importance = [ln for ln in built if "importance=" not in ln]
    assert not without_importance, (
        "these context entries are built without importance, so the field is "
        f"None over the wire: {without_importance}"
    )


# ── F3 · the outcome-feedback loop has no write surface ────────────────────
#
# Verified live: 12 recalls against a real stored `lesson` memory (importance
# 0.95), each under a distinct X-Work-Order-Id, correctly wrote 12 rows to
# qortia_session_reads. confidence_multiplier stayed exactly 1, and
# qortia_outcome_records had 0 rows *database-wide* — not just for this
# memory. The read half of the loop works; nothing exercises the write half.


def test_an_outcome_can_be_reported_over_http() -> None:
    """`confidence_multiplier` is read on every recall and multiplied into the
    score. `_record_work_order_outcome` is the only thing that writes it —
    confirmed live it was pinned at its insert-time default of 1.0 forever.

    Fixed: POST /v1/outcome (recall.py's report_outcome) now calls it.

    Checks qortia.recall.router directly, not qortia.app — importing the
    real app singleton here would freeze its one-time admin_router mounting
    decision (`if config.settings.qortia_admin_token`, app.py) before
    tests/integration/conftest.py's _app_and_loop fixture ever gets to set
    QORTIA_ADMIN_TOKEN, since this file collects/runs before tests/integration/
    in the same pytest session — breaking test_provisioning_api.py collaterally
    (see that file's own comment on the same fixture for the intended
    single-importer design this respects).
    """
    from qortia.recall import router as qortia_router

    outcome_routes = {
        r.path for r in qortia_router.routes if "outcome" in r.path or "feedback" in r.path
    }
    assert outcome_routes, (
        "no route accepts a work-order outcome; the differentiator "
        f"cannot be exercised by any client. published surface="
        f"{sorted(r.path for r in qortia_router.routes)}"
    )


def test_record_outcome_has_a_non_test_caller() -> None:
    """Guards the same gap from the other side: dead code that ranking depends on.

    Fixed alongside the endpoint above: report_outcome() is now a real caller.
    """
    import subprocess

    # git grep, not grep -r: searches tracked files only, so build artifacts
    # like .mypy_cache's serialized symbol tables (which also contain this
    # name) can't produce a false "it has a caller" positive.
    out = subprocess.run(
        ["git", "grep", "-n", "_record_work_order_outcome", "--", "src/", "evals/"],
        capture_output=True,
        text=True,
    ).stdout
    call_sites = [ln for ln in out.splitlines() if "async def _record_work_order_outcome" not in ln]
    assert call_sites, (
        "_record_work_order_outcome is defined in src/qortia/recall.py and called "
        "from nowhere in src/ or evals/ — its only caller is a unit test"
    )


# ── F4 · a bundle missing two of three buckets looks identical to a full one ─


def test_context_signals_whether_consolidation_has_run() -> None:
    """mental_models and lessons are both filtered `is_consolidated = true`.
    If reflect never runs, both arrays are empty forever, so the response
    must say something about why.

    Fixed: ContextResponse now carries `reflection_counter` (the same value
    reflect() itself returns and decrements) — a client seeing it high
    alongside thin mental_models/lessons knows consolidation is pending,
    not that there is nothing to consolidate.
    """
    from qortia.models import ContextResponse

    fields = set(ContextResponse.model_fields)
    assert fields & {"last_reflected_at", "consolidated", "reflection_counter"}, (
        "ContextResponse carries no consolidation signal "
        f"(fields={sorted(fields)}); an operator who never runs reflect gets a "
        "silently 2/3-empty bundle"
    )


# ── The published contract already exists — this one is meant to pass ──────


def test_openapi_publishes_the_memory_type_enum() -> None:
    """FastAPI already publishes the six-value enum via /openapi.json — agnova
    does not need a Qortia change to learn the vocabulary, only to read it.

    Checks MemoryItem's own JSON schema directly rather than the full
    qortia.app singleton's assembled OpenAPI document — same reasoning as
    test_an_outcome_can_be_reported_over_http above, and the schema FastAPI
    publishes is generated from exactly this model either way."""
    from qortia.models import MemoryItem

    item = MemoryItem.model_json_schema()["properties"]["type"]
    enum = item.get("enum") or item.get("allOf", [{}])[0].get("enum")
    assert enum and "lesson" in enum, f"expected the six-value enum, got {item}"
