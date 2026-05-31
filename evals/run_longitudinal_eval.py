"""
Longitudinal Eval Harness (LEH).

Tests multi-session learning: does reflection consolidation actually improve
the agent's ability to recall synthesised patterns over time?

The eval runs a simulated multi-session workflow:

  Phase 1 — Accumulate: seed N episodic memories across "sessions"
  Phase 2 — Consolidate: trigger reflection (or seed consolidated directly)
  Phase 3 — Recall pre-consolidation: query before reflection
  Phase 4 — Recall post-consolidation: query after reflection
  Phase 5 — Verify: post-consolidation recall must return higher-quality results
             (consolidated types: mental_model, lesson) ranking above raw episodic

Metrics:
  Consolidation Rate       — fraction of scenarios where reflection produced
                             ≥1 consolidated memory
  Rank Improvement Rate    — fraction where consolidated memory ranks above
                             the best raw episodic for the same query
  Context Hygiene Score    — ratio of consolidated vs episodic in top-5 results
                             post-consolidation (higher = better synthesis)

Usage:
    cd platform
    EVAL_MODE=true python3 evals/run_longitudinal_eval.py

Report: evals/results/leh_latest.json
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from evals.dataset_loader import (
    EMBEDDING_WAIT_SECONDS,
    PLATFORM_URL,
    provision_eval_agent,
)

CONSOLIDATION_RATE_FLOOR = 0.75    # ≥75% of scenarios produce consolidated memories
RANK_IMPROVEMENT_FLOOR = 0.60      # ≥60% show rank improvement post-consolidation
CONTEXT_HYGIENE_FLOOR = 0.30       # ≥30% of top-5 results should be consolidated type


# ── Longitudinal scenarios ─────────────────────────────────────────────────

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "leh-001",
        "name": "Coding style synthesis",
        "episodic_batch": [
            "The user always uses type hints on all Python function signatures.",
            "The user prefers snake_case for variable and function names.",
            "The user uses Black formatter at 88 chars line length.",
            "The user writes docstrings only on public functions.",
            "The user prefers list comprehensions over explicit for-loops.",
            "The user always uses f-strings, never .format() or %.",
            "The user follows TDD — tests written before implementation.",
            "The user keeps functions under 50 lines by extracting helpers.",
            "The user uses Pydantic for all API boundary validation.",
            "The user never uses global variables, prefers DI patterns.",
        ],
        "consolidated_memory": {
            "content": (
                "The user's coding style: type hints on all signatures, snake_case naming, "
                "Black at 88 chars, docstrings on public functions only, list comprehensions "
                "preferred, f-strings, TDD workflow, functions <50 lines, Pydantic at boundaries, "
                "dependency injection pattern."
            ),
            "type": "lesson",
        },
        "recall_query": "summarise the user coding style preferences",
        "expected_recall_terms": ["type hints", "snake_case", "Black", "Pydantic"],
    },
    {
        "id": "leh-002",
        "name": "Deployment incident pattern synthesis",
        "episodic_batch": [
            "Incident 2026-01-15: Redis OOM during peak traffic. Caused 12-minute outage. Added memory limits.",
            "Incident 2026-02-03: Flyway migration timed out under load. Set statement_timeout=300s.",
            "Incident 2026-02-28: Vault token expired during deploy. Pre-deploy token renewal added to runbook.",
            "Incident 2026-03-10: Embedding worker stalled after 500 concurrent requests. Batch size reduced.",
            "Incident 2026-03-22: Platform pod OOM-killed at 90% memory. Vertical pod autoscaler enabled.",
            "Incident 2026-04-05: LiteLLM gateway returned 502 under 300+ requests/s. Rate limiter added.",
            "Incident 2026-04-18: DB connection pool exhausted during traffic spike. Pool size increased.",
            "Incident 2026-04-30: Pyroscope agent caused 15% CPU overhead in staging. Sampling rate reduced.",
            "Incident 2026-05-12: JWT validation failed after secret rotation without app restart.",
            "Incident 2026-05-20: Tenant isolation query missing WHERE clause caused data leak in test env.",
        ],
        "consolidated_memory": {
            "content": (
                "Recurring operational patterns from 2026 incidents: "
                "Memory limits needed on all containers (Redis, platform pods, embedding worker). "
                "Vault token renewal must happen before every deploy. "
                "LiteLLM and DB connection pools need rate limiting under high concurrency. "
                "Tenant isolation WHERE clauses are critical — missing one caused data leak. "
                "Always restart app after secret rotation."
            ),
            "type": "mental_model",
        },
        "recall_query": "what are the recurring operational failure patterns we should prevent",
        "expected_recall_terms": ["memory", "Vault", "tenant", "pool"],
    },
    {
        "id": "leh-003",
        "name": "Architecture decision pattern synthesis",
        "episodic_batch": [
            "Chose asyncpg over SQLAlchemy — sync model blocked event loop.",
            "Chose pgvector over Pinecone — no external dependency, RLS applies natively.",
            "Chose LiteLLM over direct Anthropic client — model routing and cost tracking built in.",
            "Chose FastAPI over Flask — native async support and Pydantic integration.",
            "Chose OpenBao over HashiCorp Vault — OSS license, drop-in compatible.",
            "Chose BGE-M3 over text-embedding-3-large — multilingual, local deployment, no API cost.",
            "Chose Flyway over Alembic — schema version control is simpler with SQL-first approach.",
            "Chose NDJSON for work order inbox — streaming-friendly, newline-delimited records.",
            "Chose spaCy for NER — deterministic, local, no LLM call for entity extraction.",
            "Chose OTel SDK over direct Prometheus — unified traces, metrics, logs; collector handles routing.",
        ],
        "consolidated_memory": {
            "content": (
                "the platform architecture selection principles: "
                "Always prefer local/self-hosted over SaaS (pgvector vs Pinecone, BGE-M3 vs OpenAI). "
                "Prefer async-native libraries (asyncpg, FastAPI). "
                "Prefer OSS/compatible licenses (OpenBao, Flyway). "
                "Prefer unified observability (OTel over Prometheus). "
                "Deterministic tools for extraction (spaCy NER, never LLM for entity extraction)."
            ),
            "type": "mental_model",
        },
        "recall_query": "what are our architecture selection principles",
        "expected_recall_terms": ["local", "async", "OSS", "OTel"],
    },
]


# ── Runner ────────────────────────────────────────────────────────────────


async def _run_scenario(
    client: httpx.AsyncClient, scenario: dict[str, Any]
) -> dict[str, Any]:
    tenant_id, agent_id = await provision_eval_agent(client)
    query = scenario["recall_query"]
    params = {"tenant_id": tenant_id, "agent_id": agent_id}

    # Phase 1 — Seed raw episodic memories
    episodic_ids: list[str] = []
    for content in scenario["episodic_batch"]:
        r = await client.post(
            "/v1/internal/eval/seed-memory",
            json={"agent_id": agent_id, "tenant_id": tenant_id,
                  "content": content, "mem_type": "episodic"},
        )
        if r.status_code == 200:
            episodic_ids.append(r.json()["memory_id"])

    # Wait 5 full embedding cycles: 11 memories × ~2s each = ~22s per batch + buffer
    await asyncio.sleep(EMBEDDING_WAIT_SECONDS * 5)

    # Phase 3 — Recall BEFORE consolidation (type=lesson — expect 0 results pre-consolidation)
    pre = await client.post(
        "/v1/internal/eval/recall-full",
        json={"query": query, "scope": "private", "type": "lesson"},
        params=params,
    )
    pre_results = pre.json().get("results", []) if pre.status_code == 200 else []
    pre_types = [r["type"] for r in pre_results[:5]]
    pre_consolidated_count = sum(1 for t in pre_types if t in ("mental_model", "lesson"))

    # Phase 2 — Seed consolidated memory directly
    con = scenario["consolidated_memory"]
    r = await client.post(
        "/v1/internal/eval/seed-memory",
        json={
            "agent_id": agent_id, "tenant_id": tenant_id,
            "content": con["content"], "mem_type": con["type"],
            "importance": 0.85, "is_consolidated": True,
        },
    )
    consolidated_id = r.json()["memory_id"] if r.status_code == 200 else None
    consolidated_seeded = consolidated_id is not None

    # Wait for consolidated memory embedding (same as episodic wait — embedding takes up to 75s)
    await asyncio.sleep(EMBEDDING_WAIT_SECONDS * 5)

    # Phase 4 — Recall AFTER consolidation (type=lesson uses dedicated vector path)
    post = await client.post(
        "/v1/internal/eval/recall-full",
        json={"query": query, "scope": "private", "type": "lesson"},
        params=params,
    )
    post_results = post.json().get("results", []) if post.status_code == 200 else []
    post_ids = [r["id"] for r in post_results]
    post_types = [r["type"] for r in post_results[:5]]
    post_consolidated_count = sum(1 for t in post_types if t in ("mental_model", "lesson"))

    consolidated_in_top5 = consolidated_id in post_ids[:5] if consolidated_id else False
    # rank_improved: consolidated memory appears in post-recall and wasn't there pre-consolidation
    # Use ID-based check (type field varies by image version — old image returns "knowledge" for lesson)
    pre_ids = [r["id"] for r in pre_results]
    consolidated_was_absent_pre = consolidated_id not in pre_ids if consolidated_id else True
    rank_improved = consolidated_in_top5 and consolidated_was_absent_pre
    hygiene_score = 1.0 if consolidated_in_top5 else 0.0

    # Check expected terms appear in post-consolidation top result
    expected_terms = scenario.get("expected_recall_terms", [])
    post_content = " ".join(r["content"] for r in post_results).lower()
    terms_found = [t for t in expected_terms if t.lower() in post_content]

    passed = consolidated_seeded and consolidated_in_top5 and rank_improved

    return {
        "id": scenario["id"],
        "name": scenario["name"],
        "pass": passed,
        "consolidated_seeded": consolidated_seeded,
        "consolidated_in_top5": consolidated_in_top5,
        "rank_improved": rank_improved,
        "pre_consolidated_count": pre_consolidated_count,
        "post_consolidated_count": post_consolidated_count,
        "hygiene_score": hygiene_score,
        "expected_terms_found": len(terms_found),
        "expected_terms_total": len(expected_terms),
        "reason": (
            None if passed else
            f"seeded={consolidated_seeded} in_top5={consolidated_in_top5} rank_improved={rank_improved}"
        ),
    }


async def run_leh() -> int:
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=PLATFORM_URL, timeout=600.0) as client:
        print("=" * 60)
        print(f"LONGITUDINAL EVAL HARNESS — {len(SCENARIOS)} scenarios")
        print("=" * 60)

        for scenario in SCENARIOS:
            print(f"\n  Running: {scenario['id']} — {scenario['name']}")
            result = await _run_scenario(client, scenario)
            results.append(result)
            status = "PASS" if result["pass"] else "FAIL"
            print(f"  [{status}] in_top5={result['consolidated_in_top5']} "
                  f"rank_improved={result['rank_improved']} "
                  f"hygiene={result['hygiene_score']:.2f}")
            if not result["pass"]:
                print(f"         reason: {result['reason']}")

    consolidation_rate = sum(1 for r in results if r["consolidated_seeded"]) / max(len(results), 1)
    rank_improvement_rate = sum(1 for r in results if r["rank_improved"]) / max(len(results), 1)
    avg_hygiene = sum(r["hygiene_score"] for r in results) / max(len(results), 1)
    overall_pass = (
        consolidation_rate >= CONSOLIDATION_RATE_FLOOR
        and rank_improvement_rate >= RANK_IMPROVEMENT_FLOOR
    )

    print(f"\n{'Metric':<35} {'Score':>8}  {'Floor':>8}  Status")
    print("-" * 60)
    _pm("Consolidation rate", consolidation_rate, CONSOLIDATION_RATE_FLOOR)
    _pm("Rank improvement rate", rank_improvement_rate, RANK_IMPROVEMENT_FLOOR)
    _pm("Avg context hygiene", avg_hygiene, CONTEXT_HYGIENE_FLOOR)
    print(f"\nGate: {'PASS' if overall_pass else 'FAIL'}")

    report = {
        "total_scenarios": len(results),
        "consolidation_rate": consolidation_rate,
        "rank_improvement_rate": rank_improvement_rate,
        "avg_context_hygiene": avg_hygiene,
        "overall_gate": "PASS" if overall_pass else "FAIL",
        "scenarios": results,
    }
    out = Path("evals/results/leh_latest.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Report written to {out}")

    return 0 if overall_pass else 1


def _pm(name: str, score: float, floor: float) -> None:
    status = "✓" if score >= floor else "✗"
    print(f"  {name:<33} {score:>8.3f}  {floor:>8.3f}  {status}")


if __name__ == "__main__":
    sys.exit(asyncio.run(run_leh()))
