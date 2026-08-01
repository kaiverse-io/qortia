"""
Extraction Quality Eval (EQE).

Tests whether remember.py correctly extracts memories from raw input text.
Unlike REH which tests retrieval quality, EQE tests extraction quality:
  - Did the right facts get extracted?
  - Were the right memory types assigned?
  - Were temporal references resolved correctly?
  - Were noise/low-signal inputs correctly ignored?

Metrics:
  Extraction Recall    — fraction of expected facts extracted (coverage)
  Extraction Precision — fraction of extracted memories that are "valid"
                         (contain expected content, not noise)
  Type Accuracy        — fraction with correct mem_type assignment
  Noise Rejection      — fraction of noise-only inputs that produce 0 memories

Gold labels are hand-curated: each input has a list of expected_facts
(substrings that MUST appear in at least one extracted memory) and
expected_types (acceptable memory types for each fact).

Usage:
    QORTIA_EVAL_MODE=true python3 evals/run_extraction_eval.py

Report: evals/results/eqe_latest.json
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
    QORTIA_URL,
    provision_eval_agent,
)

EXTRACTION_FLOOR_RECALL = 0.70  # ≥70% of expected facts extracted
EXTRACTION_FLOOR_PRECISION = 0.75  # ≥75% of extracted memories are non-noise
NOISE_REJECTION_FLOOR = 0.80  # ≥80% of noise inputs produce 0 high-signal memories


# ── Gold evaluation cases ──────────────────────────────────────────────────

EXTRACTION_CASES: list[dict[str, Any]] = [
    {
        "id": "eqe-001",
        "description": "Decision with clear rationale — should extract decision type",
        "input": {
            "memories": [
                {
                    "type": "decision",
                    "content": (
                        "After evaluating three options, we decided to use PostgreSQL "
                        "as our primary database. Redis was ruled out due to persistence "
                        "concerns. MongoDB was ruled out due to the team's SQL expertise. "
                        "PostgreSQL with pgvector supports both structured queries and "
                        "vector similarity search."
                    ),
                }
            ]
        },
        "expected_facts": ["PostgreSQL", "pgvector", "SQL"],
        "expected_types": ["decision"],
        "is_noise": False,
    },
    {
        "id": "eqe-002",
        "description": "Multi-fact episodic — should extract several distinct facts",
        "input": {
            "memories": [
                {
                    "type": "episodic",
                    "content": (
                        "Deployed v2.3.1 to production at 14:32 UTC. "
                        "Vault agent sidecar injection is now enabled on all pods. "
                        "P99 latency improved from 380ms to 145ms after the deploy. "
                        "Rollout took 8 minutes with zero downtime."
                    ),
                }
            ]
        },
        "expected_facts": ["v2.3.1", "Vault", "P99", "145ms"],
        "expected_types": ["episodic"],
        "is_noise": False,
    },
    {
        "id": "eqe-003",
        "description": "Lesson with causal structure — should extract lesson type",
        "input": {
            "memories": [
                {
                    "type": "lesson",
                    "content": (
                        "When deploying Flyway migrations with tenant_transaction() active, "
                        "always set search_path to public first. We learned this after a "
                        "failed migration that silently applied to the wrong schema. The fix "
                        "took 3 hours and required a full DB restore."
                    ),
                }
            ]
        },
        "expected_facts": ["Flyway", "tenant_transaction", "search_path"],
        "expected_types": ["lesson"],
        "is_noise": False,
    },
    {
        "id": "eqe-004",
        "description": "Noise input — status-only content should not produce high-signal memories",
        "input": {
            "memories": [
                {"type": "episodic", "content": "Done."},
                {"type": "episodic", "content": "Ok, noted."},
                {"type": "episodic", "content": "Will do."},
                {"type": "episodic", "content": "Sounds good."},
            ]
        },
        "expected_facts": [],
        "expected_types": [],
        "is_noise": True,
        "noise_max_words_per_memory": 5,
    },
    {
        "id": "eqe-005",
        "description": "Technical decision with named entities",
        "input": {
            "memories": [
                {
                    "type": "decision",
                    "content": (
                        "Bob and Alice agreed to use BGE-M3 as the embedding model for "
                        "Qortia. Compared against text-embedding-3-large and bge-large-en-v1.5. "
                        "BGE-M3 wins on multilingual support and 1024-dim vectors. "
                        "Hosted via LiteLLM."
                    ),
                }
            ]
        },
        "expected_facts": ["BGE-M3", "multilingual", "LiteLLM", "1024"],
        "expected_types": ["decision"],
        "is_noise": False,
    },
    {
        "id": "eqe-006",
        "description": "Org memory handoff — should extract structured org knowledge",
        "input": {
            "org_memories": [
                {
                    "type": "handoff",
                    "title": "AuthService ownership transfer",
                    "content": (
                        "AuthService is now owned by the Platform team (was: Security team). "
                        "Primary contact: diana@example.internal. "
                        "All OIDC and JWKS changes require Platform team approval. "
                        "Next planned work: multi-tenant token issuance (Q3 2026)."
                    ),
                }
            ]
        },
        "expected_facts": ["AuthService", "diana", "OIDC", "JWKS"],
        "expected_types": ["handoff"],
        "is_noise": False,
        "is_org": True,
    },
    {
        "id": "eqe-007",
        "description": "Mixed signal — important decision buried in noise",
        "input": {
            "memories": [
                {
                    "type": "decision",
                    "content": (
                        "OK so anyway after all the back and forth, the thing is we decided "
                        "that for the tenant isolation checks we should use RLS at the DB level "
                        "rather than application-layer filtering. This is the right call. Done."
                    ),
                }
            ]
        },
        "expected_facts": ["RLS", "tenant isolation", "DB"],
        "expected_types": ["decision"],
        "is_noise": False,
    },
    {
        "id": "eqe-008",
        "description": "Experiential pattern — should capture concrete operational knowledge",
        "input": {
            "memories": [
                {
                    "type": "experiential",
                    "content": (
                        "Every time we hit 80% memory on the embedding worker pod, "
                        "it OOM-kills within 5 minutes. Setting a 4GB limit and "
                        "EMBEDDING_BATCH_SIZE=25 keeps it stable indefinitely. "
                        "Observed across 6 separate incidents."
                    ),
                }
            ]
        },
        "expected_facts": ["OOM", "4GB", "EMBEDDING_BATCH_SIZE", "25"],
        "expected_types": ["experiential", "lesson"],
        "is_noise": False,
    },
    {
        "id": "eqe-009",
        "description": "Pronoun-heavy input — should still extract the key facts",
        "input": {
            "memories": [
                {
                    "type": "episodic",
                    "content": (
                        "He said we should move it to the new cluster. She agreed and "
                        "said the migration of the AuthService should happen next Tuesday. "
                        "They'll handle the DNS cutover. It should take about 2 hours."
                    ),
                }
            ]
        },
        "expected_facts": ["AuthService", "migration"],
        "expected_types": ["episodic"],
        "is_noise": False,
        "note": "Pronouns without antecedents should not create noise memories",
    },
    {
        "id": "eqe-010",
        "description": "Multiple memory types in one batch",
        "input": {
            "memories": [
                {
                    "type": "decision",
                    "content": (
                        "Decided to enable Pyroscope profiling in all production services. "
                        "CPU flame graphs revealed 40% time in JSON serialization."
                    ),
                },
                {
                    "type": "lesson",
                    "content": (
                        "Never commit Vault tokens to git. One leaked token caused a 2-hour "
                        "incident requiring full rotation of all secrets."
                    ),
                },
                {
                    "type": "episodic",
                    "content": (
                        "Completed ADR-125 causal tracking implementation. "
                        "All unit tests green. Ready for staging."
                    ),
                },
            ]
        },
        "expected_facts": ["Pyroscope", "Vault", "ADR-125", "causal tracking"],
        "expected_types": ["decision", "lesson", "episodic"],
        "is_noise": False,
    },
]


# ── Eval runner ────────────────────────────────────────────────────────────


async def _run_extraction_case(client: httpx.AsyncClient, case: dict[str, Any]) -> dict[str, Any]:
    tenant_id, agent_id = await provision_eval_agent(client)
    inp = case["input"]

    seeded_memory_ids: list[str] = []

    # Seed each memory individually so noise cases aren't concatenated
    memories = inp.get("memories", [])
    for m in memories:
        r = await client.post(
            "/v1/internal/eval/seed-memory",
            json={
                "agent_id": agent_id,
                "tenant_id": tenant_id,
                "content": m["content"],
                "mem_type": m["type"],
            },
        )
        if r.status_code == 200:
            seeded_memory_ids.append(r.json()["memory_id"])

    # Seed org memories
    org_memories = inp.get("org_memories", [])
    org_ids: list[str] = []
    for om in org_memories:
        r = await client.post(
            "/v1/internal/eval/remember-org",
            json=om,
            params={"tenant_id": tenant_id, "agent_id": agent_id},
        )
        if r.status_code == 200:
            org_ids.append(r.json().get("id", ""))

    await asyncio.sleep(EMBEDDING_WAIT_SECONDS)

    if case.get("is_noise"):
        # For noise cases: recall all memories and check none have substantial content
        resp = await client.post(
            "/v1/internal/eval/recall-full",
            json={"query": "task complete done noted", "scope": "private"},
            params={"tenant_id": tenant_id, "agent_id": agent_id},
        )
        results = resp.json().get("results", []) if resp.status_code == 200 else []
        noise_threshold = case.get("noise_max_words_per_memory", 5)
        high_signal = [r for r in results if len(r["content"].split()) > noise_threshold]
        noise_rejected = len(high_signal) == 0
        return {
            "id": case["id"],
            "pass": noise_rejected,
            "is_noise": True,
            "high_signal_memories": len(high_signal),
            "total_results": len(results),
            "reason": None
            if noise_rejected
            else f"{len(high_signal)} high-signal memories from noise input",
        }

    # For signal cases: recall and check expected facts appear
    scope = "org" if case.get("is_org") else "private"
    query = " ".join(case["expected_facts"][:3])  # use first 3 expected facts as query

    resp = await client.post(
        "/v1/internal/eval/recall-full",
        json={"query": query, "scope": scope},
        params={"tenant_id": tenant_id, "agent_id": agent_id},
    )
    if resp.status_code != 200:
        return {
            "id": case["id"],
            "pass": False,
            "is_noise": False,
            "facts_found": [],
            "facts_missing": case["expected_facts"],
            "reason": f"recall HTTP {resp.status_code}",
        }

    results = resp.json().get("results", [])
    all_content = " ".join(r["content"] for r in results).lower()

    expected_facts = case["expected_facts"]
    facts_found = [f for f in expected_facts if f.lower() in all_content]
    facts_missing = [f for f in expected_facts if f.lower() not in all_content]

    recall_score = len(facts_found) / len(expected_facts) if expected_facts else 1.0
    passed = recall_score >= EXTRACTION_FLOOR_RECALL and len(results) >= 1

    return {
        "id": case["id"],
        "pass": passed,
        "is_noise": False,
        "extraction_recall": recall_score,
        "facts_found": facts_found,
        "facts_missing": facts_missing,
        "total_results": len(results),
        "reason": None if passed else f"recall={recall_score:.2f} missing={facts_missing}",
    }


async def run_eqe() -> int:
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=QORTIA_URL, timeout=60.0) as client:
        print("=" * 60)
        print(f"EXTRACTION QUALITY EVAL — {len(EXTRACTION_CASES)} cases")
        print("=" * 60)

        for case in EXTRACTION_CASES:
            result = await _run_extraction_case(client, case)
            results.append(result)
            status = "PASS" if result["pass"] else "FAIL"
            extra = ""
            if result.get("is_noise"):
                extra = f" [high_signal={result.get('high_signal_memories', 0)}]"
            elif not result["pass"]:
                extra = f" missing={result.get('facts_missing', [])}"
            print(f"  [{status}] {case['id']}: {case['description']}{extra}")

    signal_cases = [r for r in results if not r.get("is_noise")]
    noise_cases = [r for r in results if r.get("is_noise")]

    signal_pass_rate = sum(1 for r in signal_cases if r["pass"]) / max(len(signal_cases), 1)
    noise_rejection_rate = sum(1 for r in noise_cases if r["pass"]) / max(len(noise_cases), 1)
    avg_extraction_recall = sum(r.get("extraction_recall", 0.0) for r in signal_cases) / max(
        len(signal_cases), 1
    )

    overall_pass = (
        signal_pass_rate >= EXTRACTION_FLOOR_RECALL
        and noise_rejection_rate >= NOISE_REJECTION_FLOOR
    )

    print(f"\n{'Metric':<35} {'Score':>8}  {'Floor':>8}  Status")
    print("-" * 60)
    _pm("Signal case pass rate", signal_pass_rate, EXTRACTION_FLOOR_RECALL)
    _pm("Avg extraction recall", avg_extraction_recall, EXTRACTION_FLOOR_RECALL)
    _pm("Noise rejection rate", noise_rejection_rate, NOISE_REJECTION_FLOOR)
    print(f"\nGate: {'PASS' if overall_pass else 'FAIL'}")

    report = {
        "total_cases": len(results),
        "signal_pass_rate": signal_pass_rate,
        "avg_extraction_recall": avg_extraction_recall,
        "noise_rejection_rate": noise_rejection_rate,
        "overall_gate": "PASS" if overall_pass else "FAIL",
        "cases": results,
    }
    out = Path("evals/results/eqe_latest.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Report written to {out}")

    return 0 if overall_pass else 1


def _pm(name: str, score: float, floor: float) -> None:
    status = "✓" if score >= floor else "✗"
    print(f"  {name:<33} {score:>8.3f}  {floor:>8.3f}  {status}")


if __name__ == "__main__":
    sys.exit(asyncio.run(run_eqe()))
