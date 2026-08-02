"""
Layer 1b: Temporal Eval Harness (TEH).

Extends the REH with two additional dimensions:

  recall_v2_temporal.json  — Bi-temporal conflict resolution (ADR-078):
      Seeds a SUPERSEDED fact (valid_until in past) and a CURRENT fact.
      Assert: current fact ranks above superseded fact. The expired fact
      must not appear in top-5 results.

  recall_v2_supersede.json — Selective forgetting (ADR-027):
      Seeds a CURRENT fact only, and an EXPIRED fact (valid_until past).
      Assert: expired fact does NOT appear in top-5. Current fact does.

These tests require:
  - valid_until filtering in the recall pipeline (ADR-078)
  - Eval seed-memory endpoint supporting valid_until, valid_from, importance

Usage:
    QORTIA_EVAL_MODE=true python3 evals/run_temporal_eval.py

Runs both datasets. Exits 0 if all gates pass, 1 otherwise.
Report: evals/results/teh_latest.json
"""

from __future__ import annotations

import asyncio
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from evals.dataset_loader import (
    EMBEDDING_WAIT_SECONDS,
    QORTIA_URL,
    load_dataset,
    provision_eval_agent,
)

DATASETS = [
    Path("evals/datasets/recall_v2_temporal.json"),
    Path("evals/datasets/recall_v2_supersede.json"),
]

# Regression floors
PASS_RATE_FLOOR = 0.50  # ≥50% of cases must pass (some semantic cases need warm embeddings)
EXPIRED_EXCLUSION_FLOOR = 1.0  # expired memories must NEVER appear (100%) — hard gate


# ── Temporal-aware seeding ─────────────────────────────────────────────────


def _resolve_relative_date(value: str | None) -> str | None:
    """Convert '-7 days' → ISO-8601 UTC timestamp string. Pass-through for None."""
    if value is None:
        return None
    value = value.strip()
    if value.startswith("-") and "day" in value:
        days = int(value.split()[0])
        dt = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=days)
        return dt.isoformat()
    if value.startswith("+") and "day" in value:
        days = int(value.split()[0])
        dt = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=days)
        return dt.isoformat()
    return value  # already an ISO string


async def _seed_temporal_memory(
    client: httpx.AsyncClient,
    tenant_id: str,
    agent_id: str,
    mem: dict[str, Any],
) -> str:
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "tenant_id": tenant_id,
        "content": mem["content"],
        "mem_type": mem.get("type", "episodic"),
        "scope": mem.get("scope", "private"),
        "lang": mem.get("lang", "en"),
        "importance": mem.get("importance", 0.5),
        "is_consolidated": mem.get("is_consolidated", False),
    }
    if "valid_from" in mem:
        payload["valid_from"] = _resolve_relative_date(mem["valid_from"])
    if "valid_until" in mem:
        payload["valid_until"] = _resolve_relative_date(mem["valid_until"])

    resp = await client.post("/v1/internal/eval/seed-memory", json=payload)
    resp.raise_for_status()
    return resp.json()["memory_id"]


async def _seed_temporal_org_memory(
    client: httpx.AsyncClient,
    tenant_id: str,
    agent_id: str,
    mem: dict[str, Any],
) -> str:
    payload: dict[str, Any] = {
        "type": mem.get("type", "handoff"),
        "title": mem.get("title", "Untitled"),
        "content": mem["content"],
        "lang": mem.get("lang", "en"),
    }
    if "valid_until" in mem:
        payload["valid_until"] = _resolve_relative_date(mem["valid_until"])
    if "valid_from" in mem:
        payload["valid_from"] = _resolve_relative_date(mem["valid_from"])
    resp = await client.post(
        "/v1/internal/eval/remember-org",
        json=payload,
        params={"tenant_id": tenant_id, "agent_id": agent_id},
    )
    if resp.status_code != 200:
        return ""
    return resp.json().get("id", "")


# ── Per-case runner ────────────────────────────────────────────────────────


async def _seed_temporal_case(
    client: httpx.AsyncClient, tenant_id: str, agent_id: str, setup: dict[str, Any]
) -> tuple[dict[int, str], dict[int, str], set[str]]:
    """Seed hard negatives, org memories, then private memories. Returns
    (id_map, org_id_map, expired_ids) — expired_ids covers rows seeded with
    a valid_until already in the past."""
    id_map: dict[int, str] = {}
    org_id_map: dict[int, str] = {}
    expired_ids: set[str] = set()

    # Seed hard negatives first (older created_at)
    for neg in setup.get("hard_negatives", []):
        await _seed_temporal_memory(client, tenant_id, agent_id, neg)
        await asyncio.sleep(0.1)

    for i, om in enumerate(setup.get("org_memories", [])):
        mid = await _seed_temporal_org_memory(client, tenant_id, agent_id, om)
        if mid:
            org_id_map[i] = mid
            if om.get("valid_until"):
                expired_ids.add(mid)

    for i, mem in enumerate(setup.get("memories", [])):
        mid = await _seed_temporal_memory(client, tenant_id, agent_id, mem)
        id_map[i] = mid
        if mem.get("valid_until"):
            expired_ids.add(mid)
        await asyncio.sleep(0.1)

    return id_map, org_id_map, expired_ids


def _resolve_temporal_ground_truth(
    case: dict[str, Any], id_map: dict[int, str], org_id_map: dict[int, str]
) -> str | None:
    idx = case.get("ground_truth_index", 0)
    if case.get("ground_truth_source", "memories") == "org_memories":
        return org_id_map.get(idx)
    return id_map.get(idx)


def _resolve_additional_expired_ids(
    case: dict[str, Any], id_map: dict[int, str], org_id_map: dict[int, str], expired_ids: set[str]
) -> None:
    for key in ("negative_must_not_appear_index", "negative_must_not_appear_indices"):
        val = case.get(key)
        if val is None:
            continue
        indices = [val] if isinstance(val, int) else val
        for idx in indices:
            mid = id_map.get(idx) or org_id_map.get(idx)
            if mid:
                expired_ids.add(mid)


async def _execute_temporal_recall(
    client: httpx.AsyncClient, case: dict[str, Any], tenant_id: str, agent_id: str
) -> list[dict[str, Any]] | dict[str, Any]:
    """Retry once if 0 results — handles embedding race on cold-start.
    Returns the results list, or a _failed(...) dict on a non-200 response."""
    results: list[dict[str, Any]] = []
    for attempt in range(2):
        resp = await client.post(
            "/v1/internal/eval/recall-full",
            json=case["query"],
            params={"tenant_id": tenant_id, "agent_id": agent_id},
        )
        if resp.status_code != 200:
            return _failed(case["id"], f"recall HTTP {resp.status_code}: {resp.text[:200]}")
        results = resp.json().get("results", [])
        if results or attempt == 1:
            break
        await asyncio.sleep(EMBEDDING_WAIT_SECONDS)  # embedding not ready, retry after one cycle
    return results


def _temporal_case_passed(
    gt_in_top5: bool, expired_leaked: bool, case: dict[str, Any], results: list[dict[str, Any]]
) -> bool:
    passed = gt_in_top5 and not expired_leaked
    expected = case.get("expected", {})
    if passed and results and expected.get("must_contain_in_top_result"):
        top_content = results[0]["content"].lower()
        phrases = expected["must_contain_in_top_result"]
        if not any(p.lower() in top_content for p in phrases):
            passed = False
    return passed


async def _run_case(client: httpx.AsyncClient, case: dict[str, Any]) -> dict[str, Any]:
    tenant_id, agent_id = await provision_eval_agent(client)

    id_map, org_id_map, expired_ids = await _seed_temporal_case(
        client, tenant_id, agent_id, case["setup"]
    )

    # Wait 2 full embedding cycles — handles cold-start and batch backlog
    await asyncio.sleep(EMBEDDING_WAIT_SECONDS * 2)

    ground_truth_id = _resolve_temporal_ground_truth(case, id_map, org_id_map)
    _resolve_additional_expired_ids(case, id_map, org_id_map, expired_ids)

    results = await _execute_temporal_recall(client, case, tenant_id, agent_id)
    if isinstance(results, dict):  # _failed(...) sentinel
        return results

    result_ids = [r["id"] for r in results]
    top5_ids = set(result_ids[:5])

    gt_in_top5 = ground_truth_id is not None and ground_truth_id in top5_ids
    expired_leaked = bool(expired_ids & top5_ids)  # any expired ID in top-5
    passed = _temporal_case_passed(gt_in_top5, expired_leaked, case, results)

    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "pass": passed,
        "gt_in_top5": gt_in_top5,
        "expired_leaked": expired_leaked,
        "expired_ids": list(expired_ids),
        "leaked_ids": list(expired_ids & top5_ids),
        "reason": (None if passed else f"gt_in_top5={gt_in_top5} expired_leaked={expired_leaked}"),
        "top3": [{"id": r["id"], "content": r["content"][:100]} for r in results[:3]],
    }


def _failed(case_id: str, reason: str) -> dict[str, Any]:
    return {
        "id": case_id,
        "pass": False,
        "gt_in_top5": False,
        "expired_leaked": False,
        "expired_ids": [],
        "leaked_ids": [],
        "reason": reason,
        "top3": [],
    }


# ── Main ───────────────────────────────────────────────────────────────────


async def run_teh() -> int:
    all_results: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=QORTIA_URL, timeout=60.0) as client:
        for dataset_path in DATASETS:
            if not dataset_path.exists():
                print(f"SKIP: {dataset_path} not found")
                continue
            cases = load_dataset(dataset_path)
            dataset_results: list[dict[str, Any]] = []

            print(f"\n{'=' * 60}")
            print(f"Dataset: {dataset_path.name} ({len(cases)} cases)")
            print("=" * 60)

            for case in cases:
                result = await _run_case(client, case)
                dataset_results.append(result)
                all_results.append(result)
                status = "PASS" if result["pass"] else "FAIL"
                leak = " [EXPIRED LEAKED]" if result["expired_leaked"] else ""
                print(f"  [{status}]{leak} {case['id']}: {case.get('description', '')}")
                if not result["pass"]:
                    print(f"         reason: {result['reason']}")

            pass_rate = sum(1 for r in dataset_results if r["pass"]) / len(dataset_results)
            leak_rate = sum(1 for r in dataset_results if r["expired_leaked"]) / len(
                dataset_results
            )
            dataset_summaries.append(
                {
                    "dataset": dataset_path.name,
                    "cases": len(dataset_results),
                    "pass_rate": pass_rate,
                    "expired_leak_rate": leak_rate,
                    "pass_gate": pass_rate >= PASS_RATE_FLOOR,
                    "leak_gate": leak_rate == 0.0,
                }
            )

    # Print summary
    print(f"\n{'=' * 60}")
    print("TEMPORAL EVAL HARNESS — SUMMARY")
    print("=" * 60)
    print(f"\n{'Dataset':<35} {'Pass%':>6} {'LeakRate':>9} {'PassGate':>9} {'LeakGate':>9}")
    print("-" * 72)
    for s in dataset_summaries:
        print(
            f"  {s['dataset']:<33} {s['pass_rate']:>5.1%} "
            f"{s['expired_leak_rate']:>9.1%} "
            f"{'✓' if s['pass_gate'] else '✗':>9} "
            f"{'✓' if s['leak_gate'] else '✗':>9}"
        )

    total_pass = sum(1 for r in all_results if r["pass"]) / max(len(all_results), 1)
    total_leak = sum(1 for r in all_results if r["expired_leaked"]) / max(len(all_results), 1)
    overall_pass = total_pass >= PASS_RATE_FLOOR and total_leak == 0.0

    print(f"\nOverall pass rate: {total_pass:.1%}  (floor: {PASS_RATE_FLOOR:.0%})")
    print(f"Expired leak rate: {total_leak:.1%}  (floor: 0%)")
    print(f"Gate: {'PASS' if overall_pass else 'FAIL'}")

    report = {
        "total_cases": len(all_results),
        "pass_rate": total_pass,
        "expired_leak_rate": total_leak,
        "overall_gate": "PASS" if overall_pass else "FAIL",
        "datasets": dataset_summaries,
        "cases": all_results,
    }
    out = Path("evals/results/teh_latest.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_teh()))
