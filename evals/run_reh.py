"""
Layer 1: Retrieval Evaluation Harness (REH).

Measures recall.py pipeline in isolation. Outputs Recall@5, Recall@10, MRR, and
Semantic Drift gap. Exits 0 if regression floors are met, 1 otherwise.

Usage:
    cd platform
    EVAL_MODE=true python3 evals/run_reh.py [--dataset evals/datasets/recall_v1.json]

North star targets (docs/platform/05-memory-benchmarking.md §2.2):
    Recall@5            > 0.85
    Recall@10           > 0.95
    MRR                 > 0.75
    Semantic Drift gap  > 0.15

Regression floors (set 5% below measured baseline — update after first run):
    Recall@5  >= 0.80
    MRR       >= 0.65
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
    _agent_headers,
    load_dataset,
    provision_eval_agent,
    seed_case,
    seed_knowledge,
    seed_org_memories,
)

# North star targets
RECALL_AT_5_TARGET = 0.85
RECALL_AT_10_TARGET = 0.95
MRR_TARGET = 0.75
SEMANTIC_DRIFT_TARGET = 0.15

# Regression floors — set 5% below full 55-case baseline (ADR-073)
RECALL_AT_5_FLOOR = 0.95
MRR_FLOOR = 0.86


async def run_reh(dataset_path: Path) -> int:
    cases = load_dataset(dataset_path)
    case_results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=PLATFORM_URL, timeout=60.0) as client:
        for case in cases:
            result = await _run_reh_case(client, case)
            case_results.append(result)
            status = "PASS" if result["pass"] else "FAIL"
            print(f"  [{status}] {case['id']}: {case.get('description', '')}")
            if not result["pass"]:
                print(f"         reason: {result['reason']}")

    if not case_results:
        print("No cases found.")
        return 1

    recall_at_5 = sum(1 for r in case_results if r["recall_at_5"]) / len(case_results)
    recall_at_10 = sum(1 for r in case_results if r["recall_at_10"]) / len(case_results)
    mrr = sum(r["mrr"] for r in case_results) / len(case_results)
    drift_gaps = [
        r["semantic_drift_gap"]
        for r in case_results
        if r["semantic_drift_gap"] is not None
    ]
    avg_drift = sum(drift_gaps) / len(drift_gaps) if drift_gaps else 0.0

    print(f"\n{'Metric':<25} {'Score':>8}  {'Target':>8}  Status")
    print("-" * 55)
    _print_metric("Recall@5", recall_at_5, RECALL_AT_5_TARGET)
    _print_metric("Recall@10", recall_at_10, RECALL_AT_10_TARGET)
    _print_metric("MRR", mrr, MRR_TARGET)
    _print_metric("Semantic Drift gap", avg_drift, SEMANTIC_DRIFT_TARGET)

    report = {
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "mrr": mrr,
        "semantic_drift_avg": avg_drift,
        "cases": case_results,
    }
    out = Path("evals/results/reh_latest.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out}")

    passed = recall_at_5 >= RECALL_AT_5_FLOOR and mrr >= MRR_FLOOR
    print(f"Regression gate: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def _print_metric(name: str, score: float, target: float) -> None:
    status = "✓" if score >= target else "✗"
    print(f"  {name:<23} {score:>8.3f}  {target:>8.3f}  {status}")


async def _run_reh_case(
    client: httpx.AsyncClient,
    case: dict[str, Any],
) -> dict[str, Any]:
    tenant_id, agent_id = await provision_eval_agent(client)
    id_map = await seed_case(client, case, tenant_id, agent_id)
    org_id_map = await seed_org_memories(client, case, tenant_id, agent_id)
    await seed_knowledge(client, case, tenant_id, agent_id)
    await asyncio.sleep(EMBEDDING_WAIT_SECONDS)

    ground_truth_idx = case.get("ground_truth_index", 0)
    ground_truth_source = case.get("ground_truth_source", "memories")

    # Resolve ground truth ID based on source
    ground_truth_id: str | None = None
    ground_truth_content: str | None = None
    if ground_truth_source == "org_memories":
        ground_truth_id = org_id_map.get(ground_truth_idx)
    elif ground_truth_source == "knowledge":
        know_items = case["setup"].get("knowledge", [])
        if ground_truth_idx < len(know_items):
            ground_truth_content = know_items[ground_truth_idx].get("content", "")
    else:
        ground_truth_id = id_map.get(ground_truth_idx)

    query_body = case["query"]
    resp = await client.post(
        "/v1/internal/eval/recall-full",
        json=query_body,
        params={"tenant_id": tenant_id, "agent_id": agent_id},
    )
    if resp.status_code != 200:
        return _failed(case["id"], f"recall HTTP {resp.status_code}: {resp.text[:200]}")

    results = resp.json().get("results", [])
    result_ids = [r["id"] for r in results]
    result_contents = [r["content"] for r in results]

    # For knowledge cases: find ground truth by source_path + chunk_index=0
    if ground_truth_source == "knowledge" and ground_truth_content:
        from app.qortia.knowledge import split_into_sections

        sections = split_into_sections(ground_truth_content)
        fingerprint = (
            sections[0]["text"][:40].lower()
            if sections
            else ground_truth_content[:40].lower()
        )
        for i, content in enumerate(result_contents):
            if fingerprint in content.lower():
                ground_truth_id = result_ids[i]
                break

    recall_at_5 = bool(ground_truth_id and ground_truth_id in result_ids[:5])
    recall_at_10 = bool(ground_truth_id and ground_truth_id in result_ids[:10])

    mrr = 0.0
    if ground_truth_id and ground_truth_id in result_ids:
        mrr = 1.0 / (result_ids.index(ground_truth_id) + 1)

    # Semantic Drift: rank gap between ground truth and best hard negative
    setup = case["setup"]
    hard_negative_count = len(setup.get("hard_negatives", []))
    gt_ids = set(id_map.values())
    hard_negative_result_ids = [rid for rid in result_ids if rid not in gt_ids]
    if ground_truth_id:
        hard_negative_result_ids = [
            rid for rid in hard_negative_result_ids if rid != ground_truth_id
        ]

    semantic_drift_gap = None
    if ground_truth_id and hard_negative_count > 0 and ground_truth_id in result_ids:
        gt_rank = result_ids.index(ground_truth_id)
        hn_ranks = [
            result_ids.index(hid)
            for hid in hard_negative_result_ids
            if hid in result_ids
        ]
        if hn_ranks:
            best_hn_rank = min(hn_ranks)
            semantic_drift_gap = (best_hn_rank - gt_rank) / max(len(result_ids), 1)

    passed = recall_at_5

    # Check must_contain_in_top_result
    if passed and case.get("expected", {}).get("must_contain_in_top_result"):
        top_content = results[0]["content"] if results else ""
        for phrase in case["expected"]["must_contain_in_top_result"]:
            if phrase.lower() not in top_content.lower():
                passed = False
                break

    # For org/knowledge cases: also check min_results
    if passed and case.get("expected", {}).get("min_results"):
        if len(results) < case["expected"]["min_results"]:
            passed = False

    return {
        "id": case["id"],
        "pass": passed,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "mrr": mrr,
        "semantic_drift_gap": semantic_drift_gap,
        "reason": None if passed else "ground truth not in top 5",
    }


def _failed(case_id: str, reason: str) -> dict[str, Any]:
    return {
        "id": case_id,
        "pass": False,
        "recall_at_5": False,
        "recall_at_10": False,
        "mrr": 0.0,
        "semantic_drift_gap": None,
        "reason": reason,
    }


if __name__ == "__main__":
    dataset = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("evals/datasets/recall_v1.json")
    )
    sys.exit(asyncio.run(run_reh(dataset)))
