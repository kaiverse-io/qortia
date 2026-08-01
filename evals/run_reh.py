"""
Layer 1: Retrieval Evaluation Harness (REH).

Measures recall.py pipeline in isolation. Outputs Recall@5, Recall@10, MRR, and
Semantic Drift gap. Exits 0 if regression floors are met, 1 otherwise.

Usage:
    QORTIA_EVAL_MODE=true python3 evals/run_reh.py [--dataset evals/datasets/recall_v1.json]

North star targets (docs/02-benchmarking.md §2.2):
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
    QORTIA_URL,
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

    async with httpx.AsyncClient(base_url=QORTIA_URL, timeout=60.0) as client:
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
        r["semantic_drift_gap"] for r in case_results if r["semantic_drift_gap"] is not None
    ]
    avg_drift = sum(drift_gaps) / len(drift_gaps) if drift_gaps else 0.0

    # Token efficiency: avg tokens retrieved per query (Mem0 target: <7k)
    token_counts = [r.get("tokens_retrieved", 0) for r in case_results if r.get("tokens_retrieved")]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0
    token_target = 7000.0

    print(f"\n{'Metric':<25} {'Score':>8}  {'Target':>8}  Status")
    print("-" * 55)
    _print_metric("Recall@5", recall_at_5, RECALL_AT_5_TARGET)
    _print_metric("Recall@10", recall_at_10, RECALL_AT_10_TARGET)
    _print_metric("MRR", mrr, MRR_TARGET)
    _print_metric("Semantic Drift gap", avg_drift, SEMANTIC_DRIFT_TARGET)
    if avg_tokens > 0:
        print(
            f"  {'Avg tokens retrieved':<23} {avg_tokens:>8.0f}  {token_target:>8.0f}  "
            f"{'✓' if avg_tokens <= token_target else '✗'}"
        )

    report = {
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "mrr": mrr,
        "semantic_drift_avg": avg_drift,
        "avg_tokens_retrieved": avg_tokens,
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


def _resolve_ground_truth(
    case: dict[str, Any], id_map: dict[int, str], org_id_map: dict[int, str]
) -> tuple[str | None, str | None]:
    """Returns (ground_truth_id, ground_truth_content) — content is only set
    for knowledge-sourced cases, resolved to an ID later via fingerprint match."""
    idx = case.get("ground_truth_index", 0)
    source = case.get("ground_truth_source", "memories")
    if source == "org_memories":
        return org_id_map.get(idx), None
    if source == "knowledge":
        know_items = case["setup"].get("knowledge", [])
        content = know_items[idx].get("content", "") if idx < len(know_items) else None
        return None, content
    return id_map.get(idx), None


def _resolve_knowledge_ground_truth_id(
    ground_truth_content: str, result_ids: list[str], result_contents: list[str]
) -> str | None:
    """Find ground truth by source_path + chunk_index=0 fingerprint match."""
    from qortia.knowledge import split_into_sections

    sections = split_into_sections(ground_truth_content)
    fingerprint = (
        sections[0]["text"][:40].lower() if sections else ground_truth_content[:40].lower()
    )
    for i, content in enumerate(result_contents):
        if fingerprint in content.lower():
            return result_ids[i]
    return None


def _compute_semantic_drift_gap(
    ground_truth_id: str | None,
    result_ids: list[str],
    id_map: dict[int, str],
    hard_negative_count: int,
) -> float | None:
    """Rank gap between ground truth and best hard negative."""
    if not ground_truth_id or hard_negative_count == 0 or ground_truth_id not in result_ids:
        return None
    gt_ids = set(id_map.values())
    hard_negative_result_ids = [
        rid for rid in result_ids if rid not in gt_ids and rid != ground_truth_id
    ]
    hn_ranks = [result_ids.index(hid) for hid in hard_negative_result_ids if hid in result_ids]
    if not hn_ranks:
        return None
    gt_rank = result_ids.index(ground_truth_id)
    best_hn_rank = min(hn_ranks)
    return (best_hn_rank - gt_rank) / max(len(result_ids), 1)


def _case_meets_expectations(passed: bool, case: dict[str, Any], results: list[dict]) -> bool:  # type: ignore[type-arg]
    expected = case.get("expected", {})
    if passed and expected.get("must_contain_in_top_result"):
        top_content = results[0]["content"] if results else ""
        for phrase in expected["must_contain_in_top_result"]:
            if phrase.lower() not in top_content.lower():
                return False
    if passed and expected.get("min_results") and len(results) < expected["min_results"]:
        return False
    return passed


async def _run_reh_case(
    client: httpx.AsyncClient,
    case: dict[str, Any],
) -> dict[str, Any]:
    tenant_id, agent_id = await provision_eval_agent(client)
    id_map = await seed_case(client, case, tenant_id, agent_id)
    org_id_map = await seed_org_memories(client, case, tenant_id, agent_id)
    await seed_knowledge(client, case, tenant_id, agent_id)
    await asyncio.sleep(EMBEDDING_WAIT_SECONDS)

    ground_truth_id, ground_truth_content = _resolve_ground_truth(case, id_map, org_id_map)

    resp = await client.post(
        "/v1/internal/eval/recall-full",
        json=case["query"],
        params={"tenant_id": tenant_id, "agent_id": agent_id},
    )
    if resp.status_code != 200:
        return _failed(case["id"], f"recall HTTP {resp.status_code}: {resp.text[:200]}")

    results = resp.json().get("results", [])
    result_ids = [r["id"] for r in results]
    result_contents = [r["content"] for r in results]

    if case.get("ground_truth_source") == "knowledge" and ground_truth_content:
        ground_truth_id = _resolve_knowledge_ground_truth_id(
            ground_truth_content, result_ids, result_contents
        )

    recall_at_5 = bool(ground_truth_id and ground_truth_id in result_ids[:5])
    recall_at_10 = bool(ground_truth_id and ground_truth_id in result_ids[:10])

    mrr = 0.0
    if ground_truth_id and ground_truth_id in result_ids:
        mrr = 1.0 / (result_ids.index(ground_truth_id) + 1)

    hard_negative_count = len(case["setup"].get("hard_negatives", []))
    semantic_drift_gap = _compute_semantic_drift_gap(
        ground_truth_id, result_ids, id_map, hard_negative_count
    )

    passed = _case_meets_expectations(recall_at_5, case, results)

    # Token efficiency: count words in all returned results (proxy for token count)
    tokens_retrieved = sum(len(r["content"].split()) for r in results)

    return {
        "id": case["id"],
        "pass": passed,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "mrr": mrr,
        "semantic_drift_gap": semantic_drift_gap,
        "tokens_retrieved": tokens_retrieved,
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
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals/datasets/recall_v1.json")
    sys.exit(asyncio.run(run_reh(dataset)))
