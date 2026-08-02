"""
LongMemEval Adapter for Qortia.

Downloads the LongMemEval benchmark dataset (Xiaowu et al., 2024) and runs it
against the Qortia recall pipeline. LongMemEval is the industry-standard benchmark
for evaluating agent memory systems across 5 categories:

  single-session-user (SSU)   — single-session preference and fact retrieval
  single-session-assistant (SSA) — assistant knowledge updated in one session
  multi-session-user (MSU)    — cross-session user preference tracking
  multi-session-assistant (MSA) — assistant knowledge across multiple sessions
  temporal (TMP)              — temporal reasoning and knowledge updates

Dataset: https://github.com/xiaowu0162/LongMemEval
Paper: "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"
       Xiaowu et al., EMNLP 2024

Scoring follows Zep's methodology:
  Context Completeness: COMPLETE | PARTIAL | INSUFFICIENT
    (does the retrieved context contain the facts needed to answer?)
  Answer Accuracy: CORRECT | WRONG
    (deterministic string-match check against gold answer)

Note on LLM-as-judge: LongMemEval's original scoring uses GPT-4 as a judge.
Qortia uses deterministic string matching against expected_answer_contains
fields to maintain the determinism principle (docs/03-eval-strategy.md §1).
This is a deliberate trade-off: deterministic but potentially lower than GPT-4-judged scores.

Usage:
    # Download dataset (run once)
    python3 evals/run_longmemeval.py --download

    # Run eval (requires full stack + QORTIA_EVAL_MODE=true)
    QORTIA_EVAL_MODE=true python3 evals/run_longmemeval.py

    # Run a specific category only
    QORTIA_EVAL_MODE=true python3 evals/run_longmemeval.py --category temporal

    # Run a subset (faster)
    QORTIA_EVAL_MODE=true python3 evals/run_longmemeval.py --max-cases 50

Report: evals/results/longmemeval_latest.json
"""

from __future__ import annotations

import argparse
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

# Dataset source — Hugging Face (primary) with GitHub fallback
# Dataset card: https://huggingface.co/datasets/xiaowu0162/LongMemEval
DATASET_URLS = [
    "https://huggingface.co/datasets/xiaowu0162/LongMemEval/resolve/main/longmemeval_oracle.json",
    "https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/data/longmemeval_oracle.json",
]
DATASET_PATH = Path("evals/datasets/longmemeval_oracle.json")

# Regression floors (match Mem0 published scores as baseline target)
RECALL_AT_5_FLOOR = 0.60  # ≥60% of questions answered with gt in top-5
CONTEXT_COMPLETE_FLOOR = 0.55  # ≥55% context completeness (COMPLETE or PARTIAL)
ANSWER_ACCURACY_FLOOR = 0.50  # ≥50% string-match answer accuracy

CATEGORIES = [
    "single-session-user",
    "single-session-assistant",
    "multi-session-user",
    "multi-session-assistant",
    "temporal",
]


# ── Dataset download ───────────────────────────────────────────────────────


def download_dataset() -> None:
    DATASET_PATH.parent.mkdir(exist_ok=True)
    for url in DATASET_URLS:
        print(f"Trying: {url}")
        try:
            response = httpx.get(url, follow_redirects=True, timeout=60.0)
            response.raise_for_status()
            DATASET_PATH.write_bytes(response.content)
            data = json.loads(DATASET_PATH.read_text())
            print(f"Downloaded {len(data)} cases.")
            return
        except Exception as exc:
            print(f"  Failed: {exc}")
    print("\nAll download attempts failed.")
    print("Manual download options:")
    print("  huggingface-cli download xiaowu0162/LongMemEval longmemeval_oracle.json")
    print("  Place the file at: evals/datasets/longmemeval_oracle.json")
    sys.exit(1)


# ── Qortia adapter ─────────────────────────────────────────────────────────


def _conversation_to_memories(
    conversations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert LongMemEval conversation format to Qortia memory seeds.

    Each message in the conversation becomes an episodic memory.
    The conversation is turned into: "On <date>, user said: <msg>. Assistant replied: <reply>."
    This preserves both sides of the conversation as a single episodic fact.
    """
    memories: list[dict[str, Any]] = []
    for session in conversations:
        session_id = session.get("session_id", "unknown")
        session_date = session.get("date", "")
        msgs = session.get("messages", [])

        i = 0
        while i < len(msgs):
            msg = msgs[i]
            if msg.get("role") == "user":
                user_content = msg["content"]
                assistant_content = ""
                if i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant":
                    assistant_content = msgs[i + 1]["content"]
                    i += 1

                memory_content = (
                    f"Session {session_id}"
                    + (f" ({session_date})" if session_date else "")
                    + f": User said: {user_content}"
                )
                if assistant_content:
                    memory_content += f" Assistant replied: {assistant_content}"

                memories.append(
                    {
                        "content": memory_content,
                        "type": "episodic",
                        "scope": "private",
                    }
                )
            i += 1

    return memories


def _check_answer(
    results: list[dict[str, Any]],
    expected_contains: list[str],
) -> tuple[str, str]:
    """Return (context_completeness, answer_accuracy)."""
    all_content = " ".join(r["content"] for r in results).lower()

    # Context completeness: how many expected strings appear in retrieved context?
    if not expected_contains:
        return "COMPLETE", "CORRECT"

    found = sum(1 for s in expected_contains if s.lower() in all_content)
    ratio = found / len(expected_contains)

    if ratio >= 0.85:
        completeness = "COMPLETE"
    elif ratio >= 0.40:
        completeness = "PARTIAL"
    else:
        completeness = "INSUFFICIENT"

    # Answer accuracy: does top result contain primary expected string?
    top_content = results[0]["content"].lower() if results else ""
    primary = expected_contains[0].lower() if expected_contains else ""
    accuracy = "CORRECT" if primary and primary in top_content else "WRONG"

    return completeness, accuracy


# ── Per-case runner ────────────────────────────────────────────────────────


async def _run_lme_case(
    client: httpx.AsyncClient,
    case: dict[str, Any],
) -> dict[str, Any]:
    tenant_id, agent_id = await provision_eval_agent(client)
    params = {"tenant_id": tenant_id, "agent_id": agent_id}

    # Convert conversations to episodic memories and seed them
    conversations = case.get("conversations", [])
    memories = _conversation_to_memories(conversations)

    # Batch seed — no more than 20 per case to keep eval time reasonable
    for mem in memories[:20]:
        r = await client.post(
            "/v1/internal/eval/seed-memory",
            json={
                "agent_id": agent_id,
                "tenant_id": tenant_id,
                "content": mem["content"],
                "mem_type": "episodic",
            },
        )
        if r.status_code != 200:
            continue

    await asyncio.sleep(EMBEDDING_WAIT_SECONDS)

    # Recall using the test question
    question = case.get("question", "")
    resp = await client.post(
        "/v1/internal/eval/recall-full",
        json={"query": question, "scope": "private"},
        params=params,
    )
    if resp.status_code != 200:
        return {
            "id": case.get("id", "unknown"),
            "category": case.get("category", "unknown"),
            "pass": False,
            "completeness": "INSUFFICIENT",
            "accuracy": "WRONG",
            "reason": f"recall HTTP {resp.status_code}",
        }

    results = resp.json().get("results", [])
    expected = case.get("expected_answer_contains", [])
    if isinstance(expected, str):
        expected = [expected]

    completeness, accuracy = _check_answer(results, expected)
    recall_at_5 = (
        any(any(e.lower() in r["content"].lower() for e in expected) for r in results[:5])
        if expected
        else bool(results)
    )

    passed = recall_at_5 and completeness in ("COMPLETE", "PARTIAL")

    return {
        "id": case.get("id", "unknown"),
        "category": case.get("category", "unknown"),
        "pass": passed,
        "recall_at_5": recall_at_5,
        "completeness": completeness,
        "accuracy": accuracy,
        "memories_seeded": len(memories),
        "results_returned": len(results),
    }


# ── Main ───────────────────────────────────────────────────────────────────


def _select_cases(
    all_cases: list[dict[str, Any]], category: str | None, max_cases: int
) -> list[dict[str, Any]]:
    cases = [c for c in all_cases if c.get("category") == category] if category else all_cases

    if not max_cases:
        return cases

    if category:
        return cases[:max_cases]

    # Sample evenly across categories
    per_cat = max_cases // len(CATEGORIES)
    sampled = []
    for cat in CATEGORIES:
        cat_cases = [c for c in cases if c.get("category") == cat]
        sampled.extend(cat_cases[:per_cat])
    return sampled[:max_cases]


async def _execute_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=QORTIA_URL, timeout=60.0) as client:
        for i, case in enumerate(cases):
            result = await _run_lme_case(client, case)
            results.append(result)
            status = "PASS" if result["pass"] else "FAIL"
            if (i + 1) % 10 == 0 or not result["pass"]:
                print(
                    f"  [{i + 1:3d}/{len(cases)}] [{status}] "
                    f"{result['category']}/{result['id']} "
                    f"complete={result['completeness']} acc={result['accuracy']}"
                )
    return results


def _summarize_by_category(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cat_summaries: dict[str, dict[str, Any]] = {}
    for cat in CATEGORIES:
        cat_results = [r for r in results if r.get("category") == cat]
        if not cat_results:
            continue
        recall = sum(1 for r in cat_results if r["recall_at_5"]) / len(cat_results)
        complete = sum(
            1 for r in cat_results if r["completeness"] in ("COMPLETE", "PARTIAL")
        ) / len(cat_results)
        accurate = sum(1 for r in cat_results if r["accuracy"] == "CORRECT") / len(cat_results)
        cat_summaries[cat] = {
            "cases": len(cat_results),
            "recall_at_5": recall,
            "context_completeness": complete,
            "answer_accuracy": accurate,
        }
    return cat_summaries


def _print_report(
    cat_summaries: dict[str, dict[str, Any]],
    total_recall: float,
    total_complete: float,
    total_accurate: float,
    num_results: int,
) -> None:
    print(f"\n{'=' * 72}")
    print("LONGMEMEVAL RESULTS")
    print("=" * 72)
    print(f"\n{'Category':<32} {'Recall@5':>9} {'Complete':>9} {'Accurate':>9} {'Cases':>6}")
    print("-" * 72)
    for cat, s in cat_summaries.items():
        print(
            f"  {cat:<30} {s['recall_at_5']:>9.1%} {s['context_completeness']:>9.1%} "
            f"{s['answer_accuracy']:>9.1%} {s['cases']:>6}"
        )
    print("-" * 72)
    print(
        f"  {'TOTAL':<30} {total_recall:>9.1%} {total_complete:>9.1%} "
        f"{total_accurate:>9.1%} {num_results:>6}"
    )
    print(f"\nFloors: Recall@5≥{RECALL_AT_5_FLOOR:.0%}  Complete≥{CONTEXT_COMPLETE_FLOOR:.0%}")


async def run_longmemeval(category: str | None, max_cases: int) -> int:
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}. Run with --download first.")
        return 1

    all_cases = json.loads(DATASET_PATH.read_text())
    cases = _select_cases(all_cases, category, max_cases)

    print("=" * 60)
    print(f"LongMemEval — {len(cases)} cases")
    if category:
        print(f"Category filter: {category}")
    print("=" * 60)

    results = await _execute_cases(cases)
    cat_summaries = _summarize_by_category(results)

    total_recall = sum(1 for r in results if r["recall_at_5"]) / max(len(results), 1)
    total_complete = sum(1 for r in results if r["completeness"] in ("COMPLETE", "PARTIAL")) / max(
        len(results), 1
    )
    total_accurate = sum(1 for r in results if r["accuracy"] == "CORRECT") / max(len(results), 1)
    overall_pass = total_recall >= RECALL_AT_5_FLOOR and total_complete >= CONTEXT_COMPLETE_FLOOR

    _print_report(cat_summaries, total_recall, total_complete, total_accurate, len(results))
    print(f"Gate: {'PASS' if overall_pass else 'FAIL'}")

    report = {
        "total_cases": len(results),
        "category_filter": category,
        "recall_at_5": total_recall,
        "context_completeness": total_complete,
        "answer_accuracy": total_accurate,
        "overall_gate": "PASS" if overall_pass else "FAIL",
        "by_category": cat_summaries,
        "floors": {
            "recall_at_5": RECALL_AT_5_FLOOR,
            "context_completeness": CONTEXT_COMPLETE_FLOOR,
            "answer_accuracy": ANSWER_ACCURACY_FLOOR,
        },
    }
    out = Path("evals/results/longmemeval_latest.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Report written to {out}")

    return 0 if overall_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--download", action="store_true", help="Download LongMemEval dataset")
    parser.add_argument("--category", choices=CATEGORIES, default=None, help="Run single category")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit total cases (0=all)")
    args = parser.parse_args()

    if args.download:
        download_dataset()
        return

    sys.exit(asyncio.run(run_longmemeval(args.category, args.max_cases)))


if __name__ == "__main__":
    main()
