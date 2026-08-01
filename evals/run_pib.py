"""
Infrastructure Performance Benchmark (PIB) for the Qortia recall pipeline.

Measures latency percentiles, embedding throughput, cost-per-recall, and
HNSW index overhead against a seeded corpus of synthetic memories.

Usage:
    python3 evals/run_pib.py <qortia_url> <tenant_id> <agent_id> \
        [--corpus-size 100] [--iterations 10]
"""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from uuid import UUID

import httpx

# ── Synthetic corpus ────────────────────────────────────────────

_MEMORY_TYPES = ("episodic", "experiential", "mental_model", "decision", "lesson")


def _generate_synthetic_memories(count: int) -> list[dict[str, object]]:
    """Generate *count* synthetic memories spread across types."""
    memories: list[dict[str, object]] = []
    for i in range(count):
        mtype = _MEMORY_TYPES[i % len(_MEMORY_TYPES)]
        memories.append(
            {
                "type": mtype,
                "content": (
                    f"Synthetic memory #{i}: this is a benchmark memory used for "
                    f"PIB infrastructure performance testing of recall latency and "
                    f"embedding throughput for the Qortia recall pipeline"
                ),
            }
        )
    return memories


async def _seed_corpus(
    client: httpx.AsyncClient,
    platform_url: str,
    tenant_id: str,
    agent_id: str,
    corpus_size: int,
) -> float:
    """Seed *corpus_size* memories via /v1/remember in batches of 50.
    Returns wall-clock seconds for the entire seeding operation."""
    batch_size = 50
    memories = _generate_synthetic_memories(corpus_size)
    start = time.perf_counter()

    for offset in range(0, len(memories), batch_size):
        batch = memories[offset : offset + batch_size]
        for mem in batch:
            resp = await client.post(
                f"{platform_url}/v1/internal/eval/seed-memory",
                json={
                    "agent_id": agent_id,
                    "tenant_id": tenant_id,
                    "content": mem["content"],
                    "mem_type": mem["type"],
                },
            )
            resp.raise_for_status()

    elapsed = time.perf_counter() - start
    print(f"Seeded {corpus_size} memories in {elapsed:.2f}s")
    return elapsed


# ── Recall latency measurement ──────────────────────────────────

_QUERIES = [
    "Who is Scout?",
    "Redis removal decision",
    "How to deploy an agent?",
    "What is the Chief Agent role?",
    "Memory history audit trail",
]


async def _measure_recall_latencies(
    client: httpx.AsyncClient,
    platform_url: str,
    tenant_id: str,
    agent_id: str,
    iterations: int,
) -> list[float]:
    """Fire *iterations* x len(_QUERIES) recall requests. Returns all latencies."""
    latencies: list[float] = []

    for i in range(iterations):
        for query in _QUERIES:
            start = time.perf_counter()
            resp = await client.post(
                f"{platform_url}/v1/internal/eval/recall",
                params={
                    "query": query,
                    "tenant_id": tenant_id,
                    "agent_id": agent_id,
                    "limit": 10,
                },
            )
            resp.raise_for_status()
            duration = time.perf_counter() - start
            latencies.append(duration)
            print(f"  iter {i + 1}, '{query}': {duration:.3f}s")

    return latencies


# ── Embedding throughput ────────────────────────────────────────


async def _measure_embedding_throughput(
    client: httpx.AsyncClient,
    platform_url: str,
    tenant_id: str,
    agent_id: str,
    wait_seconds: int = 15,
) -> dict[str, object]:
    """Query pending_embedding count, wait, then query again to derive throughput."""

    async def _pending_count() -> int:
        resp = await client.get(
            f"{platform_url}/v1/internal/eval/pending-embeddings",
            params={"tenant_id": tenant_id, "agent_id": agent_id},
        )
        if resp.status_code == 404:
            return 0
        resp.raise_for_status()
        return int(resp.json().get("count", 0))

    before = await _pending_count()
    if before == 0:
        return {"pending_before": 0, "pending_after": 0, "throughput_per_sec": 0.0}

    await asyncio.sleep(wait_seconds)
    after = await _pending_count()
    processed = max(0, before - after)
    throughput = processed / wait_seconds if wait_seconds > 0 else 0.0
    return {
        "pending_before": before,
        "pending_after": after,
        "throughput_per_sec": round(throughput, 2),
    }


# ── Cost per recall ─────────────────────────────────────────────


async def _measure_cost_per_recall(
    client: httpx.AsyncClient,
    platform_url: str,
    tenant_id: str,
    agent_id: str,
    recall_count: int,
) -> dict[str, object]:
    """Query agent_cost_ledger sum for the measured recall calls."""
    resp = await client.get(
        f"{platform_url}/v1/internal/eval/cost-ledger",
        params={"tenant_id": tenant_id, "agent_id": agent_id},
    )
    if resp.status_code == 404:
        return {"total_cost": 0.0, "cost_per_recall": 0.0}
    resp.raise_for_status()
    total = float(resp.json().get("total_cost", 0.0))
    cost_per = total / recall_count if recall_count > 0 else 0.0
    return {"total_cost": round(total, 6), "cost_per_recall": round(cost_per, 6)}


# ── HNSW index overhead ────────────────────────────────────────


async def _measure_hnsw_overhead(
    client: httpx.AsyncClient,
    platform_url: str,
    tenant_id: str,
    agent_id: str,
) -> dict[str, object]:
    """Compare pg_relation_size('hindsight_memories') vs index size."""
    resp = await client.get(
        f"{platform_url}/v1/internal/eval/table-sizes",
        params={"tenant_id": tenant_id, "agent_id": agent_id},
    )
    if resp.status_code == 404:
        return {"table_bytes": 0, "index_bytes": 0, "overhead_ratio": 0.0}
    resp.raise_for_status()
    data = resp.json()
    table_bytes = int(data.get("table_bytes", 0))
    index_bytes = int(data.get("index_bytes", 0))
    ratio = index_bytes / table_bytes if table_bytes > 0 else 0.0
    return {
        "table_bytes": table_bytes,
        "index_bytes": index_bytes,
        "overhead_ratio": round(ratio, 3),
    }


# ── Percentile helper ──────────────────────────────────────────


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) from a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


# ── Main ────────────────────────────────────────────────────────


async def run_pib(
    platform_url: str,
    tenant_id: UUID,
    agent_id: UUID,
    iterations: int = 10,
    corpus_size: int = 100,
) -> None:
    tid = str(tenant_id)
    aid = str(agent_id)

    async with httpx.AsyncClient(timeout=120.0) as client:
        # 1. Seed corpus
        print(f"\n{'=' * 50}")
        print(f"PIB: Seeding {corpus_size} synthetic memories")
        print("=" * 50)
        seed_time = await _seed_corpus(client, platform_url, tid, aid, corpus_size)

        # 2. Recall latency
        print(f"\n{'=' * 50}")
        print("PIB: Measuring recall latency")
        print("=" * 50)
        latencies = await _measure_recall_latencies(client, platform_url, tid, aid, iterations)
        sorted_lat = sorted(latencies)
        avg = sum(latencies) / len(latencies) if latencies else 0.0
        p50 = _percentile(sorted_lat, 50)
        p95 = _percentile(sorted_lat, 95)
        p99 = _percentile(sorted_lat, 99)

        # 3. Embedding throughput
        print(f"\n{'=' * 50}")
        print("PIB: Measuring embedding throughput")
        print("=" * 50)
        emb_throughput = await _measure_embedding_throughput(client, platform_url, tid, aid)

        # 4. Cost per recall
        print(f"\n{'=' * 50}")
        print("PIB: Measuring cost per recall")
        print("=" * 50)
        recall_count = len(latencies)
        cost = await _measure_cost_per_recall(client, platform_url, tid, aid, recall_count)

        # 5. HNSW index overhead
        print(f"\n{'=' * 50}")
        print("PIB: Measuring HNSW index overhead")
        print("=" * 50)
        hnsw = await _measure_hnsw_overhead(client, platform_url, tid, aid)

    # ── Report ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("INFRASTRUCTURE PERFORMANCE BENCHMARK (PIB)")
    print("=" * 60)

    print(f"\n  Corpus size:        {corpus_size}")
    print(f"  Seed time:          {seed_time:.2f}s")
    print(f"  Total requests:     {len(latencies)}")

    print(f"\n  Avg Latency:        {avg:.3f}s")
    print(f"  p50 Latency:        {p50:.3f}s")
    print(f"  p95 Latency:        {p95:.3f}s")
    print(f"  p99 Latency:        {p99:.3f}s  (target <0.400s)")

    p99_status = "PASS" if p99 < 0.400 else "FAIL"
    print(f"  p99 Gate:           {p99_status}")

    print(f"\n  Embedding pending (before): {emb_throughput['pending_before']}")
    print(f"  Embedding pending (after):  {emb_throughput['pending_after']}")
    print(f"  Embedding throughput:       {emb_throughput['throughput_per_sec']}/s")

    print(f"\n  Total cost:         ${cost['total_cost']}")
    print(f"  Cost per recall:    ${cost['cost_per_recall']}")

    print(f"\n  Table size:         {hnsw['table_bytes']} bytes")
    print(f"  Index size:         {hnsw['index_bytes']} bytes")
    print(f"  HNSW overhead:      {hnsw['overhead_ratio']}x")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qortia Infrastructure Performance Benchmark (PIB)"
    )
    parser.add_argument("platform_url", help="Platform base URL (e.g. http://localhost:8080)")
    parser.add_argument("tenant_id", type=UUID, help="Tenant UUID")
    parser.add_argument("agent_id", type=UUID, help="Agent UUID")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Recall iterations (default: 10)"
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=100,
        help="Synthetic memories to seed (default: 100, full: 1000)",
    )

    args = parser.parse_args()
    asyncio.run(
        run_pib(
            args.platform_url,
            args.tenant_id,
            args.agent_id,
            iterations=args.iterations,
            corpus_size=args.corpus_size,
        )
    )
