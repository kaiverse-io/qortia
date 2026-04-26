import time
import asyncio
import httpx
from uuid import UUID


async def run_pib(
    platform_url: str, tenant_id: UUID, agent_id: UUID, iterations: int = 10
):
    queries = [
        "Who is Scout?",
        "Redis removal decision",
        "How to deploy an agent?",
        "What is the Chief Agent role?",
        "Memory history audit trail",
    ]

    latencies = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(iterations):
            for query in queries:
                start = time.perf_counter()
                resp = await client.post(
                    f"{platform_url}/v1/internal/eval/recall",
                    params={
                        "query": query,
                        "tenant_id": str(tenant_id),
                        "agent_id": str(agent_id),
                        "limit": 10,
                    },
                )
                resp.raise_for_status()
                duration = time.perf_counter() - start
                latencies.append(duration)
                print(f"Iteration {i+1}, Query '{query}': {duration:.3f}s")

    avg = sum(latencies) / len(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]

    print("\n" + "=" * 40)
    print("INFRASTRUCTURE PERFORMANCE (PIB)")
    print("=" * 40)
    print(f"Total requests: {len(latencies)}")
    print(f"Avg Latency:    {avg:.3f}s")
    print(f"p95 Latency:    {p95:.3f}s")
    print("=" * 40)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python3 run_pib.py <platform_url> <tenant_id> <agent_id> [iterations]"
        )
        sys.exit(1)

    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    asyncio.run(run_pib(sys.argv[1], UUID(sys.argv[2]), UUID(sys.argv[3]), iters))
