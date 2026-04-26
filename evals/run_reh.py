import json
import asyncio
import httpx
from uuid import UUID


async def run_eval(
    platform_url: str, dataset_path: str, tenant_id: UUID, agent_id: UUID
):
    with open(dataset_path, "r") as f:
        cases = json.load(f)

    total_cases = len(cases)
    recall_at_5 = 0
    mrr = 0.0

    async with httpx.AsyncClient(timeout=60.0) as client:
        for case in cases:
            print(f"Running case: {case['id']} - {case['query']}")
            resp = await client.post(
                f"{platform_url}/v1/internal/eval/recall",
                params={
                    "query": case["query"],
                    "tenant_id": str(tenant_id),
                    "agent_id": str(agent_id),
                    "limit": 5,
                },
            )
            resp.raise_for_status()
            results = resp.json()["results"]

            # Since our seeded memories in DB will have dynamic IDs,
            # we check for content match instead of ID match in this basic version,
            # or we map the content to the ID during seeding.

            # For this version, we'll look for the ground truth content in the results.
            # Ground truth IDs in the JSON correspond to the 'id' field in the 'memories' list.
            gt_contents = {
                m["content"]
                for m in case["memories"]
                if m["id"] in case["ground_truth_ids"]
            }

            found_rank = None
            for i, res in enumerate(results):
                if res["content"] in gt_contents:
                    found_rank = i + 1
                    break

            if found_rank:
                recall_at_5 += 1
                mrr += 1.0 / found_rank
                print(f"  MATCH at rank {found_rank}")
            else:
                print("  MISS")

    print("\n" + "=" * 40)
    print("RECALL EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total cases: {total_cases}")
    print(f"Recall@5:    {recall_at_5 / total_cases:.2%}")
    print(f"MRR:         {mrr / total_cases:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python3 run_reh.py <platform_url> <dataset_path> <tenant_id> <agent_id>"
        )
        sys.exit(1)

    asyncio.run(
        run_eval(sys.argv[1], sys.argv[2], UUID(sys.argv[3]), UUID(sys.argv[4]))
    )
