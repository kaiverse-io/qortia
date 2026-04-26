import json
import asyncio
import httpx
from uuid import UUID


async def seed_dataset(
    platform_url: str, dataset_path: str, tenant_id: UUID, agent_id: UUID
):
    with open(dataset_path, "r") as f:
        cases = json.load(f)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Seed agent
        print(f"Seeding agent {agent_id}...")
        resp = await client.post(
            f"{platform_url}/v1/internal/eval/seed-agent",
            params={
                "agent_id": str(agent_id),
                "tenant_id": str(tenant_id),
                "name": "eval_bot",
                "role": "custom",
            },
        )
        resp.raise_for_status()

        # 2. Seed memories
        for case in cases:
            for mem in case["memories"]:
                print(f"Seeding memory {mem['id']}...")
                resp = await client.post(
                    f"{platform_url}/v1/internal/eval/seed-memory",
                    json={
                        "agent_id": str(agent_id),
                        "tenant_id": str(tenant_id),
                        "content": mem["content"],
                        "mem_type": mem["type"],
                        "scope": mem["scope"],
                    },
                )
                resp.raise_for_status()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python3 dataset_loader.py <platform_url> <dataset_path> <tenant_id> <agent_id>"
        )
        sys.exit(1)

    asyncio.run(
        seed_dataset(sys.argv[1], sys.argv[2], UUID(sys.argv[3]), UUID(sys.argv[4]))
    )
