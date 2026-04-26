"""
Shared utilities for seeding eval agents and memories via the platform API.
Used by run_reh.py, run_alb.py, run_pib.py, and run_comparative.py.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import httpx

PLATFORM_URL = "http://localhost:8000"
EMBEDDING_WAIT_SECONDS = 12  # embedding worker runs every 10s


async def provision_eval_agent(
    client: httpx.AsyncClient,
) -> tuple[str, str, str]:
    """Provision a fresh eval agent. Returns (tenant_id, agent_id, token)."""
    tenant_id = str(uuid.uuid4())
    agent_id = str(uuid.uuid4())
    resp = await client.post(
        "/v1/internal/eval/seed-agent",
        params={
            "agent_id": agent_id,
            "tenant_id": tenant_id,
            "name": "eval_agent",
            "role": "chief",
        },
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token", "")
    return tenant_id, agent_id, token


async def seed_case(
    client: httpx.AsyncClient,
    case: dict[str, Any],
    tenant_id: str,
    agent_id: str,
) -> dict[int, str]:
    """
    Seeds all memories (setup.memories + setup.hard_negatives) for a case.
    Returns {index_in_setup_memories: seeded_memory_id}.
    """
    setup = case["setup"]
    all_memories = setup["memories"] + setup.get("hard_negatives", [])

    id_map: dict[int, str] = {}
    for i, mem in enumerate(all_memories):
        resp = await client.post(
            "/v1/internal/eval/seed-memory",
            json={
                "agent_id": agent_id,
                "tenant_id": tenant_id,
                "content": mem["content"],
                "mem_type": mem.get("type", "episodic"),
                "scope": mem.get("scope", "private"),
            },
        )
        resp.raise_for_status()
        mem_id = resp.json()["memory_id"]
        if i < len(setup["memories"]):
            id_map[i] = mem_id

    return id_map


async def seed_org_memories(
    client: httpx.AsyncClient,
    case: dict[str, Any],
    tenant_id: str,
    agent_id: str,
    token: str,
) -> None:
    for om in case["setup"].get("org_memories", []):
        await client.post(
            "/v1/remember-org",
            json=om,
            headers={"Authorization": f"Bearer {token}"},
        )


async def seed_knowledge(
    client: httpx.AsyncClient,
    case: dict[str, Any],
    token: str,
) -> None:
    for k in case["setup"].get("knowledge", []):
        await client.post(
            "/v1/knowledge",
            json=k,
            headers={"Authorization": f"Bearer {token}"},
        )


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data["cases"]
