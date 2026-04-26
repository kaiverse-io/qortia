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

PLATFORM_URL = "http://localhost:8080"
EMBEDDING_WAIT_SECONDS = 15  # embedding worker runs every 10s; 15s gives one full cycle


def _agent_headers(agent_id: str, tenant_id: str) -> dict[str, str]:
    """Trusted headers for local-env agent identity (no JWT required)."""
    return {"X-Agent-ID": agent_id, "X-Tenant-ID": tenant_id}


async def provision_eval_agent(
    client: httpx.AsyncClient,
) -> tuple[str, str]:
    """Provision a fresh eval agent. Returns (tenant_id, agent_id)."""
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
    return tenant_id, agent_id


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
) -> dict[int, str]:
    """Seed org memories. Returns {index: org_memory_id}."""
    headers = _agent_headers(agent_id, tenant_id)
    org_id_map: dict[int, str] = {}
    for i, om in enumerate(case["setup"].get("org_memories", [])):
        resp = await client.post("/v1/remember-org", json=om, headers=headers)
        if resp.status_code == 200:
            org_id_map[i] = resp.json().get("id", "")
    return org_id_map


async def seed_knowledge(
    client: httpx.AsyncClient,
    case: dict[str, Any],
    tenant_id: str,
    agent_id: str,
) -> list[str]:
    """Seed knowledge entries. Returns list of source_paths (used for content lookup)."""
    headers = _agent_headers(agent_id, tenant_id)
    source_paths: list[str] = []
    for k in case["setup"].get("knowledge", []):
        resp = await client.post("/v1/knowledge", json=k, headers=headers)
        if resp.status_code == 200:
            source_paths.append(k.get("source_path", ""))
    return source_paths


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data["cases"]
