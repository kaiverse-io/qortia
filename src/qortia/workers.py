"""Background worker process entrypoint for standalone Qortia.

The FastAPI app handles HTTP only. Embeddings, archival, idle reflection, and
weekly summaries run here so request latency is not blocked by LiteLLM batch
work. Industry default for OSS memory services: API process + worker process.

Usage:
    qortia-worker                  # all loops
    qortia-worker --only embed     # embedding worker only
    just worker

Requires the same env as the API (database + LiteLLM). See
docs/how-to/embeddings.md.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from qortia.common import close_litellm_client, init_litellm_client
from qortia.db import close_main_pool, init_main_pool
from qortia.embeddings import validate_embedding_config
from qortia.knowledge import run_weekly_summary_task
from qortia.reflect import (
    run_archival_task,
    run_background_reflection_trigger,
    run_embedding_worker,
)

logger = logging.getLogger(__name__)

_WORKERS = {
    "embed": run_embedding_worker,
    "archive": run_archival_task,
    "idle-reflect": run_background_reflection_trigger,
    "weekly-summary": run_weekly_summary_task,
}


async def _run(only: list[str] | None) -> None:
    init_litellm_client()
    await init_main_pool()
    await validate_embedding_config()

    names = only or list(_WORKERS)
    unknown = [n for n in names if n not in _WORKERS]
    if unknown:
        raise SystemExit(f"unknown worker(s): {', '.join(unknown)}; choose from {list(_WORKERS)}")

    logger.info({"event": "qortia_workers_starting", "workers": names})
    try:
        await asyncio.gather(*(_WORKERS[n]() for n in names))
    finally:
        await close_litellm_client()
        await close_main_pool()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(_WORKERS),
        help="Run a subset of workers (repeatable). Default: all.",
    )
    args = parser.parse_args(argv)
    try:
        asyncio.run(_run(args.only))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
