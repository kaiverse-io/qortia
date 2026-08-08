"""Standalone FastAPI entrypoint.

Run with:

    uvicorn qortia.app:app

Background workers (embedding, archival, idle-reflection, weekly summary) are
deliberately NOT started here — run `qortia-worker` / `just worker` alongside
the API. See docs/how-to/embeddings.md.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from qortia import config
from qortia.admin_router import router as admin_router
from qortia.common import close_litellm_client, init_litellm_client
from qortia.db import close_main_pool, init_main_pool
from qortia.embeddings import validate_embedding_config
from qortia.eval_router import router as eval_router
from qortia.knowledge import load_spacy_model
from qortia.router import router as qortia_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    init_litellm_client()
    await init_main_pool()
    await validate_embedding_config()
    try:
        load_spacy_model()
    except Exception as exc:
        # Best-effort: NER extraction call sites already degrade gracefully
        # to an empty entity list when no model is loaded.
        logger.warning({"event": "spacy_model_load_failed_at_startup", "error": str(exc)})
    try:
        yield
    finally:
        await close_litellm_client()
        await close_main_pool()


app = FastAPI(title="Qortia", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Bare root has no API meaning of its own — send a human to the docs
    rather than a bare {"detail": "Not Found"} for the first URL anyone
    types."""
    return RedirectResponse("/docs")


app.include_router(qortia_router)
if config.settings.eval_mode:
    app.include_router(eval_router)
if config.settings.qortia_admin_token:
    app.include_router(admin_router)
