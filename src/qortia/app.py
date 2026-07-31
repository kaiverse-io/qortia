"""Standalone FastAPI entrypoint.

Previously qortia only ever existed mounted inside a larger host platform's
own FastAPI app — this is the first time it boots on its own. Run with:

    uvicorn qortia.app:app

Background workers (embedding worker, archival, idle-reflection trigger,
weekly summary) are deliberately NOT started here — they're plain async
functions in qortia.reflect/qortia.knowledge meant to run in a separate
worker process, not inline with request handling. Wiring up that process is
tracked as a follow-up, not part of this entrypoint.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from qortia import config
from qortia.common import close_litellm_client, init_litellm_client
from qortia.db import close_main_pool, init_main_pool
from qortia.eval_router import router as eval_router
from qortia.knowledge import load_spacy_model
from qortia.router import router as qortia_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    init_litellm_client()
    await init_main_pool()
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
app.include_router(qortia_router)
if config.settings.eval_mode:
    app.include_router(eval_router)
