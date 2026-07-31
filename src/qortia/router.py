from __future__ import annotations

from fastapi import APIRouter

from qortia.knowledge import router as knowledge_router
from qortia.recall import router as recall_router
from qortia.reflect import router as reflect_router
from qortia.remember import router as remember_router

router = APIRouter(tags=["qortia"])
router.include_router(remember_router)
router.include_router(reflect_router)
router.include_router(recall_router)
router.include_router(knowledge_router)
