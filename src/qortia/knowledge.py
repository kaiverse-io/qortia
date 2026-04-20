from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_nlp = None


def load_spacy_model() -> None:
    global _nlp
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    logger.info({"event": "spacy_model_loaded", "model": "en_core_web_sm"})


def get_nlp():  # type: ignore[return]
    assert _nlp is not None, "spaCy model not loaded — call load_spacy_model() at startup"
    return _nlp
