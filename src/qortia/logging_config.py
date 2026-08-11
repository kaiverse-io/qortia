"""One logging configuration, shared by every qortia entrypoint.

Before this existed: `workers.py` called `logging.basicConfig(...)` for the
`qortia-worker` process; `app.py` (the actual API server) never configured
logging at all. Every `qortia.*` module logger — knowledge, entity_graph,
links, remember, recall, reflect, all of them — has no handler anywhere in its
hierarchy in that process, so records fall through to Python's
`logging.lastResort`: a bare stderr handler with no formatter and a hardcoded
WARNING floor. Two visible consequences of that, both reported directly
against a running stack: `logger.info(...)` calls anywhere in `app`'s process
go nowhere (not quieter — gone), and anything that *does* clear the WARNING
floor prints as a bare `str(dict)` with no timestamp, level, or logger name —
next to uvicorn's own access-log lines, which uvicorn formats itself, so the
same log stream visibly alternates between two unrelated formats.

That fixes qortia's own logging, but not the whole visible inconsistency —
uvicorn's `INFO:     <message>` access-log lines are a *second*, independent
logging setup that survives `basicConfig()` untouched. `uvicorn qortia.app:app`
calls `Config.configure_logging()` (its own `dictConfig`, attaching its own
formatter to the `uvicorn`/`uvicorn.error`/`uvicorn.access` loggers with
`propagate=False`) as part of server startup, which completes *before* the
ASGI app's lifespan — and therefore this module's `configure()` call inside
it — ever runs. `propagate=False` means nothing reaching root's handler
touches those three loggers either. Two genuinely separate systems, and
calling `configure()` from `lifespan()` can only ever fix one of them by
itself.

`configure()` therefore does both: `basicConfig()` for every qortia.* logger
(and anything else that propagates to root — `workers.py`'s loggers, third-party
libraries that don't set up their own handler), then reaches into uvicorn's
three already-configured loggers and swaps just their Formatter, leaving
their handler and level alone. Uvicorn builds the access-log message content
itself (`record.getMessage()`, independent of the Formatter) — swapping the
Formatter changes only the timestamp/level wrapper around that content, which
is exactly the visible inconsistency this exists to close: every line in the
stream gets the same `<timestamp> <LEVEL> <message>` shape, uvicorn's own
access/error lines included.
"""

from __future__ import annotations

import logging
import os

# Matches the format workers.py used on its own before this module existed —
# kept rather than replaced so existing log-scraping/greps built against that
# shape (timestamp, level, message) keep working; app.py now produces the
# same shape instead of the level-less dict reprs it produced before.
_FORMAT = "%(asctime)s %(levelname)s %(message)s"

# propagate=False on all three (uvicorn's own dictConfig) — reformatting them
# is the only way to reach them; nothing routed through root's handler can.
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

_configured = False


def configure() -> None:
    """Idempotent — safe to call from every entrypoint without double-attaching
    handlers (importing both `app` and `workers` in the same test process, for
    instance, would otherwise configure the root logger twice)."""
    global _configured
    if _configured:
        return
    level_name = os.environ.get("QORTIA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=_FORMAT)

    formatter = logging.Formatter(_FORMAT)
    for name in _UVICORN_LOGGERS:
        # No-op if uvicorn hasn't configured this logger yet (or at all — e.g.
        # under pytest, or in the worker process, neither of which run under
        # uvicorn): .handlers is just empty, nothing to reformat.
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(formatter)

    _configured = True
