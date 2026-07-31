"""Best-effort OTel counters — safe no-ops if OpenTelemetry isn't configured.

Replaces `app.telemetry.instruments`. Every call site already wraps use of
these in `try/except Exception: pass`, so a no-op fallback when
`opentelemetry-api` isn't installed is a correct, not a degraded, mode.
"""

from __future__ import annotations

from typing import Any, Protocol, cast


class _Counter(Protocol):
    def add(self, amount: int, attributes: dict[str, Any] | None = None) -> None: ...


class _NoOpCounter:
    def add(self, amount: int, attributes: dict[str, Any] | None = None) -> None:
        return None


def _make_counter(name: str, description: str) -> _Counter:
    try:
        from opentelemetry import metrics

        meter = metrics.get_meter("qortia")
        # opentelemetry isn't a declared dependency (optional best-effort telemetry),
        # so its types aren't visible to mypy here — this is a real Any, not silenced.
        return cast(_Counter, meter.create_counter(name, description=description))
    except Exception:
        return _NoOpCounter()


qortia_recall_degraded: _Counter = _make_counter(
    "qortia.recall.degraded",
    "Incremented whenever recall falls back to a degraded search path (e.g. embedding failure).",
)
