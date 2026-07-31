"""
evals/mlflow_logger.py — Log eval scores to MLflow + OTel span attributes.
ADR-103 §13.4: per-agent eval quality visibility.
"""

from __future__ import annotations

import logging
from uuid import UUID

logger = logging.getLogger(__name__)


def log_eval_score(
    agent_id: UUID,
    task_id: str,
    metric: str,
    score: float,
    metadata: dict,  # type: ignore[type-arg]
) -> None:
    """
    Log an eval score to MLflow and emit as OTel span attributes.
    agent_id is used for scoping only — never as a Prometheus/Mimir label.
    """
    # MLflow logging
    try:
        import mlflow

        with mlflow.start_run(run_name=f"{task_id}_{metric}", nested=True):
            mlflow.log_metric(metric, score)
            mlflow.log_param("agent_id", str(agent_id))
            mlflow.log_param("task_id", task_id)
            for k, v in metadata.items():
                mlflow.log_param(k, str(v))
    except Exception as exc:
        logger.warning({"event": "mlflow_log_failed", "error": str(exc)})

    # OTel span attribute emission
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("eval.metric", metric)
            span.set_attribute("eval.score", score)
            span.set_attribute("eval.task_id", task_id)
            # agent_id as span attribute only — never as a metric label
            span.set_attribute("eval.agent_id", str(agent_id))
    except Exception as exc:
        logger.debug({"event": "otel_span_attr_failed", "error": str(exc)})

    logger.info(
        {
            "event": "eval_score_logged",
            "task_id": task_id,
            "metric": metric,
            "score": score,
        }
    )
