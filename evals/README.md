# evals/

Real memory-quality harnesses for standalone Qortia (not a chassis placeholder).

> The chassis ADLC workflow still mentions `evals/golden/` as a bucket-D slot.
> That gate is separate from the harnesses below — do not confuse the two.

## Harnesses

| Script | Layer | Role |
|--------|-------|------|
| `run_reh.py` | Retrieval | Seeded-ID hit@K / ranking (deterministic) |
| `run_alb.py` | Agentic loop | Does the agent use the retrieved fact? |
| `run_pib.py` | Infra | Latency / cost / scale probes against a live API |
| `run_temporal_eval.py` | Temporal | `valid_from` / `valid_until` recall behavior |
| `run_longmemeval.py` | Long-context | LongMemEval-style cases |
| `run_longitudinal_eval.py` | Longitudinal | Multi-turn memory stability |
| `run_extraction_eval.py` | NER / extract | Entity extraction quality |

Supporting: `dataset_loader.py`, `mlflow_logger.py`, `datasets/`.

## Auth against standalone Qortia

Call the HTTP API with:

- `Authorization: Bearer <tenant API key>`
- `X-Agent-Id: <uuid belonging to that tenant>`

Some internal `/v1/internal/eval/*` routes exist only when `QORTIA_EVAL_MODE=true`
(`qortia.eval_router`). They are not a substitute for production auth.

## Live REH (dogfood)

```bash
just stack-up && just stack-pull-model   # or OLLAMA_MODELS_DIR=$HOME/.ollama just stack-up
# API (eval routes) + worker — use :8090 if :8080 is taken
export QORTIA_DATABASE_URL=postgresql://qortia_platform:qortia_platform@127.0.0.1:5434/qortia
export QORTIA_LITELLM_URL=http://127.0.0.1:4000
export QORTIA_LITELLM_API_KEY=sk-qortia-local
export QORTIA_EVAL_MODE=true
uv run uvicorn qortia.app:app --host 127.0.0.1 --port 8090
just worker -- --only embed

QORTIA_URL=http://127.0.0.1:8090 PYTHONPATH=. uv run python evals/run_reh.py evals/datasets/recall_smoke.json
```

`QORTIA_URL` overrides the harness default (`http://localhost:8080`).

## Dataset provenance

`datasets/` includes fixtures whose *text content* mentions host-platform
concerns (OpenBao, Vault policies, etc.). That is **corpus content for retrieval
tests**, not a claim that standalone Qortia uses Vault. Prefer portable examples
when adding new fixtures.

## Related docs

- [`docs/02-benchmarking.md`](../docs/02-benchmarking.md)
- [`docs/03-eval-strategy.md`](../docs/03-eval-strategy.md)
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — eval_router component
