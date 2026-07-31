# evals/

> **Bucket D — wired-but-waiting.** This slot is inert until a real model call exists in the
> project (activates P1 per the chassis build sequence). The CI job in
> `.github/workflows/adlc-gate.yaml` checks `evals/golden/` and is a documented no-op until
> it has content — it does not silently stay green forever.

## What goes here

- `golden/` — the golden work-order set: real (or representative) inputs + expected outputs/
  rubric scores. This is the eval harness's ground truth.
- A rubric/threshold config (e.g. `promptfoo.config.yaml` or equivalent) once the eval engine
  is chosen.

## Why eval-driven, not eval-at-the-end

The eval harness should exist at the **start** of the phase that exercises real model calls,
not as an afterthought before shipping. An unaudited LLM judge self-certifies — audit it
(bias-test error rate) before trusting it as a gate.

## Activation

1. Add real work-order fixtures to `golden/`.
2. Wire a rubric + threshold config.
3. Remove the no-op branch in `.github/workflows/adlc-gate.yaml` — the gate now blocks for real.
4. Update this README's status line.
