# Vendored eval datasets

Git-tracked, not gitignored — the point is that `run_scale_eval.py` and
`run_ner_eval.py` run offline against a real, externally-judged corpus with no
network access and no re-download on every run.

Same corpora as agnova's `evals/datasets/` (this repo has no dependency on
agnova — see AGENTS.md's HTTP-only boundary — so the files are vendored here
independently rather than shared across a repo the harness never imports).

## fiqa/

BEIR / FiQA-2018 — 57,638 user-written financial-forum posts (StackExchange-style Q&A),
6,648 queries, 1,706 human relevance judgements on the `test` split.

- Source: https://github.com/beir-cellar/beir
- Fetched from: `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip`
- License: CC BY-SA 4.0 — https://creativecommons.org/licenses/by-sa/4.0/
- Vendored: 2026-08-11
- Files kept: `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv` (the `train`/`dev` qrels
  splits from the original zip are dropped — unused here)

Used for: `run_scale_eval.py` — qortia semantic recall's precision-at-budget and
context-cost at real corpus scale (~276/1000/10000 docs), independent of any other
harness in this ecosystem.

## wikiann/

WikiANN (PAN-X) — Wikipedia-derived named-entity tagging, `PER`/`ORG`/`LOC` spans in
IOB2, per-language `test` splits.

- Source: https://huggingface.co/datasets/unimelb-nlp/wikiann (fetched via the public
  `datasets-server.huggingface.co/rows` JSON API — no `datasets` library dependency)
- License: ODC-BY (per the HF dataset card)
- Vendored: 2026-08-11
- Languages kept: `hi`, `bn`, `ta`, `te`, `mr` (`_INDIC_MODEL` routing table,
  `src/qortia/knowledge.py`) + `en` (non-Indic default path) + one control language
  outside both (`de`), to exercise the `ner_lang_fallback_to_en` path rather than only
  the two designed-for routes.

Used for: `run_ner_eval.py` — entity extraction against `PER→PERSON`, `ORG→ORG`,
`LOC→GPE` (`_INDIC_LABEL_MAP` in `knowledge.py`). Re-fetch with `fetch_wikiann.py`.
