"""Qortia multilingual NER, scored against WikiANN gold spans.

Everything else in this eval directory tests recall; this tests a different
code path entirely: qortia.knowledge.extract_entities_with_types(), called
synchronously inside remember() (see src/qortia/remember.py), which routes
hi/bn/ta/te/mr to the multilingual xx_ent_wiki_sm spaCy pipeline and
everything else (including the "unsupported" control language here) through
en_core_web_sm with a logged ner_lang_fallback_to_en event.

That routing table has never been measured against real text in any of the
five Indic languages it claims to support — this is that measurement, using
WikiANN (evals/datasets/wikiann/, vendored — see evals/datasets/README.md for
source/license/fetch_wikiann.py) because its PER/ORG/LOC tags map directly onto
qortia's own _INDIC_LABEL_MAP (PER->PERSON, ORG->ORG, LOC->GPE).

Runs entirely over HTTP against /v1/internal/eval/seed-memory (EVAL_MODE-gated,
no admin token needed): that endpoint calls the same
extract_entities_with_types() /v1/remember uses and returns the extracted
entities directly in its response — added specifically so this harness needs
neither an admin token nor a database connection.

Two matching modes are reported, because they answer different questions:
    text match      did qortia find *an* entity overlapping the gold span at
                     all, regardless of label — "is NER finding real things"
    text+type match did it also land on the right qortia-side type — "is the
                     Indic label mapping correct," specifically testing
                     _INDIC_LABEL_MAP for the five Indic languages
Matching is substring-based both directions after casefolding, not exact
string equality — spaCy's exact tokenisation/boundary choices routinely
differ from WikiANN's pre-tokenised gold by a trailing particle or punctuation
mark, and exact-match would undercount correct extractions on that alone
rather than on the model being wrong.

zero_extraction_rate is the sharpest single number here: the fraction of
sentences with >=1 gold entity where qortia found none at all. That is a
silent failure an agent has no way to detect from the API response —
seed-memory (and /v1/remember) return 200 either way.

Usage:
    QORTIA_EVAL_MODE=true uv run python evals/run_ner_eval.py [--n 100] [--langs hi,bn,en]
    QORTIA_URL=http://localhost:8080 (env or --base-url)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from evals.dataset_loader import QORTIA_URL, provision_eval_agent

DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "wikiann"
_CONCURRENCY = 20

# src/qortia/knowledge.py's own map, mirrored here so gold WikiANN types
# score against what qortia is actually trying to produce, not WikiANN's raw
# PER/ORG/LOC. GPE is the deliberate LOC standin qortia itself uses.
_GOLD_TYPE_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "GPE"}


async def _seed_and_extract(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    agent_id: str,
    tenant_id: str,
    text: str,
    lang: str,
) -> list[tuple[str, str]]:
    async with sem:
        resp = await client.post(
            "/v1/internal/eval/seed-memory",
            json={
                "agent_id": agent_id,
                "tenant_id": tenant_id,
                "content": text,
                "mem_type": "episodic",
                "lang": lang,
            },
        )
        resp.raise_for_status()
        return [(t[0], t[1]) for t in resp.json()["entities"]]


def _norm(s: str) -> str:
    return " ".join(s.casefold().split())


def _matches(pred_text: str, gold_text: str) -> bool:
    p, g = _norm(pred_text), _norm(gold_text)
    return bool(p) and bool(g) and (p in g or g in p)


def _score_language(
    examples: list[dict[str, Any]], predicted: list[list[tuple[str, str]]]
) -> dict[str, Any]:
    text_tp = type_tp = n_pred = n_gold = zero_extraction = 0
    for ex, preds in zip(examples, predicted, strict=True):
        gold = ex["gold"]  # [{"type": "PER"/"ORG"/"LOC", "text": ...}, ...]
        n_gold += len(gold)
        n_pred += len(preds)
        if gold and not preds:
            zero_extraction += 1
        matched_gold_idx: set[int] = set()
        for p_text, p_type in preds:
            for gi, g in enumerate(gold):
                if gi in matched_gold_idx:
                    continue
                if _matches(p_text, g["text"]):
                    text_tp += 1
                    if _GOLD_TYPE_MAP.get(g["type"]) == p_type:
                        type_tp += 1
                    matched_gold_idx.add(gi)
                    break
    precision = text_tp / n_pred if n_pred else 0.0
    recall = text_tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    type_precision = type_tp / n_pred if n_pred else 0.0
    return {
        "n_examples": len(examples),
        "n_gold_entities": n_gold,
        "n_predicted_entities": n_pred,
        "text_precision": precision,
        "text_recall": recall,
        "text_f1": f1,
        "type_precision": type_precision,  # of text matches, fraction also correctly typed
        "zero_extraction_rate": zero_extraction / len(examples) if examples else 0.0,
    }


async def _run(base_url: str, n: int, langs: list[str]) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        tenant_id, agent_id = await provision_eval_agent(client)
        print(f"tenant {tenant_id} agent {agent_id}\n")

        header = (
            f"{'lang':>5} {'n':>4} {'gold':>5} {'pred':>5} {'P':>6} {'R':>6} {'F1':>6} "
            f"{'type-P':>7} {'zero%':>6}"
        )
        print(header)
        print("-" * len(header))

        report: dict[str, Any] = {"dataset": "wikiann", "n_per_lang": n, "runs": []}
        sem = asyncio.Semaphore(_CONCURRENCY)
        for lang in langs:
            path = DATASET_DIR / f"{lang}.json"
            if not path.is_file():
                print(
                    f"{lang:>5}  (not vendored — run evals/fetch_wikiann.py first)",
                    file=sys.stderr,
                )
                continue
            examples = json.loads(path.read_text(encoding="utf-8"))[:n]

            predicted = await asyncio.gather(
                *(
                    _seed_and_extract(client, sem, agent_id, tenant_id, ex["text"], lang)
                    for ex in examples
                )
            )

            m = _score_language(examples, list(predicted))
            print(
                f"{lang:>5} {m['n_examples']:>4} {m['n_gold_entities']:>5} "
                f"{m['n_predicted_entities']:>5} "
                f"{m['text_precision']:>6.3f} {m['text_recall']:>6.3f} {m['text_f1']:>6.3f} "
                f"{m['type_precision']:>7.3f} {m['zero_extraction_rate'] * 100:>5.0f}%"
            )
            report["runs"].append({"lang": lang, **m})

        return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="examples per language")
    ap.add_argument("--langs", default="hi,bn,ta,te,mr,en,de")
    ap.add_argument("--base-url", default=QORTIA_URL)
    args = ap.parse_args()

    report = asyncio.run(_run(args.base_url, args.n, args.langs.split(",")))

    out = Path(__file__).resolve().parent / "results" / "ner_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
