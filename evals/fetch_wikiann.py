"""One-time fetch: vendor bounded WikiANN test-split samples into
evals/datasets/wikiann/<lang>.json so run_ner_eval.py runs offline
and doesn't hit HuggingFace on every eval run.

Uses the public datasets-server rows API directly (plain HTTPS + json, paginated
100 rows/request) rather than the `datasets` library — that library pulls in
pyarrow/pandas for a one-time fetch of a few hundred short sentences, which is a
real dependency for a script that runs once and is never imported by src/.

Usage:
    uv run python evals/fetch_wikiann.py [--per-lang 300]

See evals/datasets/README.md for the language choices, source, and license.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

DATASET = "unimelb-nlp/wikiann"
API = "https://datasets-server.huggingface.co/rows"
OUT = Path(__file__).resolve().parent / "datasets" / "wikiann"

# qortia/src/qortia/knowledge.py's _INDIC_MODEL routing (hi/bn/ta/te/mr -> the
# multilingual xx_ent_wiki_sm pipeline) + "en" (the non-Indic default path) +
# "de" as a control: a language qortia does *not* route anywhere special, to
# exercise the ner_lang_fallback_to_en path on genuinely non-English text
# rather than only ever testing the two designed-for routes.
LANGS = ("hi", "bn", "ta", "te", "mr", "en", "de")

# qortia.remember: MemoryItem.content rejects anything under 5 words. Filtering
# here means every example fetched is actually usable, not just downloaded.
_MIN_TOKENS = 5


def _fetch_page(lang: str, offset: int, length: int) -> list[dict]:  # type: ignore[type-arg]
    url = f"{API}?dataset={DATASET}&config={lang}&split=test&offset={offset}&length={length}"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - pinned https host
        body = json.loads(resp.read())
    if "error" in body:
        raise RuntimeError(f"{lang}: {body['error']}")
    return body["rows"]  # type: ignore[no-any-return]


def _fetch_lang(lang: str, target_n: int) -> list[dict]:  # type: ignore[type-arg]
    """Page through the test split until `target_n` examples with >=5 tokens
    and >=1 gold entity are collected (or the split runs out)."""
    examples = []
    offset = 0
    page = 100
    while len(examples) < target_n:
        rows = _fetch_page(lang, offset, page)
        if not rows:
            break
        for r in rows:
            row = r["row"]
            tokens: list[str] = row["tokens"]
            spans: list[str] = row["spans"]
            if len(tokens) < _MIN_TOKENS or not spans:
                continue
            gold = []
            for span in spans:
                etype, _, text = span.partition(": ")
                gold.append({"type": etype, "text": text})
            examples.append({"text": " ".join(tokens), "gold": gold})
            if len(examples) >= target_n:
                break
        offset += page
    return examples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=300)
    ap.add_argument("--force", action="store_true", help="refetch even if vendored")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        dest = OUT / f"{lang}.json"
        if dest.is_file() and not args.force:
            print(f"{lang}: already vendored ({dest}), skipping")
            continue
        t0 = time.perf_counter()
        examples = _fetch_lang(lang, args.per_lang)
        dest.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        n_gold = sum(len(e["gold"]) for e in examples)
        print(
            f"{lang}: {len(examples)} examples, {n_gold} gold entities "
            f"-> {dest} ({time.perf_counter() - t0:.1f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
