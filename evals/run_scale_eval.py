"""Qortia semantic recall, scale + context-economy eval on a real public corpus.

REH (run_reh.py) measures ranking robustness on a 55-case hand-curated dataset
whose corpora are small by construction. That's the wrong size to answer the
question an agent with a large memory actually has: once the corpus is bigger
than the recall budget, ranking stops being cosmetic and *becomes* context
management — it decides what the model sees and what it never learns exists.

So this harness uses a real, public, externally-judged corpus instead of
hand-curated cases:

    BEIR / FiQA-2018 — 57,638 user-written financial forum posts, 6,648
    queries, 1,706 human relevance judgements on the `test` split.
    https://github.com/beir-cellar/beir  (CC BY-SA 4.0)

Metrics, beyond the usual recall/MRR:

    precision_at_budget  fraction of returned characters that belong to a
                          judged relevant document — how much of the context
                          spend was earned rather than wasted
    chars_returned        absolute context cost of one /v1/recall call; lower
                          at equal recall is strictly better
    zero_result_rate      queries returning nothing — a context failure that
                          never shows up in a ranking metric

Standalone — no agnova import, no docker exec. Corpus seeding goes through
the real /v1/remember (batched, 500/request); provisioning goes through
/v1/admin/* (ADR-004, needs QORTIA_ADMIN_TOKEN) since eval-mode's simpler
seed-agent bypass issues no API key and /v1/remember needs real auth. Embed
drain-wait polls /v1/internal/eval/pending-embeddings (EVAL_MODE-gated) —
added specifically so this harness doesn't need a database connection.

Usage: bring up a live stack per this directory's README ("Live REH (dogfood)"
— db + ollama + a local `uvicorn` process with QORTIA_EVAL_MODE=true), then:

    QORTIA_URL=http://127.0.0.1:8090 \\
    QORTIA_ADMIN_TOKEN=<a value you set in the environment above> \\
    uv run python evals/run_scale_eval.py [--sizes 276,1000,10000] [--queries 100]

The corpus is vendored, git-tracked, at evals/datasets/fiqa/ (~47MB; see
evals/datasets/README.md) — a clone of this repo runs the eval with no
network access. Falls back to downloading into the gitignored evals/.cache/
if the vendored copy is ever missing.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import httpx

from evals.dataset_loader import QORTIA_URL

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
CACHE = Path(__file__).resolve().parent / ".cache"
VENDORED = Path(__file__).resolve().parent / "datasets" / "fiqa"

_BATCH = 500

# reflect.py: run_embedding_worker() sleeps 10s between batches of
# EMBEDDING_BATCH_SIZE=50. remember() returns as soon as the row is inserted
# with embedding=NULL — recall's ANN search cannot see it until the worker
# catches up.
_STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from had has have how i if in into is it
    its me my no not of on or should so than that the their them then there these they this to
    us was we what when where which who why will with would you your""".split()
)


# ── corpus ──────────────────────────────────────────────────────────────────


def _fetch() -> Path:
    """The vendored copy if present, else the download cache, else download fresh."""
    if (VENDORED / "corpus.jsonl").is_file():
        return VENDORED
    target = CACHE / "fiqa"
    if (target / "corpus.jsonl").is_file():
        return target
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"downloading {BEIR_URL} …", flush=True)
    with urllib.request.urlopen(BEIR_URL, timeout=300) as resp:  # noqa: S310 - pinned https host
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(CACHE)
    print(f"extracted to {target}", flush=True)
    return target


def _load(dataset: Path) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    corpus = {}
    for line in (dataset / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        text = f"{doc.get('title', '')} {doc['text']}".strip()
        corpus[doc["_id"]] = text
    queries = {}
    for line in (dataset / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        queries[q["_id"]] = q["text"]
    qrels: dict[str, set[str]] = {}
    with (dataset / "qrels" / "test.tsv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return corpus, queries, qrels


def _keywords(query: str, n: int) -> str:
    """The first `n` content words. qortia has no AND gate, so this only matters
    as a comparison point against the full natural-language question."""
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in _STOPWORDS and len(w) > 2]
    return " ".join(words[:n])


# ── HTTP clients ─────────────────────────────────────────────────────────────


class _Admin:
    def __init__(self, client: httpx.AsyncClient, admin_token: str) -> None:
        self.client = client
        self.headers = {"Authorization": f"Bearer {admin_token}"}

    async def provision_tenant(self, name: str) -> str:
        resp = await self.client.post(
            "/v1/admin/tenants", json={"name": name}, headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()["tenant_id"]  # type: ignore[no-any-return]

    async def provision_agent(self, tenant_id: str) -> tuple[str, str]:
        """Returns (agent_id, api_key) for a fresh agent under `tenant_id`."""
        agent = await self.client.post(
            "/v1/admin/agents", json={"tenant_id": tenant_id}, headers=self.headers
        )
        agent.raise_for_status()
        key = await self.client.post(
            "/v1/admin/keys", json={"tenant_id": tenant_id}, headers=self.headers
        )
        key.raise_for_status()
        return agent.json()["agent_id"], key.json()["api_key"]


class _AgentClient:
    def __init__(self, client: httpx.AsyncClient, api_key: str, agent_id: str) -> None:
        self.client = client
        self.headers = {"Authorization": f"Bearer {api_key}", "X-Agent-Id": agent_id}
        self.agent_id = agent_id

    async def remember_batch(self, items: list[tuple[str, str]]) -> dict[str, str]:
        """items = [(fiqa_corpus_id, content), ...]. Returns {qortia_id: fiqa_corpus_id}."""
        memories = [{"type": "episodic", "content": text[:8000]} for _, text in items]
        resp = await self.client.post(
            "/v1/remember", json={"memories": memories}, headers=self.headers
        )
        resp.raise_for_status()
        ids = resp.json()["ids"]
        return dict(zip(ids, (cid for cid, _ in items), strict=True))

    async def recall(self, query: str) -> list[dict[str, Any]]:
        resp = await self.client.post(
            "/v1/recall", json={"query": query, "scope": "all"}, headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()["results"]  # type: ignore[no-any-return]

    async def pending_embeddings(self) -> int:
        resp = await self.client.get(
            "/v1/internal/eval/pending-embeddings", params={"agent_id": self.agent_id}
        )
        resp.raise_for_status()
        return resp.json()["pending"]  # type: ignore[no-any-return]


async def _wait_for_embeddings(client: _AgentClient, *, timeout: float = 900.0) -> float:
    t0 = time.perf_counter()
    while True:
        pending = await client.pending_embeddings()
        elapsed = time.perf_counter() - t0
        if pending == 0:
            return elapsed
        if elapsed > timeout:
            raise TimeoutError(f"{pending} embeddings still pending after {timeout:.0f}s")
        time.sleep(3)


async def _seed(client: _AgentClient, docs: dict[str, str]) -> dict[str, str]:
    """Batched remember(); returns {qortia_id: fiqa_corpus_id} for scoring.

    qortia.models.MemoryItem.content_not_empty rejects anything under 5 words,
    422ing the *whole* batch it's in — real FiQA content hits this: short forum
    replies like "Yes, that's correct." (3 words) are ordinary content an
    agent's episodic memory would legitimately try to store. Filtered here
    since there's no honest way to satisfy the validator short of fabricating
    words that were never in the source text.
    """
    fitted = {i: t for i, t in docs.items() if len(t.split()) >= 5}
    skipped = len(docs) - len(fitted)
    if skipped:
        print(f"  skipping {skipped} doc(s) under qortia's 5-word content floor", file=sys.stderr)
    id_map: dict[str, str] = {}
    items = list(fitted.items())
    for i in range(0, len(items), _BATCH):
        id_map.update(await client.remember_batch(items[i : i + _BATCH]))
    return id_map


def _score(
    results_by_query: list[tuple[list[dict[str, Any]], float]],
    probes: list[tuple[str, set[str]]],
    id_map: dict[str, str],
) -> dict[str, float]:
    recall_5, rr, prec_budget, chars, zeros = [], [], [], [], 0
    for (results, _elapsed), (_query, relevant) in zip(results_by_query, probes, strict=True):
        fiqa_ids = [id_map.get(r["id"], r["id"]) for r in results]
        if not fiqa_ids:
            zeros += 1
        recall_5.append(1.0 if any(i in relevant for i in fiqa_ids[:5]) else 0.0)
        rank = next((n for n, i in enumerate(fiqa_ids, 1) if i in relevant), None)
        rr.append(1.0 / rank if rank else 0.0)
        total = sum(len(r["content"]) for r in results)
        earned = sum(
            len(r["content"]) for r, fid in zip(results, fiqa_ids, strict=True) if fid in relevant
        )
        chars.append(total)
        prec_budget.append(earned / total if total else 0.0)
    return {
        "recall_at_5": statistics.mean(recall_5),
        "mrr": statistics.mean(rr),
        "precision_at_budget": statistics.mean(prec_budget),
        "chars_returned": statistics.mean(chars),
        "zero_result_rate": zeros / len(probes),
        "ms_per_query": statistics.mean(e for _, e in results_by_query) * 1000,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    corpus, queries, qrels = _load(_fetch())
    rng = random.Random(args.seed)  # noqa: S311 — reproducible query sampling, not crypto
    judged = sorted(q for q, rel in qrels.items() if rel & corpus.keys())
    sampled = rng.sample(judged, min(args.queries, len(judged)))
    must_keep = {d for q in sampled for d in qrels[q] if d in corpus}
    filler = [d for d in sorted(corpus) if d not in must_keep]
    rng.shuffle(filler)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=120.0) as http_client:
        admin = _Admin(http_client, args.admin_token)
        tenant_id = await admin.provision_tenant("scale-eval")
        print(f"tenant {tenant_id}")

        print(
            f"\nFiQA-2018 · qortia semantic recall · {len(sampled)} judged queries · "
            f"{len(must_keep):,} relevant docs always present\n"
        )
        header = (
            f"{'size':>7} {'query':>10} {'R@5':>6} {'MRR':>6} {'prec@bud':>9} "
            f"{'chars':>7} {'zero%':>6} {'ms/q':>8}"
        )
        print(header)
        print("-" * len(header))

        report: dict[str, Any] = {"dataset": "beir/fiqa-2018", "backend": "qortia", "runs": []}
        for size in [int(s) for s in args.sizes.split(",")]:
            ids = list(must_keep) + filler[: max(0, size - len(must_keep))]
            docs = {i: corpus[i] for i in ids}

            agent_id, api_key = await admin.provision_agent(tenant_id)
            client = _AgentClient(http_client, api_key, agent_id)

            t0 = time.perf_counter()
            id_map = await _seed(client, docs)
            write_s = time.perf_counter() - t0
            print(
                f"  wrote {len(docs):,} docs in {write_s:.0f}s ({len(docs) / write_s:.0f}/s)",
                file=sys.stderr,
            )

            embed_s = await _wait_for_embeddings(client)
            print(f"  embeddings drained after {embed_s:.0f}s", file=sys.stderr)
            seed_s = write_s + embed_s

            query_style = "keywords" if args.terms else "full-question"
            if args.terms:
                probes = [(_keywords(queries[q], args.terms), qrels[q]) for q in sampled]
            else:
                probes = [(queries[q], qrels[q]) for q in sampled]

            results_by_query = []
            for query, _relevant in probes:
                t0 = time.perf_counter()
                results = await client.recall(query)
                results_by_query.append((results, time.perf_counter() - t0))

            m = _score(results_by_query, probes, id_map)
            print(
                f"{len(docs):>7,} {query_style:>10} {m['recall_at_5']:>6.3f} {m['mrr']:>6.3f} "
                f"{m['precision_at_budget']:>9.3f} {m['chars_returned']:>7.0f} "
                f"{m['zero_result_rate'] * 100:>5.0f}% {m['ms_per_query']:>7.0f}ms"
            )
            report["runs"].append(
                {"size": len(docs), "query_style": query_style, "seed_seconds": seed_s, **m}
            )

        return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="276,1000,10000")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--terms", type=int, default=0, help="0 = full natural-language query")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--base-url", default=QORTIA_URL)
    ap.add_argument("--admin-token", default=os.environ.get("QORTIA_ADMIN_TOKEN", ""))
    args = ap.parse_args()
    if not args.admin_token:
        print("QORTIA_ADMIN_TOKEN is required (env or --admin-token)", file=sys.stderr)
        return 2

    import asyncio

    report = asyncio.run(_run(args))

    out = Path(__file__).resolve().parent / "results" / "scale_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
