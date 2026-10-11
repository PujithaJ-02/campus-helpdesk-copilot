"""Retrieval eval: does the right policy page come back in the top k?

    python -m scripts.eval_retrieval --k 4

Compares BM25-only, dense-only and hybrid on the golden set. A hit means a
retrieved chunk's url starts with the expected url (so sub-pages count).
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "helpdesk_agent" / ".env")

from helpdesk_agent.rag.retriever import HybridRetriever  # noqa: E402


def is_match(url, expected):
    """Same page, a sub-page of it, or the same policy under the site's alias path."""
    url, expected = url.rstrip("/"), expected.rstrip("/")
    return url.startswith(expected) or url.rsplit("/", 1)[-1] == expected.rsplit("/", 1)[-1]


def first_hit_rank(results, expected):
    for rank, r in enumerate(results, start=1):
        if is_match(r["url"], expected):
            return rank
    return None


def evaluate(retriever, golden, k, mode, use_filter, bm25_weight=1.0):
    hits, rr, latencies, misses = 0, [], [], []
    for item in golden:
        start = time.perf_counter()
        results = retriever.search(item["question"], k=k, mode=mode, bm25_weight=bm25_weight,
                                   chapter=item.get("chapter") if use_filter else None)
        latencies.append((time.perf_counter() - start) * 1000)
        rank = first_hit_rank(results, item["expected_url"])
        hits += rank is not None
        rr.append(1 / rank if rank else 0.0)
        if rank is None:
            misses.append(item["question"])
    n = len(golden)
    return {
        "mode": mode if mode != "hybrid" else f"hybrid(bm25={bm25_weight})",
        "chapter_filter": use_filter,
        f"hit@{k}": round(hits / n, 3),
        "mrr": round(sum(rr) / n, 3),
        "latency_ms_p50": round(statistics.median(latencies)),
        "misses": misses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/policy_golden.jsonl")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--bm25-weights", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    args = parser.parse_args()

    golden = [json.loads(line) for line in Path(args.golden).read_text().splitlines() if line.strip()]
    retriever = HybridRetriever()
    report = {"n": len(golden), "index": retriever.manifest, "runs": []}
    report["runs"].append(evaluate(retriever, golden, args.k, "bm25", use_filter=False))
    report["runs"].append(evaluate(retriever, golden, args.k, "dense", use_filter=False))
    for w in args.bm25_weights:
        report["runs"].append(evaluate(retriever, golden, args.k, "hybrid", False, bm25_weight=w))
    report["runs"].append(evaluate(retriever, golden, args.k, "dense", use_filter=True))

    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/retrieval.json").write_text(json.dumps(report, indent=2))
    for run in report["runs"]:
        print({k: v for k, v in run.items() if k != "misses"})
