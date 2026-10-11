"""Hybrid retrieval: FAISS dense search + BM25, fused with Reciprocal Rank Fusion.

The chapter filter is applied to both candidate lists before fusion, so a
filtered search never returns a chunk from the wrong chapter.
"""
import json
import os
import re
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi

INDEX_DIR = Path(__file__).resolve().parents[1] / "index"
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOKEN.findall(text.lower())


class HybridRetriever:
    def __init__(self, index_dir=INDEX_DIR, embedder=None):
        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.chunks = json.loads((index_dir / "chunks.json").read_text())
        self.manifest = json.loads((index_dir / "manifest.json").read_text())
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        if embedder is None:
            from .embeddings import VertexEmbedder
            embedder = VertexEmbedder(model=self.manifest["embed_model"], dim=self.manifest["dim"])
        self.embedder = embedder

    def chapters(self):
        return sorted({c["chapter"] for c in self.chunks})

    def search(self, query, k=4, chapter=None, candidates=40, rrf_k=60, mode="hybrid", bm25_weight=None):
        """mode: "hybrid" (default), "dense" or "bm25". The last two exist for eval ablations.

        bm25_weight scales BM25's vote in the fusion (dense always counts 1.0).
        Below 1.0, keyword matches can break ties but can't outvote a strong semantic match.
        """
        if bm25_weight is None:
            bm25_weight = float(os.getenv("BM25_WEIGHT", "1.0"))
        allowed = (lambda i: self.chunks[i]["chapter"] == chapter) if chapter else (lambda i: True)
        # With a filter, search everything so the filter can't starve the candidate list.
        n_dense = self.index.ntotal if chapter else min(candidates, self.index.ntotal)

        qvec = self.embedder.embed([query], task_type="RETRIEVAL_QUERY")
        _, ids = self.index.search(qvec, n_dense)
        dense = [int(i) for i in ids[0] if i >= 0 and allowed(int(i))][:candidates]

        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sparse = [i for i in ranked if scores[i] > 0 and allowed(i)][:candidates]

        rankings = {
            "hybrid": ((dense, 1.0), (sparse, bm25_weight)),
            "dense": ((dense, 1.0),),
            "bm25": ((sparse, 1.0),),
        }[mode]
        fused = {}
        for ranking, weight in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + weight / (rrf_k + rank + 1)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [dict(self.chunks[i], score=round(fused[i], 5)) for i in top]
