"""Chunk crawled pages, embed with Vertex AI, and write the FAISS index.

    python -m scripts.build_index --pages data/pages.jsonl

Writes helpdesk_agent/index/{index.faiss, chunks.json, manifest.json}.
The index ships inside the agent package, so Agent Engine deploys it too.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "helpdesk_agent" / ".env")

from helpdesk_agent.rag.chunking import chunk_page  # noqa: E402
from helpdesk_agent.rag.embeddings import VertexEmbedder  # noqa: E402
from helpdesk_agent.rag.retriever import INDEX_DIR  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default="data/pages.jsonl")
    parser.add_argument("--max-words", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=40)
    args = parser.parse_args()

    pages = [json.loads(line) for line in Path(args.pages).read_text().splitlines() if line.strip()]
    chunks = [c for p in pages for c in chunk_page(p, args.max_words, args.overlap)]
    print(f"{len(pages)} pages -> {len(chunks)} chunks")

    embedder = VertexEmbedder()
    start = time.time()
    vectors = embedder.embed([c["text"] for c in chunks], task_type="RETRIEVAL_DOCUMENT")
    print(f"embedded in {time.time() - start:.0f}s")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks))
    (INDEX_DIR / "manifest.json").write_text(json.dumps({
        "embed_model": embedder.model,
        "dim": embedder.dim,
        "n_pages": len(pages),
        "n_chunks": len(chunks),
        "chunking": {"max_words": args.max_words, "overlap": args.overlap},
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    print(f"index written to {INDEX_DIR}")
