import hashlib
import json

import faiss
import numpy as np
import pytest

from helpdesk_agent.rag.embeddings import normalize
from helpdesk_agent.rag.retriever import tokenize


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder, so tests need no cloud access."""
    dim = 64
    model = "fake"

    def embed(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                out[row, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        return normalize(out)


CHUNKS = [
    {"id": "a", "url": "https://adminguide.stanford.edu/chapters/computing/information-security-incidents",
     "chapter": "computing", "title": "Information Security Incidents", "section": "Reporting",
     "text": "Information Security Incidents > Reporting\nReport a lost or stolen laptop with restricted data immediately to the security office."},
    {"id": "b", "url": "https://adminguide.stanford.edu/chapters/computing/identification-and-authentication",
     "chapter": "computing", "title": "Identification and Authentication", "section": "Passwords",
     "text": "Identification and Authentication > Passwords\nNever share your SUNet ID password with anyone, including assistants."},
    {"id": "c", "url": "https://adminguide.stanford.edu/chapters/purchasing-and-payments/business-and-travel-expenses/travel-cards",
     "chapter": "purchasing-and-payments", "title": "Travel Cards", "section": "Use",
     "text": "Travel Cards > Use\nThe travel card is for business travel only. Personal charges on the travel card are not allowed. Contact jane@stanford.edu."},
]


@pytest.fixture
def index_dir(tmp_path):
    emb = FakeEmbedder()
    vecs = emb.embed([c["text"] for c in CHUNKS])
    index = faiss.IndexFlatIP(emb.dim)
    index.add(vecs)
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    (tmp_path / "chunks.json").write_text(json.dumps(CHUNKS))
    (tmp_path / "manifest.json").write_text(json.dumps({"embed_model": "fake", "dim": emb.dim}))
    return tmp_path


@pytest.fixture
def retriever(index_dir):
    from helpdesk_agent.rag.retriever import HybridRetriever
    return HybridRetriever(index_dir=index_dir, embedder=FakeEmbedder())
