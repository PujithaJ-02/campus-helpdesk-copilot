"""Vertex AI text embeddings through the google-genai SDK.

Reads GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
from the environment. Vectors are L2-normalized so FAISS inner product = cosine.
"""
import os
import time

import numpy as np


class VertexEmbedder:
    def __init__(self, model=None, dim=None, batch_size=None):
        from google import genai  # imported here so tests don't need credentials

        self.model = model or os.getenv("EMBED_MODEL", "gemini-embedding-001")
        self.dim = int(dim or os.getenv("EMBED_DIM", "768"))
        # Some Vertex embedding models accept one input per request; raise this if yours allows more.
        self.batch_size = int(batch_size or os.getenv("EMBED_BATCH_SIZE", "1"))
        self.client = genai.Client()

    def embed(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        from google.genai import types

        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            for attempt in range(4):
                try:
                    resp = self.client.models.embed_content(model=self.model, contents=batch, config=config)
                    vectors.extend(e.values for e in resp.embeddings)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)  # back off on quota errors
        return normalize(np.array(vectors, dtype="float32"))


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
