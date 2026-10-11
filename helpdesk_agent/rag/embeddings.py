"""Vertex AI text embeddings through the google-genai SDK.

Reads GOOGLE_GENAI_USE_VERTEXAI and GOOGLE_CLOUD_PROJECT from the environment.
Vectors are L2-normalized so FAISS inner product = cosine.

Embeddings use their own region (EMBED_LOCATION, default us-central1) instead of
GOOGLE_CLOUD_LOCATION. Measured on this project: the global endpoint took ~10.4s
for the first request after ~15s idle, while us-central1 took ~0.3s. Gemini chat
models stay on global because some newer models are only served there.
"""
import logging
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
        self.client = make_embed_client(genai)

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
                except Exception as exc:
                    if attempt == 3:
                        raise
                    logging.getLogger("helpdesk_agent").warning(
                        "embedding retry %d after %s", attempt + 1, str(exc)[:120])
                    time.sleep(2 ** attempt)  # back off on quota errors
        return normalize(np.array(vectors, dtype="float32"))


def make_embed_client(genai):
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
    if not use_vertex:
        return genai.Client()  # Gemini API key mode: no regions to choose
    return genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.getenv("EMBED_LOCATION", "us-central1"),
    )


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
