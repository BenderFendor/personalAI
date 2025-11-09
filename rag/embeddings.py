"""Ollama embeddings wrapper used for indexing and query embedding.

This module exposes a small class that calls Ollama to produce embeddings.
It prefixes document vs query prompts as described in the RAG plan.

The implementation is defensive: if `ollama` is not installed, an ImportError
with an actionable message will be raised.
"""
from typing import List, Optional
import os

try:
    import ollama
except Exception as e:
    ollama = None  # will raise later with helpful message


class OllamaEmbeddingsWrapper:
    """Simple wrapper around the Ollama embeddings endpoint.

    Usage:
        emb = OllamaEmbeddingsWrapper(model="embeddinggemma")
        vectors = emb.embed_documents(["doc1", "doc2"])
        qvec = emb.embed_query("some question")

    Notes:
    - Prefixes used: "Retrieval-document: " for documents and
      "Retrieval-query: " for queries (per the plan).
    - Optionally truncate returned vectors via `truncate_dim` for efficiency.
    """

    def __init__(
        self,
        model: str = "embeddinggemma",
        base_url: Optional[str] = None,
        batch_size: int = 8,
        truncate_dim: Optional[int] = None,
    ):
        if ollama is None:
            raise ImportError("The `ollama` package is required for OllamaEmbeddingsWrapper. Install it and ensure it's importable.")

        # If the client supports configuring a base url via env var, set it.
        if base_url:
            os.environ.setdefault("OLLAMA_BASE_URL", base_url)

        self.model = model
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim

    def _embed(self, prompt: str) -> List[float]:
        # Ollama's python client typically exposes `embeddings` function.
        # We call it and return the `embedding` field.
        resp = ollama.embeddings(model=self.model, prompt=prompt)
        
        # Handle both dict and object responses
        if isinstance(resp, dict):
            emb = resp.get("embedding")
        else:
            # Response is likely an object with attributes
            emb = getattr(resp, "embedding", None)
        
        if emb is None:
            # try to be resilient to different client shapes
            raise RuntimeError(f"Unexpected response from ollama.embeddings: {resp}")

        if self.truncate_dim:
            return emb[: self.truncate_dim]
        return emb

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts using document-specific prompt.

        Returns a list of embedding vectors (one per input text).
        """
        embeddings = []
        batch = []
        for t in texts:
            batch.append(t)
            if len(batch) >= self.batch_size:
                embeddings.extend(self._embed_batch(batch, for_query=False))
                batch = []

        if batch:
            embeddings.extend(self._embed_batch(batch, for_query=False))

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string using query-specific prompt."""
        prompt = f"Retrieval-query: {text}"
        return self._embed(prompt)

    def _embed_batch(self, texts: List[str], for_query: bool = False) -> List[List[float]]:
        out = []
        for t in texts:
            if for_query:
                p = f"Retrieval-query: {t}"
            else:
                p = f"Retrieval-document: {t}"
            out.append(self._embed(p))
        return out
