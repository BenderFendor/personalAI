"""Llama.cpp embeddings wrapper used for indexing and query embedding.

This module exposes a small class that calls llama.cpp server to produce embeddings.
It prefixes document vs query prompts as described in the RAG plan.

The implementation uses the OpenAI-compatible /v1/embeddings endpoint.
"""

from typing import List, Optional
import requests


class LlamaCppEmbeddingsWrapper:
    """Simple wrapper around the llama.cpp embeddings endpoint.

    Usage:
        emb = LlamaCppEmbeddingsWrapper(model="embeddinggemma", base_url="http://localhost:8080")
        vectors = emb.embed_documents(["doc1", "doc2"])
        qvec = emb.embed_query("some question")

    Notes:
    - Prefixes used: "Retrieval-document: " for documents and
      "Retrieval-query: " for queries (per the plan).
    - Optionally truncate returned vectors via `truncate_dim` for efficiency.
    - Requires llama.cpp server to be started with --embedding flag
    """

    def __init__(
        self,
        model: str = "embeddinggemma",
        base_url: str = "http://localhost:8080",
        batch_size: int = 8,
        truncate_dim: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim
        self.api_key = api_key or "not-needed"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _embed(self, prompt: str) -> List[float]:
        url = f"{self.base_url}/v1/embeddings"
        payload = {
            "model": self.model,
            "input": prompt,
        }

        response = requests.post(url, json=payload, headers=self._headers, timeout=60)
        response.raise_for_status()
        result = response.json()

        # OpenAI-compatible response format: {"data": [{"embedding": [...]}]}
        if isinstance(result, dict) and "data" in result:
            emb = result["data"][0].get("embedding")
        else:
            raise RuntimeError(
                f"Unexpected response from embeddings endpoint: {result}"
            )

        if emb is None:
            raise RuntimeError(f"No embedding returned from server: {result}")

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

    def _embed_batch(
        self, texts: List[str], for_query: bool = False
    ) -> List[List[float]]:
        out = []
        for t in texts:
            if for_query:
                p = f"Retrieval-query: {t}"
            else:
                p = f"Retrieval-document: {t}"
            out.append(self._embed(p))
        return out


# Backwards compatibility alias
OllamaEmbeddingsWrapper = LlamaCppEmbeddingsWrapper
