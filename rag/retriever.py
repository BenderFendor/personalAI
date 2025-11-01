"""RAG orchestration: indexing, retrieval, and response generation.

This module wires the `OllamaEmbeddingsWrapper` and `ChromaVectorStore` to
provide a convenient `RAGRetriever` class that can index texts and run
retrieval + generation flows.
"""
from typing import List, Dict, Optional
import math
import logging

from .embeddings import OllamaEmbeddingsWrapper
from .vector_store import ChromaVectorStore

try:
    import ollama
except Exception:
    ollama = None

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base.

INSTRUCTIONS:
1. Always base your answers on the provided context from the knowledge base
2. If the context doesn't contain enough information, say so clearly
3. Cite the source documents when possible
4. If you need more information, say that a web search would be helpful
"""


class RAGRetriever:
    """Combine embeddings and vector store to index and retrieve documents.

    Example:
        emb = OllamaEmbeddingsWrapper()
        store = ChromaVectorStore()
        retriever = RAGRetriever(emb, store)
        retriever.index_texts(["doc1", "doc2"])  # simple indexing
        docs = retriever.retrieve("some question", top_k=3)
        answer = retriever.generate_rag_response("some question", docs)
    """

    def __init__(
        self,
        embeddings: OllamaEmbeddingsWrapper,
        vector_store: ChromaVectorStore,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.emb = embeddings
        self.store = vector_store
        self.system_prompt = system_prompt

    def index_texts(
        self,
        texts: List[str],
        ids_prefix: str = "doc",
        batch_size: int = 100,
        metadatas: Optional[List[dict]] = None,
    ):
        """Index a list of raw texts into the vector store.

        This method will embed texts using the document prompt and add them in
        batches to ChromaDB.
        """
        total = len(texts)
        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            try:
                embeddings = self.emb.embed_documents(batch_texts)
            except Exception as e:
                logger.exception("Failed to embed batch: %s", e)
                raise

            ids = [f"{ids_prefix}_{j}" for j in range(i, i + len(batch_texts))]
            batch_metas = None
            if metadatas:
                batch_metas = metadatas[i : i + batch_size]

            self.store.add_documents(ids=ids, embeddings=embeddings, documents=batch_texts, metadatas=batch_metas)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve the top_k most relevant documents for the query.

        Returns a list of dicts: { 'content', 'similarity', 'metadata' }
        """
        qvec = self.emb.embed_query(query)
        results = self.store.query(query_embeddings=[qvec], n_results=top_k)

        # Chroma responds with structure: documents, distances, metadatas
        docs = []
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc, dist, meta in zip(documents, distances, metadatas):
            # convert cosine distance -> similarity
            try:
                similarity = 1.0 - float(dist)
            except Exception:
                similarity = float(dist)
            docs.append({"content": doc, "similarity": similarity, "metadata": meta})

        return docs

    def generate_rag_response(
        self,
        user_query: str,
        retrieved_docs: List[Dict],
        llm_model: str = "qwen3",
        temperature: float = 0.0,
    ) -> str:
        """Produce a response grounded on retrieved_docs using Ollama chat.

        If `ollama` is not available the method will raise an ImportError.
        """
        if ollama is None:
            raise ImportError("The `ollama` package is required to generate RAG responses. Install and configure Ollama.")

        context = ""
        for i, d in enumerate(retrieved_docs, start=1):
            context += f"\n[Source {i}] (Similarity: {d.get('similarity', 0):.2f})\n{d.get('content')}\n"

        prompt = f"Context from knowledge base:\n{context}\nUser Question: {user_query}\n\nPlease provide a detailed answer based on the context above. If the context doesn't contain sufficient information, say so and suggest what additional information would be helpful."

        response = ollama.chat(
            model=llm_model,
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
            temperature=temperature,
        )

        # Ollama API typically returns a dict with a 'message' key containing another dict with 'content'.
        # Sometimes, it may return a dict with a 'choices' list, where each item may have 'message' or 'text'.
        # This logic attempts to handle both formats for compatibility.
        if isinstance(response, dict):
            message = response.get("message") or response.get("choices")
            # try common shapes
            if isinstance(message, dict):
                return message.get("content") or message.get("text") or ""
            if isinstance(message, list) and message:
                # sometimes choices list
                first = message[0]
                return first.get("message", {}).get("content") or first.get("text") or ""

        # fallback: convert to string
        return str(response)
