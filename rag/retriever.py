"""RAG orchestration: indexing, retrieval, and response generation.

This module wires the `LlamaCppEmbeddingsWrapper` and `ChromaVectorStore` to
provide a convenient `RAGRetriever` class that can index texts and run
retrieval + generation flows.
"""

from typing import List, Dict, Optional
import logging

from .embeddings import LlamaCppEmbeddingsWrapper
from .vector_store import ChromaVectorStore

try:
    from utils.llama_cpp_provider import LlamaCppProvider
except Exception:
    LlamaCppProvider = None

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
        emb = LlamaCppEmbeddingsWrapper()
        store = ChromaVectorStore()
        retriever = RAGRetriever(emb, store)
        retriever.index_texts(["doc1", "doc2"])  # simple indexing
        docs = retriever.retrieve("some question", top_k=3)
        answer = retriever.generate_rag_response("some question", docs)
    """

    def __init__(
        self,
        embeddings: LlamaCppEmbeddingsWrapper,
        vector_store: ChromaVectorStore,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        llama_cpp_base_url: str = "http://localhost:8080",
    ):
        self.emb = embeddings
        self.store = vector_store
        self.system_prompt = system_prompt
        self.llama_cpp_base_url = llama_cpp_base_url
        self._llm_provider: Optional[LlamaCppProvider] = None

    def _get_llm_provider(self, model: str) -> LlamaCppProvider:
        """Get or create the LLM provider."""
        if self._llm_provider is None:
            if LlamaCppProvider is None:
                raise ImportError("llama.cpp provider is not available.")
            self._llm_provider = LlamaCppProvider(
                model=model,
                base_url=self.llama_cpp_base_url,
                temperature=0.0,
            )
        return self._llm_provider

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

            self.store.add_documents(
                ids=ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metas,
            )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve the top_k most relevant documents for the query.

        Returns a list of dicts: { 'content', 'similarity', 'metadata' }
        """
        qvec = self.emb.embed_query(query)
        results = self.store.query(query_embeddings=[qvec], n_results=top_k)

        if not results:
            return []

        # Chroma responds with structure: documents, distances, metadatas
        docs = []
        documents = (
            results.get("documents", [[]])[0] if results.get("documents") else []
        )
        distances = (
            results.get("distances", [[]])[0] if results.get("distances") else []
        )
        metadatas = (
            results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        )

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
        """Produce a response grounded on retrieved_docs using llama.cpp chat.

        If llama.cpp provider is not available the method will raise an ImportError.
        """
        if LlamaCppProvider is None:
            raise ImportError(
                "The llama.cpp provider is required to generate RAG responses. Ensure utils.llama_cpp_provider is available."
            )

        context = ""
        for i, d in enumerate(retrieved_docs, start=1):
            context += f"\n[Source {i}] (Similarity: {d.get('similarity', 0):.2f})\n{d.get('content')}\n"

        prompt = f"Context from knowledge base:\n{context}\nUser Question: {user_query}\n\nPlease provide a detailed answer based on the context above. If the context doesn't contain sufficient information, say so and suggest what additional information would be helpful."

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        provider = self._get_llm_provider(llm_model)
        # Override temperature
        provider.temperature = temperature

        response = provider.chat(messages=messages, stream=False)

        # OpenAI-compatible response format: {"choices": [{"message": {"content": "..."}}]}
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    return message.get("content", "")

        # fallback: convert to string
        return str(response)


# Backwards compatibility alias
OllamaEmbeddingsWrapper = LlamaCppEmbeddingsWrapper
