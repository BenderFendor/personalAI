"""RAG package: simple wrappers for embeddings, vector store and retrieval orchestration.

This package provides lightweight, well-documented helpers to index documents
with Ollama embeddings and store/query them from ChromaDB.

Note: This is intentionally minimal. It expects `ollama` and `chromadb` to be
available in the environment. The higher-level integration with your chatbot
should call these helpers (see `retriever.py`).
"""

from .embeddings import OllamaEmbeddingsWrapper
from .vector_store import ChromaVectorStore
from .retriever import RAGRetriever
from .web_search_rag import WebSearchRAG

__all__ = [
    "OllamaEmbeddingsWrapper",
    "ChromaVectorStore",
    "RAGRetriever",
    "WebSearchRAG",
]
