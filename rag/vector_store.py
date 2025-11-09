"""ChromaDB vector store helper.

Provides a small wrapper around ChromaDB PersistentClient and a single
collection to keep calling code simple and focused.
"""
from typing import List, Optional

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None


class ChromaVectorStore:
    """Wrapper for ChromaDB PersistentClient and collection management."""

    def __init__(
        self,
        path: str = "./chroma_db",
        collection_name: str = "rag_documents",
        hnsw_space: str = "cosine",
        allow_reset: bool = True,
    ):
        if chromadb is None:
            raise ImportError("The `chromadb` package is required for ChromaVectorStore. Install it and ensure it's importable.")

        # Settings allow tuning; keep defaults conservative.
        settings = Settings(anonymized_telemetry=False, allow_reset=allow_reset)
        # PersistentClient stores DB on disk for persistence
        self.client = chromadb.PersistentClient(path=path, settings=settings)
        self.collection_name = collection_name

        metadata = {
            "hnsw:space": hnsw_space,
            # sensible defaults that can be tuned later
            "hnsw:construction_ef": 200,
            "hnsw:M": 16,
        }

        self.collection = self.client.get_or_create_collection(name=collection_name, metadata=metadata)

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
    ):
        """Add or update a batch of documents to the collection."""
        kwargs = {
            "ids": ids,
            "embeddings": embeddings,
            "documents": documents,
        }
        if metadatas is not None:
            kwargs["metadatas"] = metadatas

        # Use upsert to avoid duplicate ID warnings
        self.collection.upsert(**kwargs)

    def query(self, query_embeddings: List[List[float]], n_results: int = 5):
        """Query the collection by embedding(s). Returns raw chroma results dict."""
        return self.collection.query(query_embeddings=query_embeddings, n_results=n_results)

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            # Fallback: attempt to infer from docs
            try:
                docs = self.collection.get()
                return len(docs.get("documents", []))
            except Exception:
                return 0

    def reset(self):
        """Remove and recreate collection (use with care)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
