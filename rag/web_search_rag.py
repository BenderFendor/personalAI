"""Web search RAG integration: fetch pages, chunk them, and index as vectors.

This module provides a helper that:
1. Takes web search results (URLs + snippets)
2. Fetches full page content
3. Chunks the content using LangChain text splitters
4. Indexes chunks into the vector store for RAG retrieval

This enables using web search results as dynamic knowledge base entries.
"""
from typing import List, Dict, Optional
import logging

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

from .retriever import RAGRetriever

logger = logging.getLogger(__name__)


class WebSearchRAG:
    """Integrate web search results with RAG indexing.
    
    This class takes search results (URLs + content), chunks them,
    and indexes them into the vector store so they can be retrieved
    during subsequent queries.
    
    Example:
        web_rag = WebSearchRAG(retriever)
        search_results = [{'url': '...', 'content': '...', 'title': '...'}, ...]
        web_rag.index_search_results(search_results)
        # Now RAG retrieval will include these web pages
    """
    
    def __init__(
        self, 
        retriever: RAGRetriever,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        auto_index: bool = True
    ):
        """Initialize web search RAG integration.
        
        Args:
            retriever: RAGRetriever instance to use for indexing
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks
            auto_index: Whether to auto-index search results
        """
        self.retriever = retriever
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.auto_index = auto_index

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size to avoid infinite loop in chunking.")

        if RecursiveCharacterTextSplitter is None:
            logger.warning("LangChain not available - using simple chunking fallback")
            self.text_splitter = None
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if self.text_splitter:
            # Use LangChain splitter
            return self.text_splitter.split_text(text)
        else:
            # Fallback: simple chunking
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk:
                    chunks.append(chunk)
            return chunks
    
    def index_search_results(
        self, 
        search_results: List[Dict[str, str]],
        collection_prefix: str = "web"
    ) -> int:
        """Index web search results into the vector store.
        
        Args:
            search_results: List of dicts with 'url', 'content', 'title' keys
            collection_prefix: Prefix for document IDs
            
        Returns:
            Number of chunks indexed
        """
        all_texts = []
        all_metadatas = []
        
        for result in search_results:
            url = result.get('url', 'unknown')
            title = result.get('title', 'Untitled')
            content = result.get('content', '')
            
            if not content:
                continue
            
            # Chunk the content
            chunks = self._chunk_text(content)
            
            # Create metadata for each chunk
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metadatas.append({
                    'source': url,
                    'title': title,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'type': 'web_search'
                })
        
        if not all_texts:
            logger.warning("No text content to index from search results")
            return 0
        
        # Index using the retriever
        try:
            self.retriever.index_texts(
                texts=all_texts,
                ids_prefix=collection_prefix,
                metadatas=all_metadatas
            )
            logger.info(f"Indexed {len(all_texts)} chunks from {len(search_results)} web pages")
            return len(all_texts)
        except Exception as e:
            logger.exception(f"Error indexing search results: {e}")
            raise
    
    def index_single_page(
        self,
        url: str,
        content: str,
        title: Optional[str] = None
    ) -> int:
        """Index a single web page.
        
        Args:
            url: Page URL
            content: Page content
            title: Optional page title
            
        Returns:
            Number of chunks indexed
        """
        return self.index_search_results([{
            'url': url,
            'content': content,
            'title': title or url
        }])
