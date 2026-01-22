"""Vector similarity search implementation."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from contextllm.ingestion.storage import VectorStore
from contextllm.ingestion.embedder import generate_embedding
from contextllm.retrieval.query import embed_query, preprocess_query
from contextllm.utils.config import get_config

logger = logging.getLogger(__name__)


class ChunkSearcher:
    """Searcher for retrieving relevant chunks using vector similarity."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize chunk searcher.
        
        Args:
            vector_store: VectorStore instance (created if None)
        """
        self.vector_store = vector_store or VectorStore()
        self.config = get_config()
        logger.info("Chunk searcher initialized")
    
    def search(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using vector similarity.
        
        Args:
            query_text: Query text string
            top_k: Number of results to return (uses config if None)
            filter_metadata: Optional metadata filters for ChromaDB
            
        Returns:
            List of result dictionaries with 'chunk_id', 'text', 'metadata', 'similarity_score'
        """
        if not query_text or not query_text.strip():
            logger.warning("Empty query provided")
            return []
        
        # Preprocess query
        processed_query = preprocess_query(query_text)
        
        # Get top_k from config if not provided
        if top_k is None:
            top_k = self.config.get("retrieval.top_k", 50)
        
        try:
            logger.info(f"Searching for top {top_k} chunks with query: {processed_query[:100]}...")
            
            # Generate query embedding
            query_embedding = embed_query(processed_query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            
            # Format results
            formatted_results = self._format_results(results, query_text=processed_query)
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            raise
    
    def _format_results(
        self,
        results: List[Dict[str, Any]],
        query_text: str
    ) -> List[Dict[str, Any]]:
        """
        Format search results with additional metadata.
        
        Args:
            results: Raw results from vector store
            query_text: Original query text
            
        Returns:
            Formatted result dictionaries
        """
        formatted = []
        
        for result in results:
            # Extract similarity score (convert distance to similarity if needed)
            distance = result.get('distance')
            score = result.get('score')
            
            # If we have distance but not score, convert distance to similarity
            # ChromaDB uses cosine distance (0 = identical, 1 = orthogonal)
            # Similarity = 1 - distance
            if score is None and distance is not None:
                score = 1 - distance
            
            # Ensure score is between 0 and 1
            if score is not None:
                score = max(0.0, min(1.0, float(score)))
            
            formatted_result = {
                'chunk_id': result.get('id', ''),
                'text': result.get('text', ''),
                'metadata': result.get('metadata', {}),
                'similarity_score': score if score is not None else 0.0,
                'distance': distance,
                'query': query_text
            }
            
            formatted.append(formatted_result)
        
        # Sort by similarity score (descending)
        formatted.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return formatted
    
    def search_by_document(
        self,
        query_text: str,
        document_id: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks within a specific document.
        
        Args:
            query_text: Query text string
            document_id: Document ID to filter by
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        filter_metadata = {'source': document_id}
        return self.search(query_text, top_k=top_k, filter_metadata=filter_metadata)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk dictionary or None if not found
        """
        try:
            chunk = self.vector_store.get_chunk(chunk_id)
            if chunk:
                return {
                    'chunk_id': chunk.get('id', ''),
                    'text': chunk.get('text', ''),
                    'metadata': chunk.get('metadata', {}),
                    'similarity_score': None,  # Not applicable for direct lookup
                    'query': None
                }
            return None
        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id}: {e}")
            return None


# Convenience function
def search_chunks(
    query_text: str,
    top_k: Optional[int] = None,
    vector_store: Optional[VectorStore] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to search for chunks.
    
    Args:
        query_text: Query text string
        top_k: Number of results to return (uses config if None)
        vector_store: Optional VectorStore instance
        
    Returns:
        List of result dictionaries
    """
    searcher = ChunkSearcher(vector_store=vector_store)
    return searcher.search(query_text, top_k=top_k)
