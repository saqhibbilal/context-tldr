"""Query processing utilities."""

import logging
from typing import Optional
import numpy as np
from contextllm.ingestion.embedder import generate_embedding

logger = logging.getLogger(__name__)


def embed_query(query_text: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Generate embedding for a query text.
    
    Args:
        query_text: Query text string
        model_name: Optional model name (uses config if None)
        
    Returns:
        NumPy array of query embedding
    """
    if not query_text or not query_text.strip():
        raise ValueError("Query text cannot be empty")
    
    try:
        logger.debug(f"Generating embedding for query: {query_text[:50]}...")
        embedding = generate_embedding(query_text, model_name=model_name)
        logger.debug(f"Query embedding generated: shape {embedding.shape}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        raise


def preprocess_query(query_text: str) -> str:
    """
    Preprocess query text (normalize, clean, etc.).
    
    Args:
        query_text: Raw query text
        
    Returns:
        Preprocessed query text
    """
    if not query_text:
        return ""
    
    # Basic preprocessing: strip whitespace
    processed = query_text.strip()
    
    # Remove excessive whitespace
    import re
    processed = re.sub(r'\s+', ' ', processed)
    
    return processed
