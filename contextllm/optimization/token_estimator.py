"""Token estimation utilities for chunks."""

import logging
from typing import List, Dict, Any, Optional
from contextllm.utils.tokenizer import count_tokens, count_tokens_batch

logger = logging.getLogger(__name__)


def estimate_chunk_tokens(chunk: Dict[str, Any]) -> int:
    """
    Estimate token count for a chunk.
    
    Args:
        chunk: Chunk dictionary with 'text' key
        
    Returns:
        Estimated token count
    """
    text = chunk.get('text', '')
    if not text:
        return 0
    
    try:
        token_count = count_tokens(text)
        return token_count
    except Exception as e:
        logger.warning(f"Error estimating tokens for chunk: {e}, using fallback")
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


def estimate_chunks_tokens(chunks: List[Dict[str, Any]]) -> List[int]:
    """
    Estimate token counts for multiple chunks efficiently.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of token counts
    """
    if not chunks:
        return []
    
    texts = [chunk.get('text', '') for chunk in chunks]
    
    try:
        token_counts = count_tokens_batch(texts)
        return token_counts
    except Exception as e:
        logger.warning(f"Error in batch token estimation: {e}, using fallback")
        # Fallback: individual estimation
        return [estimate_chunk_tokens(chunk) for chunk in chunks]


def add_token_counts_to_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add token_count field to chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of chunks with 'token_count' added to metadata
    """
    token_counts = estimate_chunks_tokens(chunks)
    
    for chunk, token_count in zip(chunks, token_counts):
        if 'metadata' not in chunk:
            chunk['metadata'] = {}
        chunk['metadata']['token_count'] = token_count
        chunk['token_count'] = token_count  # Also add at top level for convenience
    
    return chunks


def get_chunk_token_count(chunk: Dict[str, Any]) -> int:
    """
    Get token count from a chunk (from metadata or estimate).
    
    Args:
        chunk: Chunk dictionary
        
    Returns:
        Token count
    """
    # Try to get from top level
    if 'token_count' in chunk:
        return chunk['token_count']
    
    # Try to get from metadata
    metadata = chunk.get('metadata', {})
    if 'token_count' in metadata:
        return metadata['token_count']
    
    # Estimate if not found
    return estimate_chunk_tokens(chunk)
