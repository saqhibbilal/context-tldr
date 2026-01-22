"""Relevance scoring utilities."""

import logging
from typing import List, Dict, Any, Optional
from contextllm.utils.config import get_config

logger = logging.getLogger(__name__)


def get_relevance_score(chunk: Dict[str, Any]) -> float:
    """
    Get relevance score from a chunk.
    
    Args:
        chunk: Chunk dictionary with 'similarity_score' or 'metadata'
        
    Returns:
        Relevance score (0-1)
    """
    # First, try to get similarity_score from retrieval
    if 'similarity_score' in chunk:
        score = chunk['similarity_score']
        if score is not None:
            return float(score)
    
    # Try to get from metadata
    metadata = chunk.get('metadata', {})
    if 'similarity_score' in metadata:
        score = metadata['similarity_score']
        if score is not None:
            return float(score)
    
    # Default to 0 if no score found
    logger.warning(f"No relevance score found for chunk {chunk.get('chunk_id', 'unknown')}")
    return 0.0


def apply_metadata_boost(chunk: Dict[str, Any], base_score: float) -> float:
    """
    Apply optional metadata-based boosts to relevance score.
    
    Args:
        chunk: Chunk dictionary
        base_score: Base relevance score
        
    Returns:
        Boosted relevance score
    """
    metadata = chunk.get('metadata', {})
    boosted_score = base_score
    
    # Example: Boost based on recency (if recency field exists)
    if 'recency' in metadata:
        recency = metadata['recency']
        # Normalize recency to 0-1 and apply small boost
        # This is a placeholder - implement based on your needs
        pass
    
    # Example: Boost based on importance (if importance field exists)
    if 'importance' in metadata:
        importance = metadata.get('importance', 0)
        # Apply boost based on importance
        # This is a placeholder - implement based on your needs
        pass
    
    return boosted_score


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to 0-1 range using min-max normalization.
    
    Args:
        scores: List of scores
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        # All scores are the same, return all as 1.0
        return [1.0] * len(scores)
    
    # Min-max normalization
    normalized = [(s - min_score) / (max_score - min_score) for s in scores]
    return normalized


def score_chunks(chunks: List[Dict[str, Any]], normalize: bool = False) -> List[Dict[str, Any]]:
    """
    Add relevance scores to chunks.
    
    Args:
        chunks: List of chunk dictionaries
        normalize: Whether to normalize scores
        
    Returns:
        List of chunks with 'relevance_score' added
    """
    config = get_config()
    relevance_weight = config.get("optimization.relevance_weight", 1.0)
    
    # Get base scores
    base_scores = [get_relevance_score(chunk) for chunk in chunks]
    
    # Apply metadata boosts
    boosted_scores = [
        apply_metadata_boost(chunk, base_score) * relevance_weight
        for chunk, base_score in zip(chunks, base_scores)
    ]
    
    # Normalize if requested
    if normalize:
        boosted_scores = normalize_scores(boosted_scores)
    
    # Add scores to chunks
    for chunk, score in zip(chunks, boosted_scores):
        if 'metadata' not in chunk:
            chunk['metadata'] = {}
        chunk['metadata']['relevance_score'] = score
        chunk['relevance_score'] = score  # Also add at top level
    
    return chunks
