"""Embedding generation using SentenceTransformers."""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from contextllm.utils.config import get_config

logger = logging.getLogger(__name__)

# Global embedding model instance (lazy initialization)
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """
    Get or create SentenceTransformer model instance.
    
    Args:
        model_name: Name of the model (uses config if None)
        
    Returns:
        SentenceTransformer instance
    """
    global _embedding_model
    if _embedding_model is None:
        config = get_config()
        model = model_name or config.get("embedding.model_name", "all-MiniLM-L6-v2")
        
        try:
            logger.info(f"Loading embedding model: {model}")
            _embedding_model = SentenceTransformer(model)
            logger.info(f"Embedding model loaded successfully (dimension: {_embedding_model.get_sentence_embedding_dimension()})")
        except Exception as e:
            logger.error(f"Error loading embedding model {model}: {e}")
            raise
    
    return _embedding_model


def generate_embeddings(texts: List[str], model_name: Optional[str] = None, batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model_name: Optional model name (uses config if None)
        batch_size: Batch size for processing
        
    Returns:
        NumPy array of embeddings (shape: [num_texts, embedding_dim])
    """
    if not texts:
        return np.array([])
    
    try:
        model = get_embedding_model(model_name)
        
        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")
        
        # Generate embeddings in batches for efficiency
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better similarity search
        )
        
        logger.info(f"Generated embeddings: shape {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


def generate_embedding(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Generate embedding for a single text.
    
    Args:
        text: Text string to embed
        model_name: Optional model name (uses config if None)
        
    Returns:
        NumPy array of embedding (1D array)
    """
    embeddings = generate_embeddings([text], model_name=model_name)
    return embeddings[0] if len(embeddings) > 0 else np.array([])
