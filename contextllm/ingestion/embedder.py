"""Embedding generation using SentenceTransformers."""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from contextllm.utils.config import get_config
from contextllm.utils.cache import get_embedding_cache

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
        
        # Check cache for each text
        cache = get_embedding_cache()
        cached_embeddings = {}
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            cached = cache.get(text)
            if cached is not None:
                cached_embeddings[i] = cached
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Generate embeddings only for uncached texts
        if texts_to_embed:
            embeddings_new = model.encode(
                texts_to_embed,
                batch_size=batch_size,
                show_progress_bar=len(texts_to_embed) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Cache new embeddings
            for text, emb in zip(texts_to_embed, embeddings_new):
                cache.set(text, emb)
        else:
            # All cached - get embedding dimension from first cached embedding
            if cached_embeddings:
                first_emb = list(cached_embeddings.values())[0]
                embeddings_new = np.array([]).reshape(0, len(first_emb))
            else:
                embeddings_new = np.array([])
        
        # Combine cached and new embeddings
        if cached_embeddings and len(embeddings_new) > 0:
            # Get embedding dimension
            emb_dim = embeddings_new.shape[1] if len(embeddings_new) > 0 else len(list(cached_embeddings.values())[0])
            all_embeddings = np.zeros((len(texts), emb_dim))
            
            for i in range(len(texts)):
                if i in cached_embeddings:
                    all_embeddings[i] = cached_embeddings[i]
                else:
                    emb_idx = indices_to_embed.index(i)
                    all_embeddings[i] = embeddings_new[emb_idx]
            embeddings = all_embeddings
        elif cached_embeddings:
            # All were cached
            emb_dim = len(list(cached_embeddings.values())[0])
            embeddings = np.array([cached_embeddings[i] for i in range(len(texts))])
        else:
            embeddings = embeddings_new
        
        logger.info(f"Generated embeddings: shape {embeddings.shape} ({len(cached_embeddings)} from cache)")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


def generate_embedding(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Generate embedding for a single text (with caching).
    
    Args:
        text: Text string to embed
        model_name: Optional model name (uses config if None)
        
    Returns:
        NumPy array of embedding (1D array)
    """
    # Check cache first
    cache = get_embedding_cache()
    cached = cache.get(text)
    if cached is not None:
        logger.debug("Using cached embedding")
        return cached
    
    # Generate if not cached
    embeddings = generate_embeddings([text], model_name=model_name)
    embedding = embeddings[0] if len(embeddings) > 0 else np.array([])
    
    # Cache it
    if len(embedding) > 0:
        cache.set(text, embedding)
    
    return embedding
