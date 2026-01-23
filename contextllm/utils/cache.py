"""Caching utilities for embeddings and token counts."""

import logging
import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
from contextllm.utils.config import get_config

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory for cache files (uses config if None)
        """
        config = get_config()
        if cache_dir is None:
            cache_dir = config.get("cache.embedding_dir", "./data/cache/embeddings")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Embedding cache initialized at {self.cache_dir}")
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Text to generate key for
            
        Returns:
            Cache key (hash)
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[Any]:
        """
        Get cached embedding.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None
        """
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache for {cache_key}: {e}")
                return None
        
        return None
    
    def set(self, text: str, embedding: Any) -> None:
        """
        Cache an embedding.
        
        Args:
            text: Text that was embedded
            embedding: Embedding to cache
        """
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Error caching embedding for {cache_key}: {e}")


class TokenCountCache:
    """Cache for token counts."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize token count cache.
        
        Args:
            cache_dir: Directory for cache files (uses config if None)
        """
        config = get_config()
        if cache_dir is None:
            cache_dir = config.get("cache.token_dir", "./data/cache/tokens")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for faster access
        self.memory_cache: Dict[str, int] = {}
        
        logger.info(f"Token count cache initialized at {self.cache_dir}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[int]:
        """
        Get cached token count.
        
        Args:
            text: Text to get count for
            
        Returns:
            Cached token count or None
        """
        cache_key = self._get_cache_key(text)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    count = data.get('count')
                    # Store in memory cache
                    self.memory_cache[cache_key] = count
                    return count
            except Exception as e:
                logger.warning(f"Error loading token cache: {e}")
        
        return None
    
    def set(self, text: str, count: int) -> None:
        """
        Cache a token count.
        
        Args:
            text: Text that was counted
            count: Token count to cache
        """
        cache_key = self._get_cache_key(text)
        
        # Store in memory cache
        self.memory_cache[cache_key] = count
        
        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({'count': count}, f)
        except Exception as e:
            logger.warning(f"Error caching token count: {e}")


# Global cache instances
_embedding_cache: Optional[EmbeddingCache] = None
_token_cache: Optional[TokenCountCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_token_cache() -> TokenCountCache:
    """Get or create global token count cache."""
    global _token_cache
    if _token_cache is None:
        _token_cache = TokenCountCache()
    return _token_cache
