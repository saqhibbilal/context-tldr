"""Token counting utilities using Mistral tokenizer."""

import logging
from typing import List, Optional
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Global tokenizer instance (lazy initialization)
_tokenizer: Optional[AutoTokenizer] = None


def get_tokenizer(model_name: str = "mistralai/Mistral-7B-v0.1") -> AutoTokenizer:
    """
    Get or create Mistral tokenizer instance.
    
    Args:
        model_name: HuggingFace model name for Mistral tokenizer
        
    Returns:
        AutoTokenizer instance
    """
    global _tokenizer
    if _tokenizer is None:
        try:
            logger.info(f"Loading tokenizer: {model_name}")
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            # Fallback to a simpler tokenizer if Mistral model not available
            logger.warning("Falling back to GPT-2 tokenizer for token counting")
            _tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return _tokenizer


def count_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    Count tokens in a text string using Mistral tokenizer.
    
    Args:
        text: Text to count tokens for
        model_name: Optional model name (uses default if None)
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    
    try:
        tokenizer = get_tokenizer(model_name) if model_name else get_tokenizer()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


def count_tokens_batch(texts: List[str], model_name: Optional[str] = None) -> List[int]:
    """
    Count tokens for multiple texts efficiently.
    
    Args:
        texts: List of texts to count tokens for
        model_name: Optional model name (uses default if None)
        
    Returns:
        List of token counts
    """
    if not texts:
        return []
    
    try:
        tokenizer = get_tokenizer(model_name) if model_name else get_tokenizer()
        # Batch encode for efficiency
        encoded = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
        return [len(tokens) for tokens in encoded['input_ids']]
    except Exception as e:
        logger.error(f"Error counting tokens in batch: {e}")
        # Fallback: rough estimate
        return [len(text) // 4 for text in texts]


def estimate_tokens_for_prompt(system_prompt: str, user_prompt: str, chunks: List[str]) -> int:
    """
    Estimate total tokens for a complete prompt including system message, user query, and context chunks.
    
    Args:
        system_prompt: System instruction text
        user_prompt: User query text
        chunks: List of context chunks to include
        
    Returns:
        Total estimated token count
    """
    # Count system prompt
    system_tokens = count_tokens(system_prompt)
    
    # Count user prompt
    user_tokens = count_tokens(user_prompt)
    
    # Count chunks
    chunk_tokens = sum(count_tokens(chunk) for chunk in chunks)
    
    # Add overhead for formatting (markers, separators, etc.)
    # Rough estimate: ~10 tokens per chunk for formatting
    formatting_overhead = len(chunks) * 10
    
    total = system_tokens + user_tokens + chunk_tokens + formatting_overhead
    logger.debug(f"Token estimate - System: {system_tokens}, User: {user_tokens}, "
                 f"Chunks: {chunk_tokens}, Formatting: {formatting_overhead}, Total: {total}")
    
    return total
