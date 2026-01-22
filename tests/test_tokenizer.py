"""Tests for tokenizer utilities."""

import pytest
from contextllm.utils.tokenizer import count_tokens, count_tokens_batch, estimate_tokens_for_prompt


def test_count_tokens():
    """Test token counting for single text."""
    text = "Hello world, this is a test."
    tokens = count_tokens(text)
    assert tokens > 0
    assert isinstance(tokens, int)


def test_count_tokens_empty():
    """Test token counting for empty text."""
    tokens = count_tokens("")
    assert tokens == 0


def test_count_tokens_batch():
    """Test batch token counting."""
    texts = ["Hello", "World", "Test"]
    tokens = count_tokens_batch(texts)
    assert len(tokens) == len(texts)
    assert all(isinstance(t, int) for t in tokens)
    assert all(t > 0 for t in tokens)


def test_estimate_tokens_for_prompt():
    """Test prompt token estimation."""
    system = "You are a helpful assistant."
    user = "What is the answer?"
    chunks = ["Chunk 1", "Chunk 2"]
    
    tokens = estimate_tokens_for_prompt(system, user, chunks)
    assert tokens > 0
    assert isinstance(tokens, int)
