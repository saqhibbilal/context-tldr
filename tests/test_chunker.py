"""Tests for text chunker."""

import pytest
from contextllm.ingestion.chunker import TextChunker


def test_chunker_initialization():
    """Test chunker initialization."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk_size == 100
    assert chunker.chunk_overlap == 10


def test_chunk_simple_text():
    """Test chunking simple text."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=5)
    text = "This is a test. " * 10  # ~150 characters
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 0
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_chunk_empty_text():
    """Test chunking empty text."""
    chunker = TextChunker()
    chunks = chunker.chunk("")
    assert chunks == []


def test_chunk_document():
    """Test chunking a document."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    document = {
        'text': "This is a test document. " * 20,
        'metadata': {'filename': 'test.txt'}
    }
    
    chunk_list = chunker.chunk_document(document)
    assert len(chunk_list) > 0
    assert all('text' in chunk for chunk in chunk_list)
    assert all('metadata' in chunk for chunk in chunk_list)
