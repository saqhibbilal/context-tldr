"""Text chunking utilities for splitting documents into manageable pieces."""

import logging
import re
from typing import List, Dict, Any, Optional
from contextllm.utils.config import get_config

logger = logging.getLogger(__name__)


class TextChunker:
    """Chunks text into smaller pieces with optional sentence-aware splitting."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunk_by_sentences: Optional[bool] = None
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Maximum characters per chunk (uses config if None)
            chunk_overlap: Overlap between chunks in characters (uses config if None)
            chunk_by_sentences: Whether to try chunking at sentence boundaries (uses config if None)
        """
        config = get_config()
        
        self.chunk_size = chunk_size or config.get("chunking.chunk_size", 500)
        self.chunk_overlap = chunk_overlap or config.get("chunking.chunk_overlap", 50)
        self.chunk_by_sentences = (
            chunk_by_sentences 
            if chunk_by_sentences is not None 
            else config.get("chunking.chunk_by_sentences", True)
        )
        
        # Validate parameters
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Pattern to match sentence endings (period, exclamation, question mark)
        # Followed by whitespace and capital letter or end of string
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by trying to split at sentence boundaries.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If single sentence exceeds chunk size, split it
            if sentence_length > self.chunk_size:
                # First, save current chunk if it exists
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split the long sentence by character count
                words = sentence.split()
                temp_chunk = []
                temp_length = 0
                
                for word in words:
                    word_length = len(word) + 1  # +1 for space
                    if temp_length + word_length > self.chunk_size and temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        temp_chunk = [word]
                        temp_length = len(word)
                    else:
                        temp_chunk.append(word)
                        temp_length += word_length
                
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_length = temp_length
            else:
                # Check if adding this sentence would exceed chunk size
                if current_length + sentence_length + 1 > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(' '.join(current_chunk))
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0 and current_chunk:
                        # Include last few sentences for overlap
                        overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) >= 2 else ' '.join(current_chunk)
                        overlap_text = overlap_text[-self.chunk_overlap:] if len(overlap_text) > self.chunk_overlap else overlap_text
                        current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                        current_length = len(' '.join(current_chunk))
                    else:
                        current_chunk = [sentence]
                        current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _chunk_by_characters(self, text: str) -> List[str]:
        """
        Chunk text by character count with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundary if not at end
            if end < text_length:
                # Look for last space or newline in the last 10% of chunk
                search_start = max(0, int(self.chunk_size * 0.9))
                last_space = chunk.rfind(' ', search_start)
                last_newline = chunk.rfind('\n', search_start)
                break_point = max(last_space, last_newline)
                
                if break_point > search_start:
                    chunk = chunk[:break_point]
                    end = start + break_point
            
            chunks.append(chunk.strip())
            
            # Move start position with overlap
            if end >= text_length:
                break
            start = end - self.chunk_overlap
        
        return [chunk for chunk in chunks if chunk]  # Remove empty chunks
    
    def chunk(self, text: str) -> List[str]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        if self.chunk_by_sentences:
            chunks = self._chunk_by_sentences(text)
        else:
            chunks = self._chunk_by_characters(text)
        
        logger.debug(f"Created {len(chunks)} chunks from text ({len(text)} characters)")
        return chunks
    
    def chunk_document(
        self, 
        document: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document and return chunks with metadata.
        
        Args:
            document: Document dictionary with 'text' and 'metadata' keys
            
        Returns:
            List of chunk dictionaries with 'text' and 'metadata' keys
        """
        text = document.get('text', '')
        doc_metadata = document.get('metadata', {})
        
        chunks = self.chunk(text)
        chunk_list = []
        
        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = {
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'source': doc_metadata.get('source', ''),
                'filename': doc_metadata.get('filename', ''),
                'file_type': doc_metadata.get('file_type', ''),
            }
            
            chunk_list.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        logger.info(f"Chunked document '{doc_metadata.get('filename', 'unknown')}' into {len(chunks)} chunks")
        return chunk_list
