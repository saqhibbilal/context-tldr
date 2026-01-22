"""Prompt construction utilities."""

import logging
from typing import List, Dict, Any, Optional
from contextllm.utils.config import get_config

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds prompts for LLM with context chunks."""
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize prompt builder.
        
        Args:
            system_prompt: System prompt text (uses config if None)
        """
        config = get_config()
        self.system_prompt = system_prompt or config.get(
            "generation.system_prompt",
            "You are a helpful assistant that answers questions based on the provided context."
        )
        logger.debug("Prompt builder initialized")
    
    def build_context_section(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context section from selected chunks.
        
        Args:
            chunks: List of selected chunk dictionaries
            
        Returns:
            Formatted context section string
        """
        if not chunks:
            return "No context provided."
        
        context_parts = []
        context_parts.append("Context:\n")
        
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            source = metadata.get('filename', metadata.get('source', 'unknown'))
            
            context_parts.append(f"[Context {i} from {source}]\n{text}\n")
        
        context_parts.append("\n---\n")
        
        return "\n".join(context_parts)
    
    def build_messages(
        self,
        user_query: str,
        chunks: List[Dict[str, Any]],
        include_context_metadata: bool = False
    ) -> List[Dict[str, str]]:
        """
        Build chat messages for Mistral API.
        
        Args:
            user_query: User's question/query
            chunks: List of selected context chunks
            include_context_metadata: Whether to include chunk metadata in context
            
        Returns:
            List of message dictionaries for Mistral API
        """
        messages = []
        
        # System message
        system_content = self.system_prompt
        
        # Add context to system message if chunks provided
        if chunks:
            context_section = self.build_context_section(chunks)
            system_content += f"\n\n{context_section}"
            
            if include_context_metadata:
                # Add metadata summary
                metadata_info = f"\nYou have access to {len(chunks)} context chunks."
                system_content += metadata_info
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # User message
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        logger.debug(f"Built prompt with {len(chunks)} chunks, {len(messages)} messages")
        return messages
    
    def build_simple_prompt(
        self,
        user_query: str,
        chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Build a simple text prompt (alternative format).
        
        Args:
            user_query: User's question/query
            chunks: List of selected context chunks
            
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # System instruction
        prompt_parts.append(self.system_prompt)
        prompt_parts.append("\n\n")
        
        # Context section
        if chunks:
            prompt_parts.append(self.build_context_section(chunks))
        else:
            prompt_parts.append("No context provided.\n\n")
        
        # User query
        prompt_parts.append(f"Question: {user_query}\n\n")
        prompt_parts.append("Answer:")
        
        return "".join(prompt_parts)
