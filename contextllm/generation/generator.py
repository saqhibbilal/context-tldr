"""Response generation orchestrator."""

import logging
from typing import List, Dict, Any, Optional
from contextllm.generation.mistral_client import MistralClient
from contextllm.generation.prompt_builder import PromptBuilder
from contextllm.utils.config import get_config

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Orchestrates response generation using optimized context."""
    
    def __init__(
        self,
        mistral_client: Optional[MistralClient] = None,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        """
        Initialize response generator.
        
        Args:
            mistral_client: MistralClient instance (created if None)
            prompt_builder: PromptBuilder instance (created if None)
        """
        self.mistral_client = mistral_client or MistralClient()
        self.prompt_builder = prompt_builder or PromptBuilder()
        logger.info("Response generator initialized")
    
    def generate(
        self,
        query: str,
        selected_chunks: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using selected context chunks.
        
        Args:
            query: User's question/query
            selected_chunks: List of selected chunks from optimizer
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            
        Returns:
            Dictionary with:
            - 'answer': Generated answer text
            - 'usage': Token usage statistics
            - 'model': Model used
            - 'chunks_used': Number of chunks used
            - 'metadata': Additional metadata
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            logger.info(f"Generating response for query: {query[:100]}...")
            logger.info(f"Using {len(selected_chunks)} context chunks")
            
            # Build prompt messages
            messages = self.prompt_builder.build_messages(
                user_query=query,
                chunks=selected_chunks,
                include_context_metadata=False
            )
            
            # Generate response
            response = self.mistral_client.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract answer
            answer = self.mistral_client.generate_text(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Get usage statistics
            usage = self.mistral_client.get_usage_stats(response)
            
            # Prepare result
            result = {
                'answer': answer,
                'usage': usage,
                'model': self.mistral_client.model,
                'chunks_used': len(selected_chunks),
                'metadata': {
                    'query': query,
                    'temperature': temperature or get_config().get("generation.temperature", 0.7),
                    'max_tokens': max_tokens or get_config().get("generation.max_tokens", 1000),
                    'chunk_sources': [
                        chunk.get('metadata', {}).get('filename', 'unknown')
                        for chunk in selected_chunks
                    ]
                }
            }
            
            logger.info(
                f"Response generated: {usage.get('total_tokens', 0)} tokens "
                f"({usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_with_optimization(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        budget: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate answer with automatic optimization (convenience method).
        
        This method combines optimization and generation in one call.
        
        Args:
            query: User's question/query
            chunks: List of retrieved chunks (will be optimized)
            budget: Optional token budget override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            
        Returns:
            Dictionary with answer and optimization metadata
        """
        from contextllm.optimization.optimizer import optimize_context
        
        # Optimize chunks
        optimization_result = optimize_context(chunks, budget=budget)
        selected_chunks = optimization_result.get('selected_chunks', [])
        
        # Generate answer
        generation_result = self.generate(
            query=query,
            selected_chunks=selected_chunks,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Combine results
        combined_result = {
            **generation_result,
            'optimization': {
                'chunks_evaluated': optimization_result.get('selection_metadata', {}).get('chunks_evaluated', 0),
                'chunks_selected': len(selected_chunks),
                'chunks_excluded': len(optimization_result.get('excluded_chunks', [])),
                'total_tokens': optimization_result.get('total_tokens', 0),
                'budget_used': optimization_result.get('budget_used', 0)
            }
        }
        
        return combined_result


# Convenience function
def generate_answer(
    query: str,
    selected_chunks: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate an answer.
    
    Args:
        query: User's question/query
        selected_chunks: List of selected context chunks
        temperature: Optional temperature override
        max_tokens: Optional max_tokens override
        
    Returns:
        Generation result dictionary
    """
    generator = ResponseGenerator()
    return generator.generate(query, selected_chunks, temperature, max_tokens)
