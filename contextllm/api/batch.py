"""Batch processing utilities for multiple queries."""

import logging
from typing import List, Dict, Any, Optional
from contextllm.retrieval.searcher import search_chunks
from contextllm.optimization.optimizer import optimize_context
from contextllm.generation.generator import ResponseGenerator
from contextllm.utils.progress import create_progress_bar

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process multiple queries in batch."""
    
    def __init__(self, generator: Optional[ResponseGenerator] = None):
        """
        Initialize batch processor.
        
        Args:
            generator: ResponseGenerator instance (created if None)
        """
        from contextllm.generation.generator import ResponseGenerator
        self.generator = generator or ResponseGenerator()
        logger.info("Batch processor initialized")
    
    def process_batch(
        self,
        queries: List[str],
        budget: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query strings
            budget: Token budget for all queries (uses config if None)
            show_progress: Whether to show progress bar
            
        Returns:
            List of result dictionaries
        """
        results = []
        progress = create_progress_bar(len(queries), "Processing queries") if show_progress else None
        
        try:
            for query in queries:
                try:
                    # Retrieve chunks
                    chunks = search_chunks(query, top_k=50)
                    
                    if not chunks:
                        results.append({
                            'query': query,
                            'success': False,
                            'error': 'No relevant chunks found'
                        })
                        if progress:
                            progress.update(1)
                        continue
                    
                    # Optimize
                    optimization_result = optimize_context(chunks, budget=budget)
                    selected_chunks = optimization_result.get('selected_chunks', [])
                    
                    if not selected_chunks:
                        results.append({
                            'query': query,
                            'success': False,
                            'error': 'No chunks fit within budget'
                        })
                        if progress:
                            progress.update(1)
                        continue
                    
                    # Generate answer
                    generation_result = self.generator.generate(
                        query=query,
                        selected_chunks=selected_chunks
                    )
                    
                    results.append({
                        'query': query,
                        'success': True,
                        'answer': generation_result.get('answer', ''),
                        'usage': generation_result.get('usage', {}),
                        'optimization': {
                            'chunks_selected': len(selected_chunks),
                            'total_tokens': optimization_result.get('total_tokens', 0)
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {e}")
                    results.append({
                        'query': query,
                        'success': False,
                        'error': str(e)
                    })
                
                if progress:
                    progress.update(1)
            
            if progress:
                progress.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            if progress:
                progress.close()
            raise
