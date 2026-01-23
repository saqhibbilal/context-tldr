"""Core optimization algorithm for selecting chunks within budget."""

import logging
from typing import List, Dict, Any, Optional
from contextllm.optimization.token_estimator import (
    add_token_counts_to_chunks,
    get_chunk_token_count
)
from contextllm.optimization.scorer import score_chunks, get_relevance_score
from contextllm.optimization.budget import BudgetManager, validate_budget
from contextllm.utils.errors import BudgetTooSmallError

logger = logging.getLogger(__name__)


class ContextOptimizer:
    """Optimizes context selection based on value per token."""
    
    def __init__(self, budget: Optional[int] = None):
        """
        Initialize context optimizer.
        
        Args:
            budget: Token budget (uses config if None)
        """
        validated_budget = validate_budget(budget) if budget else None
        self.budget_manager = BudgetManager(budget=validated_budget)
        logger.info("Context optimizer initialized")
    
    def calculate_value_per_token(self, chunk: Dict[str, Any]) -> float:
        """
        Calculate value per token for a chunk.
        
        Formula: value = relevance_score / token_count
        
        Args:
            chunk: Chunk dictionary with relevance_score and token_count
            
        Returns:
            Value per token (float)
        """
        relevance = get_relevance_score(chunk)
        token_count = get_chunk_token_count(chunk)
        
        if token_count == 0:
            logger.warning(f"Chunk {chunk.get('chunk_id', 'unknown')} has zero tokens")
            return 0.0
        
        value = relevance / token_count
        return value
    
    def optimize(
        self,
        chunks: List[Dict[str, Any]],
        budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize chunk selection using greedy algorithm.
        
        Args:
            chunks: List of retrieved chunks with similarity scores
            budget: Optional budget override
            
        Returns:
            Dictionary with:
            - 'selected_chunks': List of selected chunks
            - 'excluded_chunks': List of excluded chunks
            - 'total_tokens': Total tokens used
            - 'budget_used': Percentage of budget used
            - 'selection_metadata': Additional metadata
        """
        if not chunks:
            logger.warning("No chunks provided for optimization")
            return {
                'selected_chunks': [],
                'excluded_chunks': [],
                'total_tokens': 0,
                'budget_used': 0.0,
                'selection_metadata': {
                    'algorithm': 'greedy',
                    'chunks_evaluated': 0,
                    'chunks_selected': 0
                }
            }
        
        # Update budget if provided
        if budget is not None:
            validated_budget = validate_budget(budget)
            self.budget_manager = BudgetManager(budget=validated_budget)
        
        available_budget = self.budget_manager.get_available()
        
        # Add token counts and relevance scores to chunks
        chunks = add_token_counts_to_chunks(chunks)
        chunks = score_chunks(chunks)
        
        # Calculate value per token for each chunk
        for chunk in chunks:
            value = self.calculate_value_per_token(chunk)
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata']['value_per_token'] = value
            chunk['value_per_token'] = value
        
        # Sort chunks by value per token (descending)
        sorted_chunks = sorted(chunks, key=lambda x: x.get('value_per_token', 0), reverse=True)
        
        # Greedy selection: add chunks until budget is exhausted
        selected_chunks = []
        excluded_chunks = []
        total_tokens = 0
        
        for chunk in sorted_chunks:
            token_count = get_chunk_token_count(chunk)
            
            # Check if chunk fits in remaining budget
            if total_tokens + token_count <= available_budget:
                selected_chunks.append(chunk)
                total_tokens += token_count
                chunk['metadata']['included'] = True
                chunk['metadata']['inclusion_reason'] = 'fits_in_budget'
            else:
                excluded_chunks.append(chunk)
                chunk['metadata']['included'] = False
                if total_tokens == 0:
                    chunk['metadata']['inclusion_reason'] = 'exceeds_budget'
                else:
                    chunk['metadata']['inclusion_reason'] = 'budget_exhausted'
        
        # Calculate budget usage
        budget_used = (total_tokens / available_budget * 100) if available_budget > 0 else 0
        
        # Prepare metadata
        selection_metadata = {
            'algorithm': 'greedy_value_per_token',
            'chunks_evaluated': len(chunks),
            'chunks_selected': len(selected_chunks),
            'chunks_excluded': len(excluded_chunks),
            'available_budget': available_budget,
            'total_budget': self.budget_manager.get_total(),
            'reserve_tokens': self.budget_manager.get_reserve(),
            'budget_utilization': f"{budget_used:.2f}%"
        }
        
        logger.info(
            f"Optimization complete: selected {len(selected_chunks)}/{len(chunks)} chunks, "
            f"using {total_tokens}/{available_budget} tokens ({budget_used:.1f}%)"
        )
        
        return {
            'selected_chunks': selected_chunks,
            'excluded_chunks': excluded_chunks,
            'total_tokens': total_tokens,
            'budget_used': budget_used,
            'selection_metadata': selection_metadata
        }
    
    def explain_selection(self, optimization_result: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of selection.
        
        Args:
            optimization_result: Result from optimize() method
            
        Returns:
            Explanation string
        """
        metadata = optimization_result.get('selection_metadata', {})
        selected = optimization_result.get('selected_chunks', [])
        excluded = optimization_result.get('excluded_chunks', [])
        
        explanation = f"""
Optimization Summary:
- Algorithm: {metadata.get('algorithm', 'unknown')}
- Chunks evaluated: {metadata.get('chunks_evaluated', 0)}
- Chunks selected: {metadata.get('chunks_selected', 0)}
- Chunks excluded: {metadata.get('chunks_excluded', 0)}
- Budget: {metadata.get('total_budget', 0)} tokens (available: {metadata.get('available_budget', 0)})
- Tokens used: {optimization_result.get('total_tokens', 0)}
- Budget utilization: {optimization_result.get('budget_used', 0):.1f}%

Top selected chunks (by value per token):
"""
        
        # Show top 5 selected chunks
        for i, chunk in enumerate(selected[:5], 1):
            value = chunk.get('value_per_token', 0)
            tokens = get_chunk_token_count(chunk)
            relevance = get_relevance_score(chunk)
            explanation += f"  {i}. Value: {value:.4f} (relevance: {relevance:.3f}, tokens: {tokens})\n"
        
        if len(excluded) > 0:
            explanation += f"\nExcluded chunks: {len(excluded)} (budget exhausted or too large)\n"
        
        return explanation


# Convenience function
def optimize_context(
    chunks: List[Dict[str, Any]],
    budget: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to optimize context selection.
    
    Args:
        chunks: List of retrieved chunks
        budget: Optional token budget override
        
    Returns:
        Optimization result dictionary
    """
    optimizer = ContextOptimizer(budget=budget)
    return optimizer.optimize(chunks, budget=budget)
