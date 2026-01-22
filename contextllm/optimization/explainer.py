"""Explainability utilities for optimization decisions."""

import logging
from typing import List, Dict, Any
from contextllm.optimization.token_estimator import get_chunk_token_count
from contextllm.optimization.scorer import get_relevance_score

logger = logging.getLogger(__name__)


class DecisionExplainer:
    """Explains optimization decisions in human-readable format."""
    
    def explain_optimization(
        self,
        optimization_result: Dict[str, Any],
        top_n: int = 5
    ) -> str:
        """
        Generate human-readable explanation of optimization decisions.
        
        Args:
            optimization_result: Result from optimizer
            top_n: Number of top chunks to explain in detail
            
        Returns:
            Explanation string
        """
        selected = optimization_result.get('selected_chunks', [])
        excluded = optimization_result.get('excluded_chunks', [])
        metadata = optimization_result.get('selection_metadata', {})
        
        explanation_parts = []
        
        # Summary
        explanation_parts.append("=" * 60)
        explanation_parts.append("OPTIMIZATION DECISION SUMMARY")
        explanation_parts.append("=" * 60)
        explanation_parts.append("")
        
        explanation_parts.append(f"Algorithm: {metadata.get('algorithm', 'unknown')}")
        explanation_parts.append(f"Chunks Evaluated: {metadata.get('chunks_evaluated', 0)}")
        explanation_parts.append(f"Chunks Selected: {metadata.get('chunks_selected', 0)}")
        explanation_parts.append(f"Chunks Excluded: {metadata.get('chunks_excluded', 0)}")
        explanation_parts.append("")
        
        # Budget information
        explanation_parts.append("Budget Information:")
        explanation_parts.append(f"  Total Budget: {metadata.get('total_budget', 0)} tokens")
        explanation_parts.append(f"  Available Budget: {metadata.get('available_budget', 0)} tokens")
        explanation_parts.append(f"  Reserve Tokens: {metadata.get('reserve_tokens', 0)} tokens")
        explanation_parts.append(f"  Tokens Used: {optimization_result.get('total_tokens', 0)} tokens")
        explanation_parts.append(f"  Budget Utilization: {optimization_result.get('budget_used', 0):.1f}%")
        explanation_parts.append("")
        
        # Selected chunks
        if selected:
            explanation_parts.append(f"SELECTED CHUNKS (Top {min(top_n, len(selected))}):")
            explanation_parts.append("-" * 60)
            
            for i, chunk in enumerate(selected[:top_n], 1):
                chunk_id = chunk.get('chunk_id', 'unknown')
                value = chunk.get('value_per_token', 0)
                relevance = get_relevance_score(chunk)
                tokens = get_chunk_token_count(chunk)
                source = chunk.get('metadata', {}).get('filename', 'unknown')
                
                explanation_parts.append(f"\n{i}. Chunk {chunk_id[:8]}... (from {source})")
                explanation_parts.append(f"   Value per Token: {value:.4f}")
                explanation_parts.append(f"   Relevance Score: {relevance:.3f}")
                explanation_parts.append(f"   Token Count: {tokens}")
                explanation_parts.append(f"   Why Selected: Highest value per token")
            
            if len(selected) > top_n:
                explanation_parts.append(f"\n... and {len(selected) - top_n} more chunks")
        
        # Excluded chunks
        if excluded:
            explanation_parts.append("")
            explanation_parts.append(f"EXCLUDED CHUNKS (Sample):")
            explanation_parts.append("-" * 60)
            
            # Group by exclusion reason
            by_reason = {}
            for chunk in excluded:
                reason = chunk.get('metadata', {}).get('inclusion_reason', 'unknown')
                if reason not in by_reason:
                    by_reason[reason] = []
                by_reason[reason].append(chunk)
            
            for reason, chunks in by_reason.items():
                explanation_parts.append(f"\n{reason}: {len(chunks)} chunks")
                # Show example
                if chunks:
                    example = chunks[0]
                    value = example.get('value_per_token', 0)
                    tokens = get_chunk_token_count(example)
                    explanation_parts.append(f"  Example: value={value:.4f}, tokens={tokens}")
        
        explanation_parts.append("")
        explanation_parts.append("=" * 60)
        
        return "\n".join(explanation_parts)
    
    def explain_chunk_selection(
        self,
        chunk: Dict[str, Any]
    ) -> str:
        """
        Explain why a specific chunk was selected or excluded.
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            Explanation string
        """
        chunk_id = chunk.get('chunk_id', 'unknown')
        included = chunk.get('metadata', {}).get('included', False)
        reason = chunk.get('metadata', {}).get('inclusion_reason', 'unknown')
        value = chunk.get('value_per_token', 0)
        relevance = get_relevance_score(chunk)
        tokens = get_chunk_token_count(chunk)
        
        explanation = f"Chunk {chunk_id[:8]}...: "
        
        if included:
            explanation += f"SELECTED\n"
            explanation += f"  Reason: {reason}\n"
            explanation += f"  Value per Token: {value:.4f} (relevance: {relevance:.3f} / tokens: {tokens})\n"
            explanation += f"  This chunk provides the best information density within the budget."
        else:
            explanation += f"EXCLUDED\n"
            explanation += f"  Reason: {reason}\n"
            explanation += f"  Value per Token: {value:.4f} (relevance: {relevance:.3f} / tokens: {tokens})\n"
            if reason == 'exceeds_budget':
                explanation += f"  This chunk alone exceeds the available budget."
            elif reason == 'budget_exhausted':
                explanation += f"  Budget was exhausted before this chunk could be included."
            else:
                explanation += f"  This chunk was not selected due to lower value per token."
        
        return explanation
    
    def generate_summary_stats(
        self,
        optimization_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for optimization.
        
        Args:
            optimization_result: Result from optimizer
            
        Returns:
            Dictionary with summary statistics
        """
        selected = optimization_result.get('selected_chunks', [])
        excluded = optimization_result.get('excluded_chunks', [])
        
        if selected:
            selected_values = [chunk.get('value_per_token', 0) for chunk in selected]
            selected_relevance = [get_relevance_score(chunk) for chunk in selected]
            selected_tokens = [get_chunk_token_count(chunk) for chunk in selected]
        else:
            selected_values = []
            selected_relevance = []
            selected_tokens = []
        
        if excluded:
            excluded_values = [chunk.get('value_per_token', 0) for chunk in excluded]
        else:
            excluded_values = []
        
        stats = {
            'total_chunks_evaluated': len(selected) + len(excluded),
            'chunks_selected': len(selected),
            'chunks_excluded': len(excluded),
            'selection_rate': len(selected) / (len(selected) + len(excluded)) * 100 if (selected or excluded) else 0,
            'avg_value_selected': sum(selected_values) / len(selected_values) if selected_values else 0,
            'avg_value_excluded': sum(excluded_values) / len(excluded_values) if excluded_values else 0,
            'avg_relevance_selected': sum(selected_relevance) / len(selected_relevance) if selected_relevance else 0,
            'avg_tokens_selected': sum(selected_tokens) / len(selected_tokens) if selected_tokens else 0,
            'total_tokens_used': optimization_result.get('total_tokens', 0),
            'budget_utilization': optimization_result.get('budget_used', 0)
        }
        
        return stats


# Convenience function
def explain_optimization(optimization_result: Dict[str, Any]) -> str:
    """
    Convenience function to explain optimization decisions.
    
    Args:
        optimization_result: Result from optimizer
        
    Returns:
        Explanation string
    """
    explainer = DecisionExplainer()
    return explainer.explain_optimization(optimization_result)
