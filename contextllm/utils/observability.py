"""Observability utilities for tracking decisions and system behavior."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextllm.utils.config import get_config

logger = logging.getLogger(__name__)


class DecisionLogger:
    """Logs optimization and retrieval decisions for explainability."""
    
    def __init__(self):
        """Initialize decision logger."""
        self.decisions = []
        logger.debug("Decision logger initialized")
    
    def log_retrieval(
        self,
        query: str,
        chunks_retrieved: List[Dict[str, Any]],
        top_k: int
    ) -> None:
        """
        Log retrieval decisions.
        
        Args:
            query: Query text
            chunks_retrieved: List of retrieved chunks with scores
            top_k: Number of chunks requested
        """
        decision = {
            'timestamp': datetime.now().isoformat(),
            'type': 'retrieval',
            'query': query,
            'top_k': top_k,
            'chunks_retrieved': len(chunks_retrieved),
            'chunks': [
                {
                    'chunk_id': chunk.get('chunk_id', ''),
                    'similarity_score': chunk.get('similarity_score', 0),
                    'source': chunk.get('metadata', {}).get('filename', 'unknown')
                }
                for chunk in chunks_retrieved[:10]  # Log top 10
            ]
        }
        
        self.decisions.append(decision)
        logger.debug(f"Logged retrieval decision: {len(chunks_retrieved)} chunks for query")
    
    def log_optimization(
        self,
        query: str,
        chunks_evaluated: List[Dict[str, Any]],
        optimization_result: Dict[str, Any]
    ) -> None:
        """
        Log optimization decisions.
        
        Args:
            query: Query text
            chunks_evaluated: All chunks that were evaluated
            optimization_result: Result from optimizer
        """
        selected = optimization_result.get('selected_chunks', [])
        excluded = optimization_result.get('excluded_chunks', [])
        
        decision = {
            'timestamp': datetime.now().isoformat(),
            'type': 'optimization',
            'query': query,
            'chunks_evaluated': len(chunks_evaluated),
            'chunks_selected': len(selected),
            'chunks_excluded': len(excluded),
            'total_tokens': optimization_result.get('total_tokens', 0),
            'budget_used': optimization_result.get('budget_used', 0),
            'selection_details': {
                'selected': [
                    {
                        'chunk_id': chunk.get('chunk_id', ''),
                        'value_per_token': chunk.get('value_per_token', 0),
                        'relevance_score': chunk.get('relevance_score', 0),
                        'token_count': chunk.get('token_count', 0)
                    }
                    for chunk in selected[:10]  # Log top 10
                ],
                'excluded': [
                    {
                        'chunk_id': chunk.get('chunk_id', ''),
                        'reason': chunk.get('metadata', {}).get('inclusion_reason', 'unknown'),
                        'value_per_token': chunk.get('value_per_token', 0)
                    }
                    for chunk in excluded[:5]  # Log top 5 excluded
                ]
            }
        }
        
        self.decisions.append(decision)
        logger.debug(f"Logged optimization decision: {len(selected)} selected, {len(excluded)} excluded")
    
    def log_generation(
        self,
        query: str,
        generation_result: Dict[str, Any]
    ) -> None:
        """
        Log generation decisions.
        
        Args:
            query: Query text
            generation_result: Result from generator
        """
        decision = {
            'timestamp': datetime.now().isoformat(),
            'type': 'generation',
            'query': query,
            'model': generation_result.get('model', 'unknown'),
            'chunks_used': generation_result.get('chunks_used', 0),
            'usage': generation_result.get('usage', {}),
            'answer_length': len(generation_result.get('answer', ''))
        }
        
        self.decisions.append(decision)
        logger.debug(f"Logged generation decision: {generation_result.get('chunks_used', 0)} chunks used")
    
    def get_decisions(self, decision_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get logged decisions.
        
        Args:
            decision_type: Optional filter by type ('retrieval', 'optimization', 'generation')
            
        Returns:
            List of decision dictionaries
        """
        if decision_type:
            return [d for d in self.decisions if d.get('type') == decision_type]
        return self.decisions
    
    def clear(self) -> None:
        """Clear all logged decisions."""
        self.decisions.clear()
        logger.debug("Cleared all decisions")


# Global decision logger instance
_decision_logger: Optional[DecisionLogger] = None


def get_decision_logger() -> DecisionLogger:
    """Get or create global decision logger instance."""
    global _decision_logger
    if _decision_logger is None:
        _decision_logger = DecisionLogger()
    return _decision_logger
