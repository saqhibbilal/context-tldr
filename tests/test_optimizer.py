"""Tests for optimization layer."""

import pytest
from contextllm.optimization.optimizer import optimize_context
from contextllm.optimization.budget import BudgetManager, validate_budget


def test_validate_budget():
    """Test budget validation."""
    # Valid budget
    assert validate_budget(2000) == 2000
    
    # Budget below minimum should be clamped
    budget = validate_budget(100)
    assert budget >= 500  # Assuming min is 500


def test_budget_manager():
    """Test budget manager."""
    manager = BudgetManager(budget=2000, reserve_tokens=200)
    assert manager.get_total() == 2000
    assert manager.get_available() == 1800
    assert manager.can_fit(100)
    assert not manager.can_fit(2000)


def test_optimize_context():
    """Test context optimization."""
    # Create mock chunks
    chunks = [
        {
            'chunk_id': '1',
            'text': 'Short text.',
            'similarity_score': 0.9,
            'metadata': {}
        },
        {
            'chunk_id': '2',
            'text': 'Another short text.',
            'similarity_score': 0.8,
            'metadata': {}
        }
    ]
    
    result = optimize_context(chunks, budget=1000)
    
    assert 'selected_chunks' in result
    assert 'excluded_chunks' in result
    assert 'total_tokens' in result
    assert isinstance(result['selected_chunks'], list)


def test_optimize_empty_chunks():
    """Test optimization with empty chunks."""
    result = optimize_context([], budget=1000)
    assert len(result['selected_chunks']) == 0
    assert result['total_tokens'] == 0
