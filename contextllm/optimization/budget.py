"""Budget management utilities."""

import logging
from typing import Optional
from contextllm.utils.config import get_config

logger = logging.getLogger(__name__)


class BudgetManager:
    """Manages token budget for context optimization."""
    
    def __init__(
        self,
        budget: Optional[int] = None,
        reserve_tokens: Optional[int] = None
    ):
        """
        Initialize budget manager.
        
        Args:
            budget: Total token budget (uses config if None)
            reserve_tokens: Tokens to reserve for prompt template and response (uses config if None)
        """
        config = get_config()
        
        if budget is None:
            budget = config.get("optimization.default_budget", 2000)
        
        min_budget = config.get("optimization.min_budget", 500)
        max_budget = config.get("optimization.max_budget", 8000)
        
        # Validate and clamp budget
        if budget < min_budget:
            logger.warning(f"Budget {budget} below minimum {min_budget}, using minimum")
            budget = min_budget
        elif budget > max_budget:
            logger.warning(f"Budget {budget} above maximum {max_budget}, using maximum")
            budget = max_budget
        
        self.total_budget = budget
        
        if reserve_tokens is None:
            reserve_tokens = config.get("optimization.reserve_tokens", 200)
        
        self.reserve_tokens = reserve_tokens
        self.available_budget = budget - reserve_tokens
        
        if self.available_budget <= 0:
            raise ValueError(f"Reserve tokens ({reserve_tokens}) exceed or equal total budget ({budget})")
        
        logger.info(f"Budget initialized: total={budget}, reserve={reserve_tokens}, available={self.available_budget}")
    
    def can_fit(self, token_count: int) -> bool:
        """
        Check if a chunk with given token count can fit in the budget.
        
        Args:
            token_count: Token count to check
            
        Returns:
            True if chunk can fit
        """
        return token_count <= self.available_budget
    
    def get_available(self) -> int:
        """
        Get available budget.
        
        Returns:
            Available token budget
        """
        return self.available_budget
    
    def get_total(self) -> int:
        """
        Get total budget.
        
        Returns:
            Total token budget
        """
        return self.total_budget
    
    def get_reserve(self) -> int:
        """
        Get reserved tokens.
        
        Returns:
            Reserved token count
        """
        return self.reserve_tokens


def validate_budget(budget: int) -> int:
    """
    Validate and clamp budget to allowed range.
    
    Args:
        budget: Budget to validate
        
    Returns:
        Validated budget
    """
    config = get_config()
    min_budget = config.get("optimization.min_budget", 500)
    max_budget = config.get("optimization.max_budget", 8000)
    
    if budget < min_budget:
        logger.warning(f"Budget {budget} below minimum {min_budget}, using minimum")
        return min_budget
    elif budget > max_budget:
        logger.warning(f"Budget {budget} above maximum {max_budget}, using maximum")
        return max_budget
    
    return budget
