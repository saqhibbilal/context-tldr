"""Progress bar utilities for long operations."""

import logging
import sys
from typing import Optional, Callable
from contextllm.utils.config import get_config

logger = logging.getLogger(__name__)


class ProgressBar:
    """Simple progress bar for terminal output."""
    
    def __init__(self, total: int, desc: str = "Processing", width: int = 50):
        """
        Initialize progress bar.
        
        Args:
            total: Total number of items
            desc: Description text
            width: Width of progress bar in characters
        """
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self._enabled = get_config().get("ui.show_progress", True)
    
    def update(self, n: int = 1) -> None:
        """
        Update progress by n items.
        
        Args:
            n: Number of items completed
        """
        if not self._enabled:
            return
        
        self.current = min(self.current + n, self.total)
        self._draw()
    
    def _draw(self) -> None:
        """Draw the progress bar."""
        if self.total == 0:
            return
        
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        
        sys.stdout.write(f'\r{self.desc}: [{bar}] {self.current}/{self.total} ({percent*100:.1f}%)')
        sys.stdout.flush()
        
        if self.current >= self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()
    
    def close(self) -> None:
        """Close the progress bar."""
        if self._enabled and self.current < self.total:
            self.current = self.total
            self._draw()


def create_progress_bar(total: int, desc: str = "Processing") -> Optional[ProgressBar]:
    """
    Create a progress bar if enabled.
    
    Args:
        total: Total number of items
        desc: Description text
        
    Returns:
        ProgressBar instance or None if disabled
    """
    if get_config().get("ui.show_progress", True):
        return ProgressBar(total, desc)
    return None
