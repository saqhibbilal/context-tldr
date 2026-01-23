"""Custom error classes and error handling utilities."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ContextBudgetError(Exception):
    """Base exception for Context Budget Optimizer."""
    pass


class APIKeyError(ContextBudgetError):
    """Raised when API key is missing or invalid."""
    def __init__(self, message: Optional[str] = None):
        if message is None:
            message = (
                "Mistral API key not found. Please set MISTRAL_API_KEY environment variable "
                "or create a .env file with your API key. Get your key from: https://console.mistral.ai/"
            )
        super().__init__(message)


class NoDocumentsError(ContextBudgetError):
    """Raised when no documents have been ingested."""
    def __init__(self, message: Optional[str] = None):
        if message is None:
            message = (
                "No documents have been ingested yet. Please ingest documents first using:\n"
                "  python -m contextllm.main ingest <file1> <file2> ...\n"
                "Or use the web interface to upload documents."
            )
        super().__init__(message)


class InvalidFileFormatError(ContextBudgetError):
    """Raised when file format is not supported."""
    def __init__(self, file_path: str, supported_formats: list):
        message = (
            f"Unsupported file format: {file_path}\n"
            f"Supported formats: {', '.join(supported_formats)}\n"
            f"Please convert your file to one of the supported formats."
        )
        super().__init__(message)
        self.file_path = file_path
        self.supported_formats = supported_formats


class FileNotFoundError(ContextBudgetError):
    """Raised when a file is not found."""
    def __init__(self, file_path: str):
        message = f"File not found: {file_path}\nPlease check the file path and try again."
        super().__init__(message)
        self.file_path = file_path


class RateLimitError(ContextBudgetError):
    """Raised when API rate limit is exceeded."""
    def __init__(self, message: Optional[str] = None, retry_after: Optional[int] = None):
        if message is None:
            message = "API rate limit exceeded. Please wait a moment and try again."
            if retry_after:
                message += f" Retry after {retry_after} seconds."
        super().__init__(message)
        self.retry_after = retry_after


class BudgetTooSmallError(ContextBudgetError):
    """Raised when budget is too small for any chunks."""
    def __init__(self, budget: int, min_required: int):
        message = (
            f"Token budget ({budget}) is too small. "
            f"Minimum required: {min_required} tokens.\n"
            f"Please increase the budget or reduce chunk sizes in configuration."
        )
        super().__init__(message)
        self.budget = budget
        self.min_required = min_required


class NoChunksFoundError(ContextBudgetError):
    """Raised when no relevant chunks are found for a query."""
    def __init__(self, query: str):
        message = (
            f"No relevant chunks found for query: '{query[:50]}...'\n"
            f"This might mean:\n"
            f"  - No documents have been ingested\n"
            f"  - The query doesn't match any document content\n"
            f"  - Try rephrasing your query or ingesting more documents"
        )
        super().__init__(message)
        self.query = query


def handle_api_error(error: Exception) -> ContextBudgetError:
    """
    Convert API errors to user-friendly errors.
    
    Args:
        error: Original exception
        
    Returns:
        ContextBudgetError with user-friendly message
    """
    error_str = str(error).lower()
    
    # Check for rate limit
    if 'rate limit' in error_str or '429' in error_str or 'too many requests' in error_str:
        return RateLimitError()
    
    # Check for API key issues
    if 'api key' in error_str or 'unauthorized' in error_str or '401' in error_str or '403' in error_str:
        return APIKeyError("Invalid or missing Mistral API key. Please check your MISTRAL_API_KEY.")
    
    # Return generic error
    return ContextBudgetError(f"API error: {str(error)}")
