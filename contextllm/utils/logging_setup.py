"""Logging configuration setup."""

import logging
import sys
from pathlib import Path
from typing import Optional
from contextllm.utils.config import get_config


def setup_logging(log_file: Optional[str] = None, log_level: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (uses config if None)
        log_level: Logging level (uses config if None)
    """
    config = get_config()
    
    # Get log level
    if log_level is None:
        log_level = config.get("logging.level", "INFO")
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get log file path
    if log_file is None:
        log_file = config.get("logging.log_file", "./data/app.log")
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")
