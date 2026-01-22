"""Configuration management using YAML files."""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the Context Budget Optimizer."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config YAML file. If None, looks for config.yaml in project root.
        """
        if config_path is None:
            # Try to find config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            self.config = self._get_default_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            self.config = self._get_default_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "embedding.model_name")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "embedding.model_name")
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save current configuration to YAML file."""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "embedding": {
                "model_name": "all-MiniLM-L6-v2"
            },
            "chunking": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "chunk_by_sentences": True
            },
            "vector_db": {
                "type": "chroma",
                "persist_directory": "./data/vector_db",
                "collection_name": "context_chunks"
            },
            "retrieval": {
                "top_k": 50
            },
            "optimization": {
                "default_budget": 2000,
                "min_budget": 500,
                "max_budget": 8000,
                "reserve_tokens": 200,
                "relevance_weight": 1.0
            },
            "generation": {
                "model": "mistral-small",
                "temperature": 0.7,
                "max_tokens": 1000,
                "system_prompt": "You are a helpful assistant that answers questions based on the provided context."
            },
            "metadata": {
                "db_path": "./data/metadata.db"
            },
            "logging": {
                "level": "INFO",
                "log_file": "./data/app.log"
            }
        }
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.get("vector_db.persist_directory", "./data/vector_db"),
            Path(self.get("metadata.db_path", "./data/metadata.db")).parent,
            Path(self.get("logging.log_file", "./data/app.log")).parent,
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")


# Global config instance (lazy initialization)
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
        _config.ensure_directories()
    return _config
