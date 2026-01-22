"""Mistral API client integration."""

import logging
import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models import ChatCompletionResponse
from contextllm.utils.config import get_config

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class MistralClient:
    """Client for interacting with Mistral API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize Mistral API client.
        
        Args:
            api_key: Mistral API key (uses env var if None)
            model: Model name (uses config if None)
        """
        # Get API key
        if api_key is None:
            api_key = os.getenv("MISTRAL_API_KEY")
        
        if not api_key:
            raise ValueError(
                "Mistral API key not found. Set MISTRAL_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Get model name
        config = get_config()
        if model is None:
            model = os.getenv("MISTRAL_MODEL") or config.get("generation.model", "mistral-small")
        
        self.api_key = api_key
        self.model = model
        
        # Initialize Mistral client
        try:
            self.client = Mistral(api_key=api_key)
            logger.info(f"Mistral client initialized with model: {model}")
        except Exception as e:
            logger.error(f"Error initializing Mistral client: {e}")
            raise
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> ChatCompletionResponse:
        """
        Generate response using Mistral API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (uses config if None)
            max_tokens: Maximum tokens for response (uses config if None)
            stream: Whether to stream the response
            
        Returns:
            ChatCompletionResponse object
        """
        config = get_config()
        
        if temperature is None:
            temperature = config.get("generation.temperature", 0.7)
        
        if max_tokens is None:
            max_tokens = config.get("generation.max_tokens", 1000)
        
        try:
            logger.debug(f"Calling Mistral API: model={self.model}, messages={len(messages)}, "
                        f"temperature={temperature}, max_tokens={max_tokens}")
            
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            logger.info("Mistral API call successful")
            return response
            
        except Exception as e:
            logger.error(f"Error calling Mistral API: {e}")
            raise
    
    def generate_text(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text response (convenience method).
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens for response
            
        Returns:
            Generated text string
        """
        response = self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            raise ValueError("No response content in API response")
    
    def get_usage_stats(self, response: ChatCompletionResponse) -> Dict[str, int]:
        """
        Extract usage statistics from API response.
        
        Args:
            response: ChatCompletionResponse object
            
        Returns:
            Dictionary with 'prompt_tokens', 'completion_tokens', 'total_tokens'
        """
        if response.usage:
            return {
                'prompt_tokens': response.usage.prompt_tokens or 0,
                'completion_tokens': response.usage.completion_tokens or 0,
                'total_tokens': response.usage.total_tokens or 0
            }
        return {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
