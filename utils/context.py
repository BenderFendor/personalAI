"""Context window calculation utilities."""

import re
from typing import List, Dict, Any
import ollama
from models import ContextUsage


class ContextCalculator:
    """Handles context window size calculations and tracking."""
    
    def __init__(self, model: str, default_size: int = 8192):
        """Initialize context calculator.
        
        Args:
            model: Model name
            default_size: Default context window size
        """
        self.model = model
        self.context_window_size = self._get_model_context_size(default_size)
    
    def _get_model_context_size(self, default_size: int) -> int:
        """Get model's context window size from Ollama.
        
        Args:
            default_size: Default size if unable to determine
            
        Returns:
            Context window size in tokens
        """
        try:
            model_info = ollama.show(self.model)
            
            # Check parameters string for num_ctx
            if 'parameters' in model_info:
                params = model_info.get('parameters', '')
                if 'num_ctx' in params:
                    match = re.search(r'num_ctx\s+(\d+)', params)
                    if match:
                        return int(match.group(1))
            
            # Check modelfile for PARAMETER num_ctx
            if 'modelfile' in model_info:
                modelfile = model_info.get('modelfile', '')
                if 'num_ctx' in modelfile:
                    match = re.search(r'PARAMETER\s+num_ctx\s+(\d+)', modelfile, re.IGNORECASE)
                    if match:
                        return int(match.group(1))
            
            # Check details for context_length
            if 'details' in model_info:
                details = model_info.get('details', {})
                if isinstance(details, dict) and 'context_length' in details:
                    return details['context_length']
            
            return default_size
            
        except Exception:
            return default_size
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count from text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count (~4 chars per token)
        """
        return len(text) // 4
    
    def calculate_usage(self, messages: List[Dict[str, Any]], system_prompt: str = "") -> ContextUsage:
        """Calculate current context window usage.
        
        Args:
            messages: List of message dictionaries
            system_prompt: System prompt text
            
        Returns:
            ContextUsage object with usage statistics
        """
        total_chars = len(system_prompt)
        
        for msg in messages:
            if isinstance(msg, dict):
                total_chars += len(str(msg.get('content', '')))
                if 'tool_calls' in msg:
                    total_chars += len(str(msg['tool_calls']))
        
        estimated_tokens = self.estimate_tokens(str(total_chars))
        
        return ContextUsage(
            current_tokens=estimated_tokens,
            max_tokens=self.context_window_size
        )
