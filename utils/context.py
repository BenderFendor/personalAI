"""Context window calculation utilities."""

import re
from typing import List, Dict, Any
import ollama
from models import ContextUsage


class ContextCalculator:
    """Handles context window size calculations and tracking with calibration."""
    
    def __init__(self, model: str, default_size: int = 8192):
        """Initialize context calculator.
        
        Args:
            model: Model name
            default_size: Default context window size
        """
        self.model = model
        self.context_window_size = self._get_model_context_size(default_size)
        # Calibration factor to align heuristic estimates with actual prompt_eval_count
        self._calibration_factor: float = 1.0
        # Exponential moving average smoothing for stability
        self._alpha: float = 0.3
    
    def _get_model_context_size(self, default_size: int) -> int:
        """Get model's context window size from Ollama.
        
        Args:
            default_size: Default size if unable to determine
            
        Returns:
            Context window size in tokens
        """
        try:
            info = ollama.show(self.model)

            # 1) Preferred: parse PARAMETER num_ctx from 'parameters' block
            num_ctx: int | None = None
            params = info.get('parameters', '') if isinstance(info, dict) else ''
            if isinstance(params, str):
                m = re.search(r"(?im)^\s*num_ctx\s+(\d+)\s*$", params)
                if m:
                    try:
                        num_ctx = int(m.group(1))
                    except Exception:
                        num_ctx = None

            # 2) Fallback: parse from modelfile text
            if num_ctx is None:
                modelfile = info.get('modelfile', '') if isinstance(info, dict) else ''
                if isinstance(modelfile, str):
                    m = re.search(r"(?im)^\s*PARAMETER\s+num_ctx\s+(\d+)\s*$", modelfile)
                    if m:
                        try:
                            num_ctx = int(m.group(1))
                        except Exception:
                            num_ctx = None

            # 3) Fallback: model_info.*.context_length (varies by family)
            max_ctx: int | None = None
            model_info = info.get('model_info', {}) if isinstance(info, dict) else {}
            if isinstance(model_info, dict):
                # try direct key first
                if 'context_length' in model_info and isinstance(model_info['context_length'], int):
                    max_ctx = model_info['context_length']
                else:
                    # search any nested key that ends with 'context_length'
                    for k, v in model_info.items():
                        if isinstance(k, str) and 'context_length' in k and isinstance(v, int):
                            max_ctx = v
                            break

            # 4) As a last resort, check details.context_length
            if max_ctx is None:
                details = info.get('details', {}) if isinstance(info, dict) else {}
                if isinstance(details, dict):
                    v = details.get('context_length')
                    if isinstance(v, int):
                        max_ctx = v

            # Choose the configured num_ctx if present, otherwise the maximum known
            if isinstance(num_ctx, int) and num_ctx > 0:
                return num_ctx
            if isinstance(max_ctx, int) and max_ctx > 0:
                return max_ctx
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

    def register_actual(self, actual_prompt_tokens: int, estimated_prompt_tokens: int) -> None:
        """Update calibration using the actual tokens consumed vs estimated.

        Args:
            actual_prompt_tokens: Tokens Ollama reported for the prompt
            estimated_prompt_tokens: Our heuristic estimate used for the same call
        """
        try:
            if actual_prompt_tokens > 0 and estimated_prompt_tokens > 0:
                ratio = actual_prompt_tokens / max(1, estimated_prompt_tokens)
                # Clamp ratio to avoid outliers
                ratio = max(0.5, min(2.0, ratio))
                self._calibration_factor = (1 - self._alpha) * self._calibration_factor + self._alpha * ratio
        except Exception:
            # Never let calibration errors break chat
            pass
    
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
        # Apply calibration factor (rounded)
        estimated_tokens = int(max(0, round(estimated_tokens * self._calibration_factor)))
        # Optional future calibration hook: could apply dynamic scaling factor
        # based on prompt_eval_count averages stored externally.
        
        return ContextUsage(
            current_tokens=estimated_tokens,
            max_tokens=self.context_window_size
        )
