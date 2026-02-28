"""Context window calculation utilities."""

import re
import requests
from typing import List, Dict, Any
from models import ContextUsage


class ContextCalculator:
    """Handles context window size calculations and tracking with calibration."""

    def __init__(
        self,
        default_size: int = 8192,
        base_url: str = "http://localhost:8080",
        model: str = None,
    ):
        """Initialize context calculator.

        Args:
            default_size: Default context window size
            base_url: llama.cpp server base URL
            model: Model name (deprecated, kept for backwards compatibility)
        """
        self.base_url = base_url
        self.context_window_size = self._get_model_context_size(default_size)
        # Calibration factor to align heuristic estimates with actual prompt_eval_count
        self._calibration_factor: float = 1.0
        # Exponential moving average smoothing for stability
        self._alpha: float = 0.3

    def _get_model_context_size(self, default_size: int) -> int:
        """Get model's context window size from llama.cpp server.

        Args:
            default_size: Default size if unable to determine

        Returns:
            Context window size in tokens
        """
        try:
            # Try llama.cpp server's /props endpoint for default generation settings
            url = f"{self.base_url}/props"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                info = response.json()
                # /props returns default_generation_settings with n_ctx
                if isinstance(info, dict):
                    settings = info.get("default_generation_settings", {})
                    if isinstance(settings, dict):
                        ctx = settings.get("n_ctx")
                        if isinstance(ctx, int) and ctx > 0:
                            return ctx

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

    def register_actual(
        self, actual_prompt_tokens: int, estimated_prompt_tokens: int
    ) -> None:
        """Update calibration using the actual tokens consumed vs estimated.

        Args:
            actual_prompt_tokens: Tokens the LLM reported for the prompt
            estimated_prompt_tokens: Our heuristic estimate used for the same call
        """
        try:
            if actual_prompt_tokens > 0 and estimated_prompt_tokens > 0:
                ratio = actual_prompt_tokens / max(1, estimated_prompt_tokens)
                # Clamp ratio to avoid outliers
                ratio = max(0.5, min(2.0, ratio))
                self._calibration_factor = (
                    1 - self._alpha
                ) * self._calibration_factor + self._alpha * ratio
        except Exception:
            # Never let calibration errors break chat
            pass

    def calculate_usage(
        self, messages: List[Dict[str, Any]], system_prompt: str = ""
    ) -> ContextUsage:
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
                total_chars += len(str(msg.get("content", "")))
                if "tool_calls" in msg:
                    total_chars += len(str(msg["tool_calls"]))

        estimated_tokens = self.estimate_tokens(str(total_chars))
        # Apply calibration factor (rounded)
        estimated_tokens = int(
            max(0, round(estimated_tokens * self._calibration_factor))
        )

        return ContextUsage(
            current_tokens=estimated_tokens, max_tokens=self.context_window_size
        )
