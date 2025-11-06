"""Configuration management for the Personal AI Chatbot."""

import json
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Manages loading, saving, and accessing chatbot configuration."""
    
    DEFAULT_CONFIG = {
        "model": "qwen3",
        "temperature": 0.7,
        "web_search_enabled": True,
        "max_search_results": 20,
        "thinking_enabled": True,
        "show_thinking": True,
        "tools_enabled": True,
        "markdown_rendering": True,
        "max_tool_iterations": 5,
        "context_window_size": 8192,
        "auto_fetch_urls": True,
        "auto_fetch_tools": ["news_search", "web_search"],
        "auto_fetch_threshold": 0.6,
        # LLM provider settings
        "llm_provider": "ollama",  # "ollama" or "gemini"
        "gemini_model": "gemini-flash-latest",
        "gemini_safety_minimal": True
    }
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default.
        
        Returns:
            Configuration dictionary
        """
        if self.config_path.exists():
            try:
                raw = self.config_path.read_text().strip()
                if not raw:
                    raise ValueError("Empty config file")
                file_cfg = json.loads(raw)
            except Exception:
                # Backup corrupt file
                try:
                    backup_path = self.config_path.with_suffix('.corrupt')
                    self.config_path.rename(backup_path)
                except Exception:
                    pass
                file_cfg = {}

            merged = {**self.DEFAULT_CONFIG, **file_cfg}
            # Back-compat: if nested rag.enabled is present, mirror to top-level rag_enabled
            if isinstance(file_cfg.get("rag"), dict):
                rag_enabled_nested = file_cfg.get("rag", {}).get("enabled")
                if isinstance(rag_enabled_nested, bool):
                    merged.setdefault("rag_enabled", rag_enabled_nested)
            return merged
        else:
            # Create default config on first run without relying on self._config
            default_cfg = self.DEFAULT_CONFIG.copy()
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_cfg, f, indent=2)
            except Exception:
                # If writing fails, still return defaults so app can run
                pass
            return default_cfg
    
    def save(self) -> None:
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
    
    def toggle(self, key: str) -> bool:
        """Toggle boolean configuration value.
        
        Args:
            key: Configuration key
            
        Returns:
            New value after toggle
        """
        self._config[key] = not self._config.get(key, False)
        return self._config[key]
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()
    
    @property
    def model(self) -> str:
        """Get current model name."""
        return self._config['model']
    
    @property
    def temperature(self) -> float:
        """Get current temperature setting."""
        return self._config['temperature']
    
    @property
    def tools_enabled(self) -> bool:
        """Check if tools are enabled."""
        return self._config['tools_enabled']
    
    @property
    def show_thinking(self) -> bool:
        """Check if thinking display is enabled."""
        return self._config['show_thinking']
    
    @property
    def markdown_rendering(self) -> bool:
        """Check if markdown rendering is enabled."""
        return self._config['markdown_rendering']
    
    @property
    def max_tool_iterations(self) -> int:
        """Get maximum tool iterations."""
        return self._config['max_tool_iterations']

    # Provider-specific convenience accessors
    @property
    def llm_provider(self) -> str:
        return self._config.get("llm_provider", "ollama")

    @property
    def gemini_model(self) -> str:
        return self._config.get("gemini_model", "gemini-flash-latest")

    @property
    def gemini_api_key(self) -> str:
        # Prefer GOOGLE_API_KEY; allow fallback to GOOGLE_GENERATIVE_AI_API_KEY
        import os
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY", "")

    @property
    def gemini_safety_minimal(self) -> bool:
        return bool(self._config.get("gemini_safety_minimal", True))
