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
        "auto_fetch_threshold": 0.6
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
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                return {**self.DEFAULT_CONFIG, **config}
        else:
            self.save()
            return self.DEFAULT_CONFIG.copy()
    
    def save(self) -> None:
        """Save current configuration to file."""
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
