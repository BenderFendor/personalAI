"""Data models for the Personal AI Chatbot."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class Message:
    """Represents a chat message."""
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    thinking: Optional[str] = None
    sources: Optional[List[Dict[str, str]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp
        }
        if self.thinking:
            data['thinking'] = self.thinking
        if self.sources:
            data['sources'] = self.sources
        if self.tool_calls:
            data['tool_calls'] = self.tool_calls
        return data
    
    def to_ollama_format(self) -> Dict[str, Any]:
        """Convert to Ollama API format."""
        data = {
            'role': self.role,
            'content': self.content
        }
        if self.tool_calls:
            data['tool_calls'] = self.tool_calls
        return data


@dataclass
class ContextUsage:
    """Represents context window usage information."""
    current_tokens: int
    max_tokens: int
    
    @property
    def percentage(self) -> float:
        """Calculate percentage of context used."""
        return (self.current_tokens / self.max_tokens) * 100 if self.max_tokens > 0 else 0
    
    @property
    def remaining_tokens(self) -> int:
        """Calculate remaining tokens."""
        return self.max_tokens - self.current_tokens
    
    @property
    def is_high_usage(self) -> bool:
        """Check if context usage is high (>75%)."""
        return self.percentage > 75
    
    @property
    def color(self) -> str:
        """Get color based on usage level."""
        if self.percentage < 50:
            return "green"
        elif self.percentage < 75:
            return "yellow"
        else:
            return "red"


@dataclass
class SearchResult:
    """Represents a web search result."""
    title: str
    url: str
    snippet: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet
        }


@dataclass
class ToolCall:
    """Represents a tool call request."""
    name: str
    arguments: Dict[str, Any]
    
    @classmethod
    def from_ollama(cls, tool_call: Dict[str, Any]) -> 'ToolCall':
        """Create from Ollama tool call format."""
        return cls(
            name=tool_call['function']['name'],
            arguments=tool_call['function']['arguments']
        )
