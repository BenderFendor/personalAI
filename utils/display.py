"""Display utilities for Rich console output."""

import json
import re
from typing import List, Dict, Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from models import ContextUsage

try:
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


class DisplayHelper:
    """Handles Rich console display formatting."""
    
    def __init__(self, console: Console):
        """Initialize display helper.
        
        Args:
            console: Rich console instance
        """
        self.console = console
        self._semantic_model = None
        if SEMANTIC_AVAILABLE:
            try:
                self._semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not load semantic model: {e}[/yellow]")
                self._semantic_model = None
    
    def extract_and_rank_urls(
        self, 
        search_results: str, 
        query: str,
        threshold: float = 0.6
    ) -> List[Dict[str, str]]:
        """Extract URLs from search results and rank by semantic relevance.
        
        Args:
            search_results: Raw search results text
            query: Original search query
            threshold: Relevance score threshold (0-1)
            
        Returns:
            List of dicts with 'url', 'title', 'score' keys, sorted by score descending
        """
        urls_data = []
        
        # Extract URLs and associated titles from search results
        # Pattern: "N. Title\n   Source/URL: ...\n"
        url_pattern = r'(?:URL|url):\s*(https?://[^\n]+)'
        title_pattern = r'^\d+\.\s+(.+?)(?:\n|$)'
        
        for match in re.finditer(url_pattern, search_results):
            url = match.group(1).strip()
            if url and url.startswith(('http://', 'https://')):
                # Find the preceding title
                start_pos = max(0, match.start() - 200)
                preceding_text = search_results[start_pos:match.start()]
                title_match = re.search(r'(\d+\.\s+.+?)(?:\n|$)', preceding_text[-150:] if len(preceding_text) > 150 else preceding_text)
                title = title_match.group(1) if title_match else url
                
                urls_data.append({
                    'url': url,
                    'title': title,
                    'score': 0.0
                })
        
        # Remove duplicates
        unique_urls = {item['url']: item for item in urls_data}
        urls_data = list(unique_urls.values())
        
        # Score by semantic similarity if model available
        if self._semantic_model and urls_data:
            try:
                query_embedding = self._semantic_model.encode(query, convert_to_tensor=True)
                
                for item in urls_data:
                    title_text = item['title']
                    # Remove numbering if present
                    title_text = re.sub(r'^\d+\.\s+', '', title_text)
                    
                    title_embedding = self._semantic_model.encode(title_text, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(query_embedding, title_embedding).item()
                    item['score'] = max(0.0, similarity)  # Clamp to 0-1
            except Exception as e:
                self.console.print(f"[yellow]Warning: Semantic scoring failed: {e}[/yellow]")
                # Fall back to uniform scoring
                for item in urls_data:
                    item['score'] = 1.0
        else:
            # Fallback: uniform scores if no semantic model
            for item in urls_data:
                item['score'] = 1.0
        
        # Filter by threshold and sort by score
        filtered_urls = [item for item in urls_data if item['score'] >= threshold]
        filtered_urls.sort(key=lambda x: x['score'], reverse=True)
        
        return filtered_urls
    
    def display_context_bar(self, context_usage: ContextUsage) -> str:
        """Create context window usage display string.
        
        Args:
            context_usage: ContextUsage object
            
        Returns:
            Formatted context usage string
        """
        percentage = context_usage.percentage
        current = context_usage.current_tokens
        max_tokens = context_usage.max_tokens
        color = context_usage.color
        
        # Create progress bar
        bar_width = 20
        filled = int((percentage / 100) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        return f"[{color}]Context: {current}/{max_tokens} tokens ({percentage:.1f}%) [{bar}][/{color}]"
    
    def display_welcome_banner(self, config: dict) -> None:
        """Display welcome banner.
        
        Args:
            config: Configuration dictionary
        """
        welcome_panel = Panel.fit(
            "[bold cyan]Simple Personal AI Chatbot[/bold cyan]\n"
            f"[yellow]Model:[/yellow] {config.get('model', 'Unknown')}\n"
            f"[yellow]Tools:[/yellow] {'Enabled' if config.get('tools_enabled') else 'Disabled'}\n"
            f"[yellow]Thinking:[/yellow] {'Enabled' if config.get('thinking_enabled') else 'Disabled'}",
            title="[bold]Welcome[/bold]",
            border_style="green"
        )
        self.console.print(welcome_panel)
    
    def display_help(self) -> None:
        """Display help information."""
        help_text = """
[bold]Commands:[/bold]
  [cyan]/quit or /exit[/cyan] - Exit the chat
  [cyan]/save[/cyan] - Save chat log
  [cyan]/clear[/cyan] - Clear chat history
  [cyan]/config[/cyan] - Show configuration
  [cyan]/context[/cyan] - Show context window usage
  [cyan]/history[/cyan] - View past chat sessions (or press Ctrl+])
  [cyan]/toggle-tools[/cyan] - Toggle tool use on/off
  [cyan]/toggle-thinking[/cyan] - Toggle thinking display on/off
  [cyan]/toggle-markdown[/cyan] - Toggle markdown rendering
  [cyan]/help[/cyan] - Show this help message

[bold]Keyboard Shortcuts:[/bold]
  [cyan]Ctrl+][/cyan] - Toggle chat history sidebar
  [cyan]Ctrl+C[/cyan] - Interrupt current operation
"""
        self.console.print(Panel(help_text, border_style="blue"))
    
    def display_config(self, config: dict) -> None:
        """Display current configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_text = "\n[bold]Current Configuration:[/bold]\n"
        for key, value in config.items():
            config_text += f"  [cyan]{key}:[/cyan] {value}\n"
        self.console.print(Panel(config_text, title="Configuration", border_style="yellow"))
    
    def display_context_info(self, context_usage: ContextUsage, config: dict, message_count: int) -> None:
        """Display context window information.
        
        Args:
            context_usage: ContextUsage object
            config: Configuration dictionary
            message_count: Number of messages in history
        """
        context_info = (
            f"\n[bold]Context Window Status:[/bold]\n\n"
            f"  [cyan]Model:[/cyan] {config.get('model', 'Unknown')}\n"
            f"  [cyan]Current Usage:[/cyan] {context_usage.current_tokens} tokens\n"
            f"  [cyan]Maximum:[/cyan] {context_usage.max_tokens} tokens\n"
            f"  [cyan]Remaining:[/cyan] {context_usage.remaining_tokens} tokens\n"
            f"  [cyan]Percentage:[/cyan] {context_usage.percentage:.1f}%\n\n"
            f"  {self.display_context_bar(context_usage)}\n\n"
            f"  [dim]Messages in history: {message_count}[/dim]\n"
        )
        
        if context_usage.is_high_usage:
            context_info += "\n  [yellow]Warning: Context window is filling up. Consider using /clear to reset.[/yellow]\n"
        
        self.console.print(Panel(context_info, title="Context Window", border_style="cyan"))
    
    def render_markdown(self, text: str) -> None:
        """Render text as markdown.
        
        Args:
            text: Text to render
        """
        md = Markdown(text)
        self.console.print(md)
    
    def show_thinking(self, thinking_text: str, show: bool = True) -> None:
        """Display thinking process.
        
        Args:
            thinking_text: Thinking text to display
            show: Whether to actually show it
        """
        if show and thinking_text:
            self.console.print(f"[dim]{thinking_text}[/dim]", end='')
    
    def show_tool_call(self, tool_name: str, arguments: dict, iteration: int = 1) -> None:
        """Display tool call information.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            iteration: Current iteration number
        """
        import json
        self.console.print(f"\n[bold cyan]>> Tool Call: {tool_name}[/bold cyan]")
        
        requested_iterations = arguments.get('iterations', 1) if tool_name == 'web_search' else 1
        if requested_iterations > 1:
            self.console.print(f"[bold yellow]   Multi-iteration search requested: {requested_iterations} searches planned[/bold yellow]")
        
        self.console.print(f"[dim]   Arguments: {json.dumps(arguments, indent=2)}[/dim]")
        self.console.print("[yellow]   Executing...[/yellow]")
    
    def show_tool_result(self, result: str) -> None:
        """Display abbreviated tool result.
        
        Args:
            result: Tool result text
        """
        result_preview = result[:200] + "..." if len(result) > 200 else result
        self.console.print(f"[dim]   Result: {result_preview}[/dim]")
