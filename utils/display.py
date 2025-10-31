"""Display utilities for Rich console output."""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from models import ContextUsage


class DisplayHelper:
    """Handles Rich console display formatting."""
    
    def __init__(self, console: Console):
        """Initialize display helper.
        
        Args:
            console: Rich console instance
        """
        self.console = console
    
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
  [cyan]/toggle-tools[/cyan] - Toggle tool use on/off
  [cyan]/toggle-thinking[/cyan] - Toggle thinking display on/off
  [cyan]/toggle-markdown[/cyan] - Toggle markdown rendering
  [cyan]/help[/cyan] - Show this help message
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
