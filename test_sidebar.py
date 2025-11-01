#!/usr/bin/env python3
"""Test script for the chat history sidebar feature."""

from rich.console import Console
from utils.sidebar import ChatSidebar


def main():
    """Test the sidebar functionality."""
    console = Console()
    sidebar = ChatSidebar(console)
    
    console.print("[bold cyan]Testing Chat History Sidebar[/bold cyan]\n")
    
    # Display help
    sidebar.display_help()
    
    console.print("\n[yellow]Loading chat sessions...[/yellow]\n")
    
    # Render the sidebar
    sidebar.render()
    
    console.print("\n[green]✓ Sidebar rendered successfully![/green]")
    console.print("[dim]In the actual app, you can navigate with ↑/↓ and select with Enter[/dim]")


if __name__ == "__main__":
    main()
