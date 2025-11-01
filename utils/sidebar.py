"""Sidebar UI component for viewing and switching chat sessions."""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text


class ChatSidebar:
    """Sidebar for viewing and switching between chat sessions."""
    
    def __init__(self, console: Console, chat_logs_dir: str = "chat_logs"):
        """Initialize sidebar.
        
        Args:
            console: Rich console instance
            chat_logs_dir: Directory containing chat logs
        """
        self.console = console
        self.chat_logs_dir = Path(chat_logs_dir)
        self.selected_index = 0
        self.sessions: List[Dict] = []
    
    def load_sessions(self) -> None:
        """Load all available chat sessions from disk."""
        if not self.chat_logs_dir.exists():
            self.sessions = []
            return
        
        sessions = []
        for log_file in sorted(self.chat_logs_dir.glob("chat_*.md"), reverse=True):
            # Parse filename: chat_YYYYMMDD_HHMMSS.md
            filename = log_file.name
            try:
                session_id = filename.replace("chat_", "").replace(".md", "")
                date_str = session_id.split("_")[0]
                time_str = session_id.split("_")[1]
                
                # Parse date and time
                dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                
                # Read first few lines to get a preview
                preview = self._get_session_preview(log_file)
                
                sessions.append({
                    'id': session_id,
                    'filename': filename,
                    'path': str(log_file),
                    'datetime': dt,
                    'preview': preview
                })
            except Exception as e:
                # Skip malformed files
                continue
        
        self.sessions = sessions
        self.selected_index = min(self.selected_index, len(self.sessions) - 1)
    
    def _get_session_preview(self, log_file: Path) -> str:
        """Get a preview of the chat session.
        
        Args:
            log_file: Path to log file
            
        Returns:
            Preview text
        """
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Find first user message
                for i, line in enumerate(lines):
                    if line.startswith("## USER"):
                        # Get the next non-empty line as preview
                        for j in range(i + 1, min(i + 5, len(lines))):
                            preview_line = lines[j].strip()
                            if preview_line and not preview_line.startswith("---"):
                                return preview_line[:60] + ("..." if len(preview_line) > 60 else "")
                
                return "No messages"
        except:
            return "Error reading file"
    
    def render(self) -> None:
        """Render the sidebar."""
        self.load_sessions()
        
        if not self.sessions:
            self.console.print(Panel(
                "[yellow]No chat sessions found[/yellow]",
                title="Chat History",
                border_style="blue"
            ))
            return
        
        # Create table
        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
        table.add_column("", width=2)
        table.add_column("Date & Time", style="dim", width=20)
        table.add_column("Preview", width=50)
        
        # Add rows
        for i, session in enumerate(self.sessions):
            is_selected = (i == self.selected_index)
            marker = "▶" if is_selected else " "
            
            date_str = session['datetime'].strftime("%Y-%m-%d %H:%M")
            preview = session['preview']
            
            style = "bold green" if is_selected else ""
            
            table.add_row(
                f"[{style}]{marker}[/{style}]",
                f"[{style}]{date_str}[/{style}]",
                f"[{style}]{preview}[/{style}]"
            )
        
        # Create panel
        panel = Panel(
            table,
            title="[bold cyan]Chat History[/bold cyan]",
            subtitle="[dim]↑/↓: Navigate | Enter: Load | Esc: Close | Ctrl+]: Toggle[/dim]",
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def move_up(self) -> None:
        """Move selection up."""
        if self.sessions and self.selected_index > 0:
            self.selected_index -= 1
    
    def move_down(self) -> None:
        """Move selection down."""
        if self.sessions and self.selected_index < len(self.sessions) - 1:
            self.selected_index += 1
    
    def get_selected_session(self) -> Optional[Dict]:
        """Get the currently selected session.
        
        Returns:
            Session dict or None if no selection
        """
        if not self.sessions or self.selected_index >= len(self.sessions):
            return None
        return self.sessions[self.selected_index]
    
    def display_help(self) -> None:
        """Display sidebar help text."""
        help_text = Text()
        help_text.append("Keyboard Shortcuts:\n\n", style="bold cyan")
        help_text.append("  Ctrl+]      ", style="bold yellow")
        help_text.append("Toggle chat history sidebar\n")
        help_text.append("  ↑/↓         ", style="bold yellow")
        help_text.append("Navigate through sessions\n")
        help_text.append("  Enter       ", style="bold yellow")
        help_text.append("Load selected session\n")
        help_text.append("  Esc         ", style="bold yellow")
        help_text.append("Close sidebar\n")
        
        self.console.print(Panel(help_text, title="Chat History", border_style="cyan"))
