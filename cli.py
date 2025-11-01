"""Command-line interface for the Personal AI Chatbot."""

import sys
import select
from chat import ChatBot
from utils.keyboard import KeyboardHandler
from utils.sidebar import ChatSidebar
from rich.panel import Panel
from rich.markup import escape


class ChatCLI:
    """Interactive command-line interface for the chatbot."""
    
    def __init__(self, chatbot: ChatBot):
        """Initialize CLI.
        
        Args:
            chatbot: ChatBot instance
        """
        self.chatbot = chatbot
        self.console = chatbot.console
        self.display = chatbot.display
        self.sidebar = ChatSidebar(self.console)
        self.keyboard = KeyboardHandler()
    
    def run(self) -> None:
        """Run the interactive chat loop."""
        # Display welcome
        self.display.display_welcome_banner(self.chatbot.config.get_all())
        self.display.display_help()
        
        while True:
            try:
                user_input = self.console.input("\n[bold blue]You:[/bold blue] ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if self._handle_command(user_input):
                        break  # Exit requested
                    continue
                
                # Process chat message
                self.chatbot.chat(user_input)
                
            except KeyboardInterrupt:
                self._exit_chat()
                break
            except Exception as e:
                self.console.print(f"\n[red]Unexpected error: {escape(str(e))}[/red]")
    
    def _handle_command(self, command: str) -> bool:
        """Handle slash commands.
        
        Args:
            command: Command string
            
        Returns:
            True if exit requested, False otherwise
        """
        if command in ['/quit', '/exit']:
            self._exit_chat()
            return True
        
        elif command == '/save':
            self.chatbot.save_chat_log()
            self.console.print(
                f"[green]Chat log saved to chat_logs/chat_{self.chatbot.current_session}.md[/green]"
            )
        
        elif command == '/clear':
            self.chatbot.clear_history()
            self.console.print("[green]Chat history cleared![/green]")
        
        elif command == '/config':
            self.display.display_config(self.chatbot.config.get_all())
        
        elif command == '/context':
            context_usage = self.chatbot.get_context_usage()
            self.display.display_context_info(
                context_usage,
                self.chatbot.config.get_all(),
                len(self.chatbot.messages)
            )
        
        elif command == '/toggle-tools':
            enabled = self.chatbot.config.toggle('tools_enabled')
            self.chatbot.config.save()
            status = "enabled" if enabled else "disabled"
            self.console.print(f"[green]Tools {status}[/green]")
        
        elif command == '/toggle-thinking':
            enabled = self.chatbot.config.toggle('show_thinking')
            self.chatbot.config.save()
            status = "shown" if enabled else "hidden"
            self.console.print(f"[green]Thinking process will be {status}[/green]")
        
        elif command == '/toggle-markdown':
            enabled = self.chatbot.config.toggle('markdown_rendering')
            self.chatbot.config.save()
            status = "enabled" if enabled else "disabled"
            self.console.print(f"[green]Markdown rendering {status}[/green]")
        
        elif command == '/help':
            self.display.display_help()
        
        elif command == '/history':
            self._show_sidebar()
        
        else:
            self.console.print("[red]Unknown command. Type /help for available commands.[/red]")
        
        return False
    
    def _show_sidebar(self) -> None:
        """Show sidebar with chat history navigation."""
        self.console.clear()
        self.sidebar.render()
        
        # Navigation loop
        while True:
            key = self.keyboard.get_key()
            
            if not key:
                break
            
            # Handle navigation keys
            if key == self.keyboard.UP:
                self.sidebar.move_up()
                self.console.clear()
                self.sidebar.render()
            
            elif key == self.keyboard.DOWN:
                self.sidebar.move_down()
                self.console.clear()
                self.sidebar.render()
            
            elif key == self.keyboard.ENTER:
                # Load selected session
                session = self.sidebar.get_selected_session()
                if session:
                    self.console.clear()
                    self._load_session(session)
                break
            
            elif key == self.keyboard.ESC or self.keyboard.is_ctrl_tab(key):
                # Close sidebar
                self.console.clear()
                break
            
            elif key == self.keyboard.CTRL_C:
                self.console.clear()
                break
        
        # Redisplay the interface
        self.console.print("\n[dim]Press Ctrl+] to open chat history[/dim]")
    
    def _load_session(self, session: dict) -> None:
        """Load a previous chat session.
        
        Args:
            session: Session dictionary with path and metadata
        """
        try:
            with open(session['path'], 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.console.print(Panel(
                f"[green]Loaded session from {session['datetime'].strftime('%Y-%m-%d %H:%M')}[/green]\n\n"
                f"[dim]{content[:500]}...[/dim]",
                title="Previous Chat Session",
                border_style="green"
            ))
            
            self.console.print("\n[yellow]Note: This is a read-only view. Start chatting to begin a new session.[/yellow]")
            
        except Exception as e:
            self.console.print(f"[red]Error loading session: {escape(str(e))}[/red]")
    
    def _exit_chat(self) -> None:
        """Exit chat and save log."""
        self.console.print("\n[yellow]Saving chat log...[/yellow]")
        self.chatbot.save_chat_log()
        self.console.print("[green]Goodbye![/green]")
