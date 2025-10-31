"""Command-line interface for the Personal AI Chatbot."""

from chatbot import ChatBot


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
                self.console.print(f"\n[red]Unexpected error: {e}[/red]")
    
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
        
        else:
            self.console.print("[red]Unknown command. Type /help for available commands.[/red]")
        
        return False
    
    def _exit_chat(self) -> None:
        """Exit chat and save log."""
        self.console.print("\n[yellow]Saving chat log...[/yellow]")
        self.chatbot.save_chat_log()
        self.console.print("[green]Goodbye![/green]")
