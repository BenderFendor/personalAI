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
                # Avoid markup parsing on arbitrary exception text by escaping and
                # using the style argument instead of inline markup tags.
                self.console.print(f"\nUnexpected error: {escape(str(e))}", style="red")
    
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
        
        elif command == '/rag-status':
            self._rag_status()
        
        elif command.startswith('/rag-index '):
            file_path = command[11:].strip()
            self._rag_index(file_path)
        
        elif command.startswith('/rag-search '):
            query = command[12:].strip()
            self._rag_search(query)
        
        elif command == '/rag-clear':
            self._rag_clear()
        
        elif command == '/rag-rebuild':
            self._rag_rebuild()
        
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
            
            # Escape file content before rendering inside the panel to avoid
            # accidental markup interpretation from the saved chat text.
            loaded_header = f"Loaded session from {session['datetime'].strftime('%Y-%m-%d %H:%M')}"
            snippet = escape(content[:500]) + "..."
            self.console.print(Panel(
                f"{loaded_header}\n\n{snippet}",
                title="Previous Chat Session",
                border_style="green"
            ))

            self.console.print("\nNote: This is a read-only view. Start chatting to begin a new session.", style="yellow")
            
        except Exception as e:
            # Use style instead of inline markup when printing dynamic error text.
            self.console.print(f"Error loading session: {escape(str(e))}", style="red")
    
    def _exit_chat(self) -> None:
        """Exit chat and save log."""
        self.console.print("\n[yellow]Saving chat log...[/yellow]")
        self.chatbot.save_chat_log()
        self.console.print("[green]Goodbye![/green]")
    
    def _rag_status(self) -> None:
        """Display RAG system status."""
        try:
            status = self.chatbot.get_rag_status()
            self.console.print(f"\n[bold cyan]RAG System Status[/bold cyan]")
            self.console.print(f"Enabled: [green]{status['enabled']}[/green]")
            self.console.print(f"Documents indexed: [yellow]{status['doc_count']}[/yellow]")
            self.console.print(f"Collection: {status['collection']}")
            self.console.print(f"Embedding model: {status['embedding_model']}")
        except Exception as e:
            self.console.print(f"[red]Error getting RAG status: {escape(str(e))}[/red]")
    
    def _rag_index(self, file_path: str) -> None:
        """Index a file into the RAG knowledge base."""
        try:
            self.console.print(f"[yellow]Indexing {file_path}...[/yellow]")
            count = self.chatbot.rag_index_file(file_path)
            self.console.print(f"[green]Successfully indexed {count} chunks from {file_path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error indexing file: {escape(str(e))}[/red]")
    
    def _rag_search(self, query: str) -> None:
        """Test RAG retrieval without generating response."""
        try:
            results = self.chatbot.rag_search(query)
            self.console.print(f"\n[bold cyan]RAG Search Results for: {query}[/bold cyan]\n")
            for i, doc in enumerate(results, 1):
                self.console.print(f"[yellow]Result {i}[/yellow] (Similarity: {doc['similarity']:.3f})")
                self.console.print(Panel(escape(doc['content'][:200] + '...'), border_style="dim"))
        except Exception as e:
            self.console.print(f"[red]Error searching: {escape(str(e))}[/red]")
    
    def _rag_clear(self) -> None:
        """Clear the RAG vector database."""
        try:
            confirm = self.console.input("[yellow]Are you sure you want to clear the vector database? (yes/no): [/yellow]")
            if confirm.lower() == 'yes':
                self.chatbot.rag_clear()
                self.console.print("[green]Vector database cleared![/green]")
            else:
                self.console.print("[dim]Cancelled.[/dim]")
        except Exception as e:
            self.console.print(f"[red]Error clearing database: {escape(str(e))}[/red]")
    
    def _rag_rebuild(self) -> None:
        """Rebuild RAG index from source documents."""
        try:
            self.console.print("[yellow]Rebuilding RAG index...[/yellow]")
            count = self.chatbot.rag_rebuild()
            self.console.print(f"[green]Rebuilt index with {count} documents![/green]")
        except Exception as e:
            self.console.print(f"[red]Error rebuilding index: {escape(str(e))}[/red]")
