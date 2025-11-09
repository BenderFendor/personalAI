"""Command-line interface for the Personal AI Chatbot.

Uses Rich library for all rendering and input, with KeyboardHandler
for special key detection (Ctrl-b sidebar, Ctrl-n new session).
"""

from __future__ import annotations

from chat import ChatBot
from utils.sidebar import ChatSidebar
from utils.keyboard import KeyboardHandler
from rich.panel import Panel
from rich.markup import escape
from rich.markdown import Markdown
from rich.text import Text
from typing import Optional


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
        
        self.console.print("[dim]Press Ctrl-b for sidebar, Ctrl-n for new session, /help for commands.[/dim]")

        while True:
            try:
                # Simple input with Rich
                user_input = self.console.input("[bold cyan]You:[/bold cyan] ")
                user_text = user_input.strip()

                if not user_text:
                    continue

                # Commands start with '/'
                if user_text.startswith('/'):
                    if self._handle_command(user_text):
                        break
                    continue

                # Normal chat message
                self.chatbot.chat(user_text)

            except KeyboardInterrupt:
                self._exit_chat()
                break
            except EOFError:
                self._exit_chat()
                break
            except Exception as e:
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
            if self.chatbot.save_chat_log():
                self.console.print(
                    f"[green]Chat log saved to chat_logs/chat_{self.chatbot.current_session}.md[/green]"
                )
            else:
                self.console.print("[yellow]No log saved (empty session).[/yellow]")

        elif command == '/clear':
            self.chatbot.clear_history()
            self.console.print("[green]Chat history cleared![/green]")

        elif command == '/new':
            # Start a new session (saving current if it has content)
            wrote = self.chatbot.save_chat_log()
            self.chatbot.start_new_session(save_current=False)
            if wrote:
                self.console.print("[green]Previous session saved. New session started.[/green]")
            else:
                self.console.print("[green]New session started (previous was empty).[/green]")
            self.console.print(f"[dim]Session ID: {self.chatbot.current_session}[/dim]")
        
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
        
        elif command == '/toggle-chunk-previews':
            enabled = self.chatbot.config.toggle('show_chunk_previews')
            self.chatbot.config.save()
            status = "shown" if enabled else "hidden"
            self.console.print(f"[green]Chunk previews will be {status} during indexing[/green]")
        
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

        elif command == '/rag-hard-delete':
            self._rag_hard_delete()
        
        else:
            self.console.print("[red]Unknown command. Type /help for available commands.[/red]")
        
        return False
    
    def _show_sidebar(self) -> None:
        """Show sidebar with chat history navigation."""
        self.console.clear()
        self.sidebar.render()
        self.console.print("[dim]Use ↑/↓ to navigate, Enter to open, Esc to close.[/dim]")
        
        # Navigation loop using KeyboardHandler
        while True:
            key = self.keyboard.get_key()
            if not key:
                break
            
            if key == self.keyboard.UP:
                self.sidebar.move_up()
                self.console.clear()
                self.sidebar.render()
                self.console.print("[dim]Use ↑/↓ to navigate, Enter to open, Esc to close.[/dim]")
            
            elif key == self.keyboard.DOWN:
                self.sidebar.move_down()
                self.console.clear()
                self.sidebar.render()
                self.console.print("[dim]Use ↑/↓ to navigate, Enter to open, Esc to close.[/dim]")
            
            elif key == self.keyboard.ENTER:
                session = self.sidebar.get_selected_session()
                if session:
                    self._load_session(session)
                break
            
            elif key == self.keyboard.ESC or key == self.keyboard.CTRL_C:
                break
        
        # Clear and return to chat
        self.console.clear()
        self.console.print("[dim]Sidebar closed. Continue chatting...[/dim]")
    
    def _load_session(self, session: dict) -> None:
        """Load a previous chat session.
        
        Args:
            session: Session dictionary with path and metadata
        """
        try:
            with open(session['path'], 'r', encoding='utf-8') as f:
                content = f.read()

            # Show full markdown with scrolling so large logs are navigable.
            title = "Previous Chat Session"
            self._view_markdown_scrollable(content, title, session['datetime'])
            
        except Exception as e:
            # Use style instead of inline markup when printing dynamic error text.
            self.console.print(f"Error loading session: {escape(str(e))}", style="red")
    
    def _view_markdown_scrollable(self, markdown_text: str, title: str, dt) -> None:
        """Render markdown with Rich and allow scrolling through the content.
        
        This renders the markdown to ANSI once, then shows a viewport that can
        be scrolled with the keyboard. This keeps the header visible and avoids
        re-rendering on every key press for performance.
        """
        # Render markdown to ANSI using a capture buffer so we can paginate it.
        md = Markdown(markdown_text)
        with self.console.capture() as cap:
            # Use a panel header during capture so wrapping matches terminal width.
            self.console.print(Panel(md, title=f"Chat Log – {dt.strftime('%Y%m%d_%H%M%S')}", border_style="green"))
        ansi_render = cap.get()

        lines = ansi_render.splitlines()
        total = len(lines)

        # Compute viewport height: reserve space for borders/help on redraws
        term_height = max(10, self.console.size.height)
        viewport = max(3, term_height - 4)
        offset = 0

        def redraw() -> None:
            self.console.clear()
            start = offset
            end = min(total, start + viewport)
            slice_text = "\n".join(lines[start:end])
            subtitle = f"[dim]{start + 1}-{end} of {total}  |  ↑/↓: Scroll  |  Enter/Esc: Close  |  Ctrl+]: Toggle history[/dim]"
            self.console.print(Panel(Text.from_ansi(slice_text), title=title, subtitle=subtitle, border_style="green"))

        redraw()
        while True:
            key = self.keyboard.get_key()
            if not key:
                break
            if key == self.keyboard.UP:
                if offset > 0:
                    offset -= 1
                    redraw()
            elif key == self.keyboard.DOWN:
                if offset < max(0, total - viewport):
                    offset += 1
                    redraw()
            elif key in (self.keyboard.ENTER, self.keyboard.ESC) or self.keyboard.is_ctrl_tab(key) or key == self.keyboard.CTRL_C:
                break

    def _exit_chat(self) -> None:
        """Exit chat and save log."""
        self.console.print("\n[yellow]Ending session...[/yellow]")
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

    def _rag_hard_delete(self) -> None:
        """Hard delete the RAG collection (drop & recreate)."""
        try:
            confirm = self.console.input("[yellow]This will DROP and recreate the vector collection. Type 'DELETE' to confirm: [/yellow]")
            if confirm.strip().upper() == 'DELETE':
                self.chatbot.rag_hard_delete()
                self.console.print("[green]RAG collection dropped & recreated (empty).[/green]")
            else:
                self.console.print("[dim]Cancelled.[/dim]")
        except Exception as e:
            self.console.print(f"[red]Error performing hard delete: {escape(str(e))}[/red]")
