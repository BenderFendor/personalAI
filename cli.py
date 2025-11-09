"""Command-line interface for the Personal AI Chatbot.

Refactored to use prompt_toolkit for robust input handling with a
non-deletable 'You:' prompt prefix and global keybindings (Ctrl-b sidebar,
Ctrl-n new session). Rich remains responsible for all output rendering.
"""

from __future__ import annotations

from chat import ChatBot
from utils.sidebar import ChatSidebar
from utils.keyboard import KeyboardHandler
from rich.panel import Panel
from rich.markup import escape
from rich.markdown import Markdown
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.document import Document
from prompt_toolkit.application import run_in_terminal
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
        # prompt_toolkit session & key bindings will be created lazily.
        self._session: Optional[PromptSession] = None
        self._kb = self._create_key_bindings()
        self._sidebar_active = False
        # Retain legacy keyboard handler for scrollable log viewer (run in terminal).
        self.keyboard = KeyboardHandler()
    
    def run(self) -> None:
        """Run the interactive chat loop."""
        # Display welcome
        self.display.display_welcome_banner(self.chatbot.config.get_all())
        self.display.display_help()
        
        # Interactive loop using prompt_toolkit for protected prefix.
        prefix = "You: "  # Non-editable prefix

        # Validator ensures the text always starts with prefix; if user somehow deletes it (e.g. paste anomaly) we restore.
        class PrefixValidator(Validator):
            def validate(self, document: Document) -> None:  # type: ignore[override]
                if not document.text.startswith(prefix):
                    raise ValidationError(message="Prompt prefix missing; will auto-restore.")

        if self._session is None:
            self._session = PromptSession(
                message="",  # We'll inject prefix manually below
                key_bindings=self._kb,
                validator=PrefixValidator(),
                validate_while_typing=False,
            )

        self.console.print("[dim]Press Ctrl-b for sidebar, Ctrl-n for new session, /help for commands.[/dim]")

        while True:
            try:
                with patch_stdout():  # Allow Rich printing during input
                    # Build initial buffer content with prefix if empty or corrupted.
                    buf = self._session.default_buffer
                    if not buf.text.startswith(prefix):
                        buf.text = prefix
                        buf.cursor_position = len(buf.text)

                    # Read line (user editing after prefix). Backspace across prefix blocked via key binding.
                    line = self._session.prompt()

                # Normalize and strip prefix
                if line.startswith(prefix):
                    user_text = line[len(prefix):].strip()
                else:
                    # Fallback: treat entire line as user input
                    user_text = line.strip()

                if not user_text:
                    continue

                # Commands start with '/'
                if user_text.startswith('/'):
                    if self._handle_command(user_text):
                        break
                    continue

                # Normal chat message
                self.chatbot.chat(user_text)

                # Reset buffer to prefix for next turn
                buf.text = prefix
                buf.cursor_position = len(prefix)

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
            self._toggle_sidebar()
        
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
        # Render inside run_in_terminal so ANSI output isn't mangled by the prompt.
        def _render():
            self.console.clear()
            self.sidebar.render()
            self.console.print("[dim]Press Ctrl-b or Esc to return to chat. Use ↑/↓ to navigate, Enter to open.[/dim]")
        run_in_terminal(_render)
        
        # Navigation loop
        # Sidebar is modal; use key bindings to navigate while active.

    def _toggle_sidebar(self) -> None:
        if not self._sidebar_active:
            self._sidebar_active = True
            self._show_sidebar()
        else:
            self._sidebar_active = False
            def _close():
                self.console.clear()
                self.console.print("[dim]Sidebar closed.[/dim]")
                self.console.print("[dim]Press Ctrl-b for sidebar, Ctrl-n for new session, /help for commands.[/dim]")
            run_in_terminal(_close)

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add('c-b')
        def _(event):  # type: ignore
            # Toggle sidebar without losing current buffer content
            self._toggle_sidebar()
            # If closing, ensure buffer has prefix
            buf = self._session.default_buffer if self._session else None
            if buf and not buf.text.startswith('You: '):
                buf.text = 'You: '
                buf.cursor_position = len(buf.text)

        @kb.add('c-n')
        def _(event):  # type: ignore
            # Start new session (save current if dirty)
            wrote = self.chatbot.save_chat_log()
            self.chatbot.start_new_session(save_current=False)
            def _notify():
                self.console.print(
                    "[green]New session started." + (
                        " Previous saved." if wrote else " Previous was empty."),
                )
                self.console.print(f"[dim]Session ID: {self.chatbot.current_session}[/dim]")
            run_in_terminal(_notify)
            buf = self._session.default_buffer if self._session else None
            if buf:
                buf.text = 'You: '
                buf.cursor_position = len(buf.text)

        # Prevent backspacing into prefix: if cursor at or before len(prefix) block deletion
        @kb.add('backspace')
        def _(event):  # type: ignore
            buf = event.current_buffer
            prefix_len = len('You: ')
            if buf.cursor_position <= prefix_len:
                event.app.bell()
            else:
                buf.delete_before_cursor(1)

        # Sidebar navigation keys when active
        @kb.add('up')
        def _(event):  # type: ignore
            if self._sidebar_active:
                self.sidebar.move_up()
                run_in_terminal(lambda: (self.console.clear(), self.sidebar.render(), self.console.print("[dim]Press Ctrl-b or Esc to return to chat. Use ↑/↓ to navigate, Enter to open.[/dim]")))

        @kb.add('down')
        def _(event):  # type: ignore
            if self._sidebar_active:
                self.sidebar.move_down()
                run_in_terminal(lambda: (self.console.clear(), self.sidebar.render(), self.console.print("[dim]Press Ctrl-b or Esc to return to chat. Use ↑/↓ to navigate, Enter to open.[/dim]")))

        @kb.add('enter')
        def _(event):  # type: ignore
            if self._sidebar_active:
                session = self.sidebar.get_selected_session()
                if session:
                    # Suspend prompt and show the markdown viewer which uses raw keyboard
                    run_in_terminal(lambda: self._load_session(session))
                # Close sidebar either way
                self._sidebar_active = False
                run_in_terminal(lambda: (self.console.clear(), self.console.print("[dim]Sidebar closed.[/dim]")))

        @kb.add('escape')
        def _(event):  # type: ignore
            if self._sidebar_active:
                self._sidebar_active = False
                run_in_terminal(lambda: (self.console.clear(), self.console.print("[dim]Sidebar closed.[/dim]")))

        return kb
    
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
