"""Core chatbot implementation."""

from datetime import datetime
from typing import List, Dict, Any
import json
import ollama
from rich.console import Console
from rich.markup import escape

from models import Message, ContextUsage
from config_manager import ConfigManager
from tools import get_tool_definitions, ToolExecutor
from utils import ContextCalculator, ChatLogger, DisplayHelper


class ChatBot:
    """Core chatbot with AI conversation and tool use capabilities."""
    
    def __init__(self, config_path: str = "config.json", logs_dir: str = "chat_logs"):
        """Initialize the chatbot.
        
        Args:
            config_path: Path to configuration file
            logs_dir: Directory for chat logs
        """
        self.console = Console()
        self.config = ConfigManager(config_path)
        self.context_calc = ContextCalculator(self.config.model)
        self.logger = ChatLogger(logs_dir)
        self.display = DisplayHelper(self.console)
        
        self.messages: List[Message] = []
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tools = get_tool_definitions()
        self.url_cache: Dict[str, str] = {}
        
        # Initialize RAG if enabled
        self.rag_retriever = None
        self.web_search_rag = None
        if self.config.get('rag_enabled', False):
            self._initialize_rag()
        
        # Initialize tool executor (after RAG initialization)
        self.tool_executor = ToolExecutor(self.config.get_all(), self.console, self.web_search_rag)
    
    def _get_system_prompt(self) -> str:
        """Generate the system prompt dynamically.
        
        Returns:
            System prompt string
        """
        auto_fetch_enabled = self.config.get('auto_fetch_urls', True)
        fetch_info = (
            "\n\nAutomatic URL Fetching:\n"
            "- Full article content is automatically fetched for the most relevant search results.\n"
            "- You will receive both the search summary AND the complete article text.\n"
            "- Use this fetched content to provide more detailed and accurate answers.\n"
            "- Do NOT ask the user to fetch URLs - they're already being fetched for you.\n"
        ) if auto_fetch_enabled else ""
        
        rag_info = ""
        if self.config.get('rag_enabled', False) and self.rag_retriever:
            doc_count = self.rag_retriever.store.count()
            rag_info = (
                f"\n\nKnowledge Base (RAG):\n"
                f"- You have access to a knowledge base with {doc_count} indexed documents\n"
                "- When answering questions, relevant context from the knowledge base will be provided\n"
                "- Always cite sources from the knowledge base when using that information\n"
                "- If web search results are indexed, you can retrieve and use that information too\n"
            )
        
        return (
            "You are a helpful AI assistant with access to various tools. "
            f"The current date is {datetime.now().strftime('%Y-%m-%d')}. "
            "\n\nAvailable Search Tools:\n"
            "- web_search: General web search using DuckDuckGo (use for most queries)\n"
            "- news_search: Search recent news articles with date filtering (use for current events, breaking news)\n"
            "- fetch_url_content: Extract full text content from any URL (use to read full articles from search results)\n"
            "\n\nBest Practices:\n"
            "1. Use web_search for general information and research\n"
            "2. Use news_search when the user asks about recent events, news, or current affairs\n"
            "3. The system automatically fetches relevant article content for you\n"
            "4. Chain tools together: search -> use fetched content for synthesis\n"
            "5. If you don't have the context or information use web_search\n"
            "6. Give concise, accurate, and well-sourced answers based on tool results using snippets quotes and excerpts when needed from the sources\n"
            f"{fetch_info}"
            f"{rag_info}"
            "\nThe web_search tool supports an 'iterations' parameter (1-5). "
            "For complex questions requiring multiple searches:\n"
            "- Call web_search(query='...', iterations=3) to request 3 search cycles\n"
            "- After each search, you'll get results and guidance to refine your next search\n"
            "- Each iteration should build on previous results with more specific queries\n"
            "- Use this for deep research, fact-checking, or gathering comprehensive information"
        )
    
    def save_chat_log(self) -> None:
        """Save current chat session to file."""
        self.logger.save_session(
            self.current_session,
            self.messages,
            self.config.get_all()
        )
    
    def clear_history(self) -> None:
        """Clear chat message history."""
        self.messages = []
    
    def get_context_usage(self) -> ContextUsage:
        """Get current context window usage.
        
        Returns:
            ContextUsage object
        """
        ollama_messages = [{'role': 'system', 'content': self._get_system_prompt()}]
        for msg in self.messages:
            ollama_messages.append(msg.to_ollama_format())
        
        return self.context_calc.calculate_usage(
            ollama_messages,
            self._get_system_prompt()
        )
    
    def chat(self, user_input: str) -> str:
        """Process user input and get AI response.
        
        Args:
            user_input: User's message
            
        Returns:
            AI assistant's response
        """
        # Add user message
        user_msg = Message(role='user', content=user_input)
        self.messages.append(user_msg)
        
        # Prepare messages for Ollama
        ollama_messages = [{'role': 'system', 'content': self._get_system_prompt()}]
        for msg in self.messages[:-1]:
            ollama_messages.append(msg.to_ollama_format())
        ollama_messages.append({'role': 'user', 'content': user_input})
        
        # Show context usage
        context_usage = self.context_calc.calculate_usage(ollama_messages, self._get_system_prompt())
        self.console.print(f"\n[dim]{self.display.display_context_bar(context_usage)}[/dim]")
        
        try:
            # Initial response with streaming
            response = self._stream_response(ollama_messages)
            full_thinking, full_response, tool_calls = response
            
            # Agentic tool loop - allow iterative tool use
            max_iterations = self.config.max_tool_iterations
            iteration = 0
            
            while tool_calls and iteration < max_iterations:
                iteration += 1
                self.console.print(f"\n[bold blue]>> Tool Iteration {iteration}[/bold blue]")
                # Add assistant's tool request
                ollama_messages.append({
                    'role': 'assistant',
                    'content': full_response if full_response else '',
                    'tool_calls': tool_calls
                })
                
                # Execute tools
                for tool_call in tool_calls:
                    # Handle both dict and ToolCall object formats
                    if isinstance(tool_call, dict):
                        function_name = tool_call['function']['name']
                        arguments = tool_call['function']['arguments']
                    else:
                        # ToolCall object from Ollama
                        function_name = tool_call.function.name
                        arguments = tool_call.function.arguments
                    
                    self.display.show_tool_call(function_name, arguments, iteration)
                    
                    # Execute tool
                    tool_result = self.tool_executor.execute(function_name, arguments)
                    self.display.show_tool_result(tool_result)
                    
                    # Auto-fetch URLs from search results if applicable
                    tool_result = self._auto_fetch_urls(function_name, tool_result, user_input)
                    
                    # Add result to messages
                    ollama_messages.append({
                        'role': 'tool',
                        'content': tool_result
                    })
                
                # Get next response
                context_usage = self.context_calc.calculate_usage(ollama_messages, self._get_system_prompt())
                self.console.print(f"\n[bold yellow]>> Model Call: {self.config.model}[/bold yellow]")
                self.console.print("[dim]   Processing tool results...[/dim]")
                self.console.print(f"[dim]   {self.display.display_context_bar(context_usage)}[/dim]")
                
                response = self._stream_response(ollama_messages)
                iteration_thinking, full_response, tool_calls = response
                full_thinking += iteration_thinking
                
                if not tool_calls:
                    self.console.print("\n[bold green]Final Response:[/bold green]")
                    break
            
            # Check max iterations
            if iteration >= max_iterations and tool_calls:
                self.console.print("\n[yellow]Warning: Maximum tool iterations reached.[/yellow]")
                self.console.print("[bold green]Partial Response:[/bold green]")
            
            # Render response
            if self.config.markdown_rendering and full_response:
                self.display.render_markdown(full_response)
            elif full_response:
                self.console.print(full_response)
            
            # Save assistant message
            assistant_msg = Message(
                role='assistant',
                content=full_response,
                thinking=full_thinking if full_thinking else None
            )
            self.messages.append(assistant_msg)
            
            return full_response
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.console.print(f"\n[red]ERROR: {escape(error_msg)}[/red]")
            return error_msg
    
    def _stream_response(self, messages: List[Dict[str, Any]]) -> tuple[str, str, list]:
        """Stream response from Ollama.
        
        Args:
            messages: Messages to send to Ollama
            
        Returns:
            Tuple of (thinking_text, response_text, tool_calls)
        """
        response = ollama.chat(
            model=self.config.model,
            messages=messages,
            tools=self.tools if self.config.tools_enabled else None,
            stream=True,
            options={'temperature': self.config.temperature}
        )
        
        full_thinking = ""
        full_response = ""
        tool_calls = []
        started_thinking = False
        finished_thinking = False
        
        for chunk in response:
            message = chunk.get('message', {})
            
            # Handle thinking
            if 'thinking' in message and message['thinking']:
                if not started_thinking and self.config.show_thinking:
                    self.console.print("\n[bold magenta]Thinking:[/bold magenta]")
                    self.console.print("[dim]" + "=" * 60 + "[/dim]")
                    started_thinking = True
                
                thinking_chunk = message['thinking']
                full_thinking += thinking_chunk
                
                if self.config.show_thinking:
                    self.console.print(f"[dim]{thinking_chunk}[/dim]", end='')
            
            # Handle content
            if 'content' in message and message['content']:
                if started_thinking and not finished_thinking and self.config.show_thinking:
                    self.console.print("\n[dim]" + "=" * 60 + "[/dim]")
                    finished_thinking = True
                
                if not full_response and not started_thinking:
                    self.console.print("\n[bold green]Assistant:[/bold green]")
                
                content_chunk = message['content']
                full_response += content_chunk
            
            # Handle tool calls
            if 'tool_calls' in message:
                raw_tool_calls = message['tool_calls']
                # Convert ToolCall objects to dicts for consistency
                tool_calls = []
                for tc in raw_tool_calls:
                    if isinstance(tc, dict):
                        tool_calls.append(tc)
                    else:
                        # Convert Ollama ToolCall object to dict
                        tool_calls.append({
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments
                            }
                        })
        
        return full_thinking, full_response, tool_calls
    
    def _auto_fetch_urls(self, tool_name: str, tool_result: str, user_query: str) -> str:
        """Automatically fetch and parse URLs from search results.
        
        Args:
            tool_name: Name of the tool that generated the results
            tool_result: Raw tool result text
            user_query: Original user query for relevance scoring
            
        Returns:
            Enhanced tool result with fetched content appended
        """
        if not self.config.get('auto_fetch_urls', True):
            return tool_result
        
        auto_fetch_tools = self.config.get('auto_fetch_tools', ['news_search', 'web_search'])
        if tool_name not in auto_fetch_tools:
            return tool_result
        
        try:
            # Extract and rank URLs by semantic relevance
            threshold = self.config.get('auto_fetch_threshold', 0.6)
            ranked_urls = self.display.extract_and_rank_urls(
                tool_result, 
                user_query, 
                threshold
            )
            
            if not ranked_urls:
                return tool_result
            
            # Limit to top 3 URLs to avoid excessive fetching
            urls_to_fetch = ranked_urls[:3]
            
            # Display what we're going to fetch
            self.console.print(f"\n[bold blue]>> Auto-fetching top {len(urls_to_fetch)} URLs[/bold blue]")
            
            fetched_content = "\n\n[AUTO-FETCHED CONTENT]\n" + "=" * 60 + "\n"
            any_fetched = False
            
            for idx, url_data in enumerate(urls_to_fetch, 1):
                url = url_data['url']
                score = url_data['score']
                
                # Check cache first
                if url in self.url_cache:
                    self.console.print(f"   [{idx}] [cyan]✓ Cached:[/cyan] {url[:70]}")
                    fetched_content += f"\n[{idx}] {url} (relevance: {score:.2f}) [CACHED]\n"
                    fetched_content += "-" * 60 + "\n"
                    fetched_content += self.url_cache[url][:1500] + "\n"
                    any_fetched = True
                    continue
                
                # Fetch content
                self.console.print(f"   [{idx}] [yellow]⟳ Fetching:[/yellow] {url[:70]}")
                
                try:
                    content = self.tool_executor.execute('fetch_url_content', {'url': url, 'max_length': 2000})
                    
                    # Check if fetch was successful (not an error message)
                    if not content.startswith('Error'):
                        # Cache it
                        self.url_cache[url] = content
                        
                        fetched_content += f"\n[{idx}] {url} (relevance: {score:.2f})\n"
                        fetched_content += "-" * 60 + "\n"
                        fetched_content += content[:1500] + "\n"
                        self.console.print(f"   [{idx}] [green]✓ Success[/green]")
                        any_fetched = True
                    else:
                        # Skip unsupported content (videos, etc.) silently, only show other errors
                        if 'Cannot extract text from' in content or 'not currently supported' in content:
                            self.console.print(f"   [{idx}] [dim]⊘ Skipped (unsupported content)[/dim]")
                        else:
                            self.console.print(f"   [{idx}] [red]✗ {escape(content[:70])}[/red]")
                
                except Exception as e:
                    self.console.print(f"   [{idx}] [red]✗ Error: {escape(str(e)[:50])}[/red]")
            
            if any_fetched:
                fetched_content += "\n" + "=" * 60
                return tool_result + fetched_content
            
            return tool_result
            
        except Exception as e:
            self.console.print(f"[yellow]Warning: Auto-fetch failed: {str(e)}[/yellow]")
            return tool_result
    
    def _initialize_rag(self) -> None:
        """Initialize RAG system components."""
        try:
            from rag import OllamaEmbeddingsWrapper, ChromaVectorStore, RAGRetriever, WebSearchRAG
            
            # Initialize embeddings
            embedding_model = self.config.get('embedding_model', 'gemma:2b')
            embeddings = OllamaEmbeddingsWrapper(
                model=embedding_model,
                base_url="http://localhost:11434"
            )
            
            # Initialize vector store
            db_path = self.config.get('chroma_db_path', './chroma_db')
            collection_name = self.config.get('rag_collection', 'rag_documents')
            vector_store = ChromaVectorStore(
                path=db_path,
                collection_name=collection_name
            )
            
            # Initialize retriever
            self.rag_retriever = RAGRetriever(embeddings, vector_store)
            
            # Initialize web search RAG if enabled
            if self.config.get('web_search_rag_enabled', True):
                chunk_size = self.config.get('chunk_size', 500)
                chunk_overlap = self.config.get('chunk_overlap', 100)
                auto_index = self.config.get('web_search_auto_index', True)
                
                self.web_search_rag = WebSearchRAG(
                    self.rag_retriever,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    auto_index=auto_index
                )
            
            self.console.print("[dim]RAG system initialized[/dim]")
        except ImportError as e:
            self.console.print(f"[yellow]Warning: RAG dependencies not available: {e}[/yellow]")
            self.rag_retriever = None
            self.web_search_rag = None
        except Exception as e:
            self.console.print(f"[yellow]Warning: Failed to initialize RAG: {e}[/yellow]")
            self.rag_retriever = None
            self.web_search_rag = None
    
    def get_rag_status(self) -> Dict[str, Any]:
        """Get RAG system status."""
        if not self.rag_retriever:
            return {
                'enabled': False,
                'doc_count': 0,
                'collection': 'N/A',
                'embedding_model': 'N/A'
            }
        
        return {
            'enabled': True,
            'doc_count': self.rag_retriever.store.count(),
            'collection': self.config.get('rag_collection', 'rag_documents'),
            'embedding_model': self.config.get('embedding_model', 'gemma:2b')
        }
    
    def rag_index_file(self, file_path: str) -> int:
        """Index a file into the RAG knowledge base.
        
        Args:
            file_path: Path to file to index
            
        Returns:
            Number of chunks indexed
        """
        if not self.rag_retriever:
            raise RuntimeError("RAG system not initialized")
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use web_search_rag for chunking if available
        if self.web_search_rag:
            return self.web_search_rag.index_single_page(
                url=f"file://{file_path}",
                content=content,
                title=file_path
            )
        else:
            # Simple indexing without chunking
            self.rag_retriever.index_texts([content], ids_prefix="file")
            return 1
    
    def rag_search(self, query: str) -> List[Dict]:
        """Search RAG knowledge base.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        if not self.rag_retriever:
            raise RuntimeError("RAG system not initialized")
        
        top_k = self.config.get('rag_top_k', 3)
        return self.rag_retriever.retrieve(query, top_k=top_k)
    
    def rag_clear(self) -> None:
        """Clear the RAG vector database."""
        if not self.rag_retriever:
            raise RuntimeError("RAG system not initialized")
        
        self.rag_retriever.store.reset()
    
    def rag_rebuild(self) -> int:
        """Placeholder: Returns current document count; does not rebuild RAG index.
        
        Returns:
            Number of documents currently indexed (no rebuild performed)
        """
        if not self.rag_retriever:
            raise RuntimeError("RAG system not initialized")
        
        # This is a placeholder - implement actual rebuild logic based on your document sources
        # For now, just return current count
        return self.rag_retriever.store.count()
