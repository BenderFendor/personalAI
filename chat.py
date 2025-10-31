"""Core chatbot implementation."""

from datetime import datetime
from typing import List, Dict, Any
import json
import ollama
from rich.console import Console

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
        self.tool_executor = ToolExecutor(self.config.get_all(), self.console)
        
        self.messages: List[Message] = []
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tools = get_tool_definitions()
        self.url_cache: Dict[str, str] = {}
    
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
            self.console.print(f"\n[red]ERROR: {error_msg}[/red]")
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
                        self.console.print(f"   [{idx}] [red]✗ {content[:50]}[/red]")
                
                except Exception as e:
                    self.console.print(f"   [{idx}] [red]✗ Error: {str(e)[:50]}[/red]")
            
            if any_fetched:
                fetched_content += "\n" + "=" * 60
                return tool_result + fetched_content
            
            return tool_result
            
        except Exception as e:
            self.console.print(f"[yellow]Warning: Auto-fetch failed: {str(e)}[/yellow]")
            return tool_result
