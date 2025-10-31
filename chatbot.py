"""Core chatbot implementation."""

from datetime import datetime
from typing import List, Dict, Any
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
    
    def _get_system_prompt(self) -> str:
        """Generate the system prompt dynamically.
        
        Returns:
            System prompt string
        """
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
            "3. Use fetch_url_content to get detailed information from specific URLs in search results\n"
            "4. Chain tools together: search -> fetch_url_content for deeper research\n"
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
                    function_name = tool_call['function']['name']
                    arguments = tool_call['function']['arguments']
                    
                    self.display.show_tool_call(function_name, arguments, iteration)
                    
                    # Execute tool
                    tool_result = self.tool_executor.execute(function_name, arguments)
                    self.display.show_tool_result(tool_result)
                    
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
                tool_calls = message['tool_calls']
        
        return full_thinking, full_response, tool_calls
