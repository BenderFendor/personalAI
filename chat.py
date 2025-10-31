#!/usr/bin/env python3
"""
Simple Personal AI Chatbot CLI
A file-first chatbot using Ollama with web search capabilities, thinking models, and tool use
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Callable, List
import ollama
from ddgs import DDGS
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.live import Live
from rich.text import Text


class ChatBot:
    def __init__(self, config_path="config.json", logs_dir="chat_logs"):
        """Initialize the chatbot with configuration and logging."""
        self.config_path = Path(config_path)
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize Rich console
        self.console = Console()
        
        # Load or create config
        self.config = self.load_config()
        
        # Initialize chat history
        self.messages = []
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize tools
        self.tools = self._setup_tools()
        self.available_functions = self._setup_functions()
        
        # Context window tracking
        self.context_window_size = 16000  # Default, will be updated from model info
        self._update_model_info()
        
    def load_config(self):
        """Load configuration from file or create default."""
        default_config = {
            "model": "qwen3",
            "temperature": 0.7,
            "system_prompt": (
                "You are a helpful AI assistant with access to various tools. "
                f"The current date is {datetime.now().strftime('%Y-%m-%d')}. "
                "Use the web_search tool to get any information you don't know from after this date. "
                "\n\nIMPORTANT: The web_search tool supports an 'iterations' parameter (1-5). "
                "For complex questions requiring multiple searches:\n"
                "- Call web_search(query='...', iterations=3) to request 3 search cycles\n"
                "- After each search, you'll get results and guidance to refine your next search\n"
                "- Each iteration should build on previous results with more specific queries\n"
                "- Use this for deep research, fact-checking, or gathering comprehensive information"
            ),
            "web_search_enabled": True,
            "max_search_results": 20,
            "thinking_enabled": True,
            "show_thinking": True,
            "tools_enabled": True,
            "markdown_rendering": True,
            "max_tool_iterations": 5
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                return {**default_config, **config}
        else:
            self.save_config(default_config)
            return default_config
    
    def _update_model_info(self):
        """Get model information including context window size from Ollama."""
        try:
            # Use ollama.show() to get model details
            model_info = ollama.show(self.config['model'])
            
            # The context window size is typically in the modelfile or parameters
            # Check parameters string for num_ctx
            if 'parameters' in model_info:
                params = model_info.get('parameters', '')
                if 'num_ctx' in params:
                    import re
                    # Look for "num_ctx 8192" pattern
                    match = re.search(r'num_ctx\s+(\d+)', params)
                    if match:
                        self.context_window_size = int(match.group(1))
                        return
            
            # Check modelfile for PARAMETER num_ctx
            if 'modelfile' in model_info:
                modelfile = model_info.get('modelfile', '')
                if 'num_ctx' in modelfile:
                    import re
                    match = re.search(r'PARAMETER\s+num_ctx\s+(\d+)', modelfile, re.IGNORECASE)
                    if match:
                        self.context_window_size = int(match.group(1))
                        return
            
            # Check details for context_length
            if 'details' in model_info:
                details = model_info.get('details', {})
                if isinstance(details, dict):
                    # Some models report context_length in details
                    if 'context_length' in details:
                        self.context_window_size = details['context_length']
                        return
            
            # Default fallback based on common model sizes
            self.console.print(f"[dim]Using default context window size: 8192 tokens[/dim]")
            self.context_window_size = 8192
            
        except Exception as e:
            # If we can't get model info, use reasonable default
            self.console.print(f"[dim]Could not fetch model info: {e}. Using default context size: 8192[/dim]")
            self.context_window_size = 8192
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        return len(text) // 4
    
    def _calculate_context_usage(self, messages: list) -> dict:
        """Calculate current context window usage."""
        total_chars = 0
        
        # System prompt
        total_chars += len(self.config['system_prompt'])
        
        # All messages
        for msg in messages:
            if isinstance(msg, dict):
                total_chars += len(str(msg.get('content', '')))
                if 'tool_calls' in msg:
                    total_chars += len(str(msg['tool_calls']))
        
        estimated_tokens = self._estimate_tokens(str(total_chars))
        percentage = (estimated_tokens / self.context_window_size) * 100
        
        return {
            'current_tokens': estimated_tokens,
            'max_tokens': self.context_window_size,
            'percentage': percentage,
            'remaining_tokens': self.context_window_size - estimated_tokens
        }
    
    def _display_context_window(self, context_usage: dict):
        """Display context window usage in a compact format."""
        percentage = context_usage['percentage']
        current = context_usage['current_tokens']
        max_tokens = context_usage['max_tokens']
        
        # Choose color based on usage
        if percentage < 50:
            color = "green"
        elif percentage < 75:
            color = "yellow"
        else:
            color = "red"
        
        # Create progress bar
        bar_width = 20
        filled = int((percentage / 100) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        return f"[{color}]Context: {current}/{max_tokens} tokens ({percentage:.1f}%) [{bar}][/{color}]"

    def save_config(self, config=None):
        """Save configuration to file."""
        if config is None:
            config = self.config
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _setup_tools(self) -> List[Dict[str, Any]]:
        """Setup tool definitions for Ollama."""
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'web_search',
                    'description': 'Search the web for current information using DuckDuckGo. Can perform multiple searches with reflection between each search. Use iterations parameter to control how many times to search and refine the query.',
                    'parameters': {
                        'type': 'object',
                        'required': ['query'],
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The search query'
                            },
                            'iterations': {
                                'type': 'integer',
                                'description': 'Number of search iterations to perform (1-5). After each search, results are analyzed before the next search. Default is 1.',
                                'default': 1,
                                'minimum': 1,
                                'maximum': 5
                            }
                        }
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'calculate',
                    'description': 'Perform mathematical calculations',
                    'parameters': {
                        'type': 'object',
                        'required': ['expression'],
                        'properties': {
                            'expression': {
                                'type': 'string',
                                'description': 'Mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)")'
                            }
                        }
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_current_time',
                    'description': 'Get the current date and time',
                    'parameters': {
                        'type': 'object',
                        'properties': {}
                    }
                }
            }
        ]
    
    def _setup_functions(self) -> Dict[str, Callable]:
        """Setup actual function implementations."""
        def web_search_tool(query: str, iterations: int = 1) -> str:
            """Execute web search with optional iterations for deeper research."""
            iterations = max(1, min(iterations, 5))  # Clamp between 1 and 5
            
            # Execute the search
            results = self.web_search(query)
            if not results:
                return "No search results found."
            
            output = f"Search results for '{query}'"
            
            if iterations > 1:
                output += f" (Iteration 1 of {iterations} - Analyze these results and refine your next search)"
            
            output += ":\n\n"
            
            for i, result in enumerate(results, 1):
                output += f"{i}. {result['title']}\n"
                output += f"   URL: {result['url']}\n"
                output += f"   {result['snippet']}\n\n"
            
            if iterations > 1:
                output += (
                    f"\n[ITERATION GUIDANCE]\n"
                    f"You requested {iterations} search iterations.\n"
                    f"After analyzing these results:\n"
                    f"- Identify gaps in information\n"
                    f"- Refine your search query\n"
                    f"- Call web_search() again with the improved query\n"
                    f"- Repeat {iterations - 1} more time(s)\n"
                    f"Each search should build on the previous results.\n"
                )
            
            return output
        
        def calculate_tool(expression: str) -> str:
            """Perform calculation."""
            try:
                # Safe eval with limited scope
                import math
                allowed_names = {
                    'abs': abs, 'round': round, 'min': min, 'max': max,
                    'sum': sum, 'pow': pow, 'sqrt': math.sqrt,
                    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                    'pi': math.pi, 'e': math.e
                }
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return f"Result: {result}"
            except Exception as e:
                return f"Error calculating: {str(e)}"
        
        def get_current_time_tool() -> str:
            """Get current time."""
            now = datetime.now()
            return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        return {
            'web_search': web_search_tool,
            'calculate': calculate_tool,
            'get_current_time': get_current_time_tool
        }
    
    def save_chat_log(self):
        """Save current chat session to markdown file."""
        log_file = self.logs_dir / f"chat_{self.current_session}.md"
        
        with open(log_file, 'w') as f:
            f.write(f"# Chat Log - {self.current_session}\n\n")
            f.write(f"**Model:** {self.config['model']}\n")
            f.write(f"**Temperature:** {self.config['temperature']}\n")
            f.write(f"**Tools Enabled:** {self.config['tools_enabled']}\n")
            f.write(f"**Thinking Enabled:** {self.config['thinking_enabled']}\n\n")
            f.write("---\n\n")
            
            for msg in self.messages:
                role = msg['role'].upper()
                content = msg['content']
                timestamp = msg.get('timestamp', 'N/A')
                
                f.write(f"## {role} [{timestamp}]\n\n")
                
                # Include thinking content if available
                if 'thinking' in msg and msg['thinking']:
                    f.write("### Thinking Process\n\n")
                    f.write(f"```\n{msg['thinking']}\n```\n\n")
                
                f.write(f"{content}\n\n")
                
                if 'sources' in msg:
                    f.write("**Sources:**\n")
                    for i, source in enumerate(msg['sources'], 1):
                        f.write(f"{i}. [{source['title']}]({source['url']})\n")
                        f.write(f"   - {source['snippet']}\n")
                    f.write("\n")
                
                f.write("---\n\n")
    
    def web_search(self, query):
        """Perform web search using DuckDuckGo."""
        if not self.config['web_search_enabled']:
            return None
        
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=self.config['max_search_results'])
            
            search_results = []
            for result in results:
                search_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')
                })
            
            return search_results
        except Exception as e:
            self.console.print(f"[red]Search error: {e}[/red]")
            return None
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool function."""
        if tool_name in self.available_functions:
            try:
                func = self.available_functions[tool_name]
                result = func(**arguments)
                return result
            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"
        return f"Tool {tool_name} not found"
    
    def chat(self, user_input):
        """Process user input and get response from AI."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add user message to history
        user_msg = {
            'role': 'user',
            'content': user_input,
            'timestamp': timestamp
        }
        self.messages.append(user_msg)
        
        # Prepare messages for Ollama
        ollama_messages = [
            {'role': 'system', 'content': self.config['system_prompt'] + ""}
        ]
        
        # Add conversation history
        for msg in self.messages[:-1]:
            ollama_messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
        ollama_messages.append({'role': 'user', 'content': user_input})
        
        # Show context window before making call
        context_usage = self._calculate_context_usage(ollama_messages)
        self.console.print(f"\n[dim]{self._display_context_window(context_usage)}[/dim]")
        
        # Get AI response with tool support
        try:
            # First call with tools and streaming
            response = ollama.chat(
                model=self.config['model'],
                messages=ollama_messages,
                tools=self.tools if self.config['tools_enabled'] else None,
                stream=True,
                options={
                    'temperature': self.config['temperature']
                }
            )
            
            # Track thinking and response content
            full_thinking = ""
            full_response = ""
            started_thinking = False
            finished_thinking = False
            tool_calls = []
            
            # Stream the response
            for chunk in response:
                message = chunk.get('message', {})
                
                # Handle thinking content
                if 'thinking' in message and message['thinking']:
                    if not started_thinking and self.config['show_thinking']:
                        self.console.print("\n[bold magenta]Thinking:[/bold magenta]")
                        self.console.print("[dim]" + "=" * 60 + "[/dim]")
                        started_thinking = True
                    
                    thinking_chunk = message['thinking']
                    full_thinking += thinking_chunk
                    
                    if self.config['show_thinking']:
                        self.console.print(f"[dim]{thinking_chunk}[/dim]", end='')
                
                # Handle regular content
                if 'content' in message and message['content']:
                    if started_thinking and not finished_thinking and self.config['show_thinking']:
                        self.console.print("\n[dim]" + "=" * 60 + "[/dim]")
                        finished_thinking = True
                    
                    if not full_response and not started_thinking:
                        self.console.print("\n[bold green]Assistant:[/bold green]")
                    
                    content_chunk = message['content']
                    full_response += content_chunk
                
                # Check for tool calls
                if 'tool_calls' in message:
                    tool_calls = message['tool_calls']
            
            # Agentic Tool Loop: Allow model to iteratively use tools
            # The model can:
            # 1. Call a tool (e.g., web_search)
            # 2. Receive and analyze the results
            # 3. Decide to call the same or different tool again
            # 4. Continue until it has enough information or max iterations reached
            # This enables multi-step reasoning and information gathering
            
            # Handle tool calls if present - allow iterative tool use
            max_iterations = self.config.get('max_tool_iterations', 5)  # Get from config
            iteration = 0
            
            while tool_calls and iteration < max_iterations:
                iteration += 1
                self.console.print(f"\n[bold blue]>> Tool Iteration {iteration}[/bold blue]")
                
                # Add assistant's tool request to messages
                ollama_messages.append({
                    'role': 'assistant',
                    'content': full_response if full_response else '',
                    'tool_calls': tool_calls
                })
                
                # Execute each tool
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    arguments = tool_call['function']['arguments']
                    
                    # Check if iterations parameter was specified
                    requested_iterations = arguments.get('iterations', 1) if function_name == 'web_search' else 1
                    
                    self.console.print(f"\n[bold cyan]>> Tool Call: {function_name}[/bold cyan]")
                    if requested_iterations > 1:
                        self.console.print(f"[bold yellow]   Multi-iteration search requested: {requested_iterations} searches planned[/bold yellow]")
                    self.console.print(f"[dim]   Arguments: {json.dumps(arguments, indent=2)}[/dim]")
                    
                    # Execute the tool
                    self.console.print("[yellow]   Executing...[/yellow]")
                    tool_result = self._execute_tool(function_name, arguments)
                    
                    # Show abbreviated result
                    result_preview = tool_result[:200] + "..." if len(tool_result) > 200 else tool_result
                    self.console.print(f"[dim]   Result: {result_preview}[/dim]")
                    
                    # Add tool result to messages
                    ollama_messages.append({
                        'role': 'tool',
                        'content': tool_result
                    })
                
                # Get response with tool results - allow model to call tools again if needed
                context_usage = self._calculate_context_usage(ollama_messages)
                
                self.console.print(f"\n[bold yellow]>> Model Call: {self.config['model']}[/bold yellow]")
                self.console.print("[dim]   Processing tool results...[/dim]")
                self.console.print(f"[dim]   Temperature: {self.config['temperature']}[/dim]")
                self.console.print(f"[dim]   Messages in context: {len(ollama_messages)}[/dim]")
                self.console.print(f"[dim]   Tools available: {self.config['tools_enabled']}[/dim]")
                self.console.print(f"[dim]   {self._display_context_window(context_usage)}[/dim]")
                
                # Stream the response to show thinking
                next_response = ollama.chat(
                    model=self.config['model'],
                    messages=ollama_messages,
                    tools=self.tools if self.config['tools_enabled'] else None,
                    stream=True,
                    options={
                        'temperature': self.config['temperature']
                    }
                )
                
                # Reset for next iteration
                full_response = ""
                tool_calls = []
                iteration_thinking = ""
                started_thinking = False
                finished_thinking = False
                
                # Process the streamed response
                for chunk in next_response:
                    message = chunk.get('message', {})
                    
                    # Handle thinking content
                    if 'thinking' in message and message['thinking']:
                        if not started_thinking and self.config['show_thinking']:
                            self.console.print("\n[bold magenta]Thinking:[/bold magenta]")
                            self.console.print("[dim]" + "=" * 60 + "[/dim]")
                            started_thinking = True
                        
                        thinking_chunk = message['thinking']
                        iteration_thinking += thinking_chunk
                        full_thinking += thinking_chunk
                        
                        if self.config['show_thinking']:
                            self.console.print(f"[dim]{thinking_chunk}[/dim]", end='')
                    
                    # Handle regular content
                    if 'content' in message and message['content']:
                        if started_thinking and not finished_thinking and self.config['show_thinking']:
                            self.console.print("\n[dim]" + "=" * 60 + "[/dim]")
                            finished_thinking = True
                        
                        content_chunk = message['content']
                        full_response += content_chunk
                    
                    # Check for more tool calls
                    if 'tool_calls' in message and message['tool_calls']:
                        tool_calls = message['tool_calls']
                
                # If no more tool calls, we're done
                if not tool_calls:
                    self.console.print("\n[bold green]Final Response:[/bold green]")
                    break
                else:
                    self.console.print("\n[dim]Model wants to use more tools...[/dim]")
            
            # Check if we hit max iterations
            if iteration >= max_iterations and tool_calls:
                self.console.print("\n[yellow]Warning: Maximum tool iterations reached. Stopping.[/yellow]")
                self.console.print("[bold green]Partial Response:[/bold green]")
            
            # Render response with markdown if enabled
            if self.config['markdown_rendering'] and full_response:
                md = Markdown(full_response)
                self.console.print(md)
            elif full_response:
                self.console.print(full_response)
            
            # Add assistant message to history (with thinking if available)
            assistant_msg = {
                'role': 'assistant',
                'content': full_response,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if full_thinking:
                assistant_msg['thinking'] = full_thinking
            
            self.messages.append(assistant_msg)
            
            return full_response
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.console.print(f"\n[red]ERROR: {error_msg}[/red]")
            return error_msg
    
    def run(self):
        """Run the interactive chat loop."""
        # Display welcome banner
        welcome_panel = Panel.fit(
            "[bold cyan]Simple Personal AI Chatbot[/bold cyan]\n"
            f"[yellow]Model:[/yellow] {self.config['model']}\n"
            f"[yellow]Tools:[/yellow] {'Enabled' if self.config['tools_enabled'] else 'Disabled'}\n"
            f"[yellow]Thinking:[/yellow] {'Enabled' if self.config['thinking_enabled'] else 'Disabled'}",
            title="[bold]Welcome[/bold]",
            border_style="green"
        )
        self.console.print(welcome_panel)
        
        # Display help
        help_text = """
[bold]Commands:[/bold]
  [cyan]/quit or /exit[/cyan] - Exit the chat
  [cyan]/save[/cyan] - Save chat log
  [cyan]/clear[/cyan] - Clear chat history
  [cyan]/config[/cyan] - Show configuration
  [cyan]/context[/cyan] - Show context window usage
  [cyan]/toggle-tools[/cyan] - Toggle tool use on/off
  [cyan]/toggle-thinking[/cyan] - Toggle thinking display on/off
  [cyan]/toggle-markdown[/cyan] - Toggle markdown rendering
  [cyan]/help[/cyan] - Show this help message
"""
        self.console.print(Panel(help_text, border_style="blue"))
        
        while True:
            try:
                user_input = self.console.input("\n[bold blue]You:[/bold blue] ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input in ['/quit', '/exit']:
                        self.console.print("\n[yellow]Saving chat log...[/yellow]")
                        self.save_chat_log()
                        self.console.print("[green]Goodbye![/green]")
                        break
                    elif user_input == '/save':
                        self.save_chat_log()
                        self.console.print(f"[green]Chat log saved to chat_logs/chat_{self.current_session}.md[/green]")
                        continue
                    elif user_input == '/clear':
                        self.messages = []
                        self.console.print("[green]Chat history cleared![/green]")
                        continue
                    elif user_input == '/config':
                        config_text = "\n[bold]Current Configuration:[/bold]\n"
                        for key, value in self.config.items():
                            config_text += f"  [cyan]{key}:[/cyan] {value}\n"
                        self.console.print(Panel(config_text, title="Configuration", border_style="yellow"))
                        continue
                    elif user_input == '/context':
                        # Prepare messages for context calculation
                        temp_messages = [
                            {'role': 'system', 'content': self.config['system_prompt']}
                        ]
                        for msg in self.messages:
                            temp_messages.append({
                                'role': msg['role'],
                                'content': msg['content']
                            })
                        
                        context_usage = self._calculate_context_usage(temp_messages)
                        
                        context_info = (
                            f"\n[bold]Context Window Status:[/bold]\n\n"
                            f"  [cyan]Model:[/cyan] {self.config['model']}\n"
                            f"  [cyan]Current Usage:[/cyan] {context_usage['current_tokens']} tokens\n"
                            f"  [cyan]Maximum:[/cyan] {context_usage['max_tokens']} tokens\n"
                            f"  [cyan]Remaining:[/cyan] {context_usage['remaining_tokens']} tokens\n"
                            f"  [cyan]Percentage:[/cyan] {context_usage['percentage']:.1f}%\n\n"
                            f"  {self._display_context_window(context_usage)}\n\n"
                            f"  [dim]Messages in history: {len(self.messages)}[/dim]\n"
                        )
                        
                        # Add warning if high usage
                        if context_usage['percentage'] > 75:
                            context_info += "\n  [yellow]Warning: Context window is filling up. Consider using /clear to reset.[/yellow]\n"
                        
                        self.console.print(Panel(context_info, title="Context Window", border_style="cyan"))
                        continue
                    elif user_input == '/toggle-tools':
                        self.config['tools_enabled'] = not self.config['tools_enabled']
                        status = "enabled" if self.config['tools_enabled'] else "disabled"
                        self.console.print(f"[green]Tools {status}[/green]")
                        self.save_config()
                        continue
                    elif user_input == '/toggle-thinking':
                        self.config['show_thinking'] = not self.config['show_thinking']
                        status = "shown" if self.config['show_thinking'] else "hidden"
                        self.console.print(f"[green]Thinking process will be {status}[/green]")
                        self.save_config()
                        continue
                    elif user_input == '/toggle-markdown':
                        self.config['markdown_rendering'] = not self.config['markdown_rendering']
                        status = "enabled" if self.config['markdown_rendering'] else "disabled"
                        self.console.print(f"[green]Markdown rendering {status}[/green]")
                        self.save_config()
                        continue
                    elif user_input == '/help':
                        help_text = """
[bold]Commands:[/bold]
  [cyan]/quit or /exit[/cyan] - Exit the chat
  [cyan]/save[/cyan] - Save chat log
  [cyan]/clear[/cyan] - Clear chat history
  [cyan]/config[/cyan] - Show configuration
  [cyan]/context[/cyan] - Show context window usage
  [cyan]/toggle-tools[/cyan] - Toggle tool use on/off
  [cyan]/toggle-thinking[/cyan] - Toggle thinking display on/off
  [cyan]/toggle-markdown[/cyan] - Toggle markdown rendering
  [cyan]/help[/cyan] - Show this help message
"""
                        self.console.print(Panel(help_text, border_style="blue"))
                        continue
                    else:
                        self.console.print("[red]Unknown command. Type /help for available commands.[/red]")
                        continue
                
                # Process chat message
                self.chat(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n\n[yellow]Saving chat log...[/yellow]")
                self.save_chat_log()
                self.console.print("[green]Goodbye![/green]")
                break
            except Exception as e:
                self.console.print(f"\n[red]Unexpected error: {e}[/red]")


def main():
    """Main entry point."""
    chatbot = ChatBot()
    chatbot.run()


if __name__ == "__main__":
    main()
