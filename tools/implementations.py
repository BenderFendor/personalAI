"""Tool implementations for web search, news, URL fetching, and more."""

import math
from datetime import datetime
from typing import Dict, Any, Optional
import requests
import trafilatura
from trafilatura.settings import use_config
from ddgs import DDGS
from rich.console import Console


class ToolExecutor:
    """Handles execution of all tool functions."""
    
    def __init__(self, config: Dict[str, Any], console: Console):
        """Initialize tool executor.
        
        Args:
            config: Configuration dictionary
            console: Rich console for output
        """
        self.config = config
        self.console = console
        self._tools = self._register_tools()
    
    def _register_tools(self) -> Dict[str, callable]:
        """Register all available tool functions.
        
        Returns:
            Dictionary mapping tool names to functions
        """
        return {
            'web_search': self.web_search,
            'news_search': self.news_search,
            'fetch_url_content': self.fetch_url_content,
            'calculate': self.calculate,
            'get_current_time': self.get_current_time
        }
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool function by name.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result as string
        """
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            func = self._tools[tool_name]
            result = func(**arguments)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def web_search(self, query: str, iterations: int = 1) -> str:
        """Execute web search with optional iterations.
        
        Args:
            query: Search query
            iterations: Number of search iterations (1-5)
            
        Returns:
            Formatted search results
        """
        if not self.config.get('web_search_enabled', True):
            return "Web search is disabled"
        
        iterations = max(1, min(iterations, 5))  # Clamp between 1 and 5
        
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=self.config.get('max_search_results', 20))
            
            if not results:
                return f"No search results found for '{query}'"
            
            output = f"Search results for '{query}'"
            
            if iterations > 1:
                output += f" (Iteration 1 of {iterations} - Analyze these results and refine your next search)"
            
            output += ":\n\n"
            
            for i, result in enumerate(results, 1):
                output += f"{i}. {result.get('title', 'No title')}\n"
                output += f"   URL: {result.get('href', 'No URL')}\n"
                output += f"   {result.get('body', 'No description')}\n\n"
            
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
            
        except Exception as e:
            error_msg = f"Error performing web search: {str(e)}"
            self.console.print(f"[red]{error_msg}[/red]")
            return error_msg
    
    def news_search(
        self,
        keywords: str,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        max_results: int = 10
    ) -> str:
        """Search for recent news articles.
        
        Args:
            keywords: Keywords to search for
            region: Region code (e.g., "us-en", "uk-en")
            safesearch: Safe search level ("on", "moderate", "off")
            timelimit: Time filter ("d", "w", "m", or None)
            max_results: Maximum number of results (1-50)
            
        Returns:
            Formatted news search results
        """
        if not self.config.get('web_search_enabled', True):
            return "News search is disabled"
        
        try:
            max_results = max(1, min(max_results, 50))
            
            ddgs = DDGS()
            results = ddgs.news(
                query=keywords,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results
            )
            
            if not results:
                return f"No news articles found for '{keywords}'"
            
            output = f"News search results for '{keywords}'"
            if timelimit:
                time_desc = {"d": "past day", "w": "past week", "m": "past month"}.get(timelimit, "")
                output += f" ({time_desc})"
            output += f" - Found {len(results)} articles:\n\n"
            
            for i, article in enumerate(results, 1):
                output += f"{i}. {article.get('title', 'No title')}\n"
                output += f"   Source: {article.get('source', 'Unknown source')}\n"
                output += f"   Date: {article.get('date', 'Unknown date')}\n"
                output += f"   URL: {article.get('url', 'No URL')}\n"
                
                body = article.get('body', '')
                if body:
                    body_preview = body[:300] + "..." if len(body) > 300 else body
                    output += f"   Summary: {body_preview}\n"
                
                if article.get('image'):
                    output += f"   Image: {article['image']}\n"
                
                output += "\n"
            
            output += "\n[TIP] Use fetch_url_content(url='...') to read the full article text from any URL above.\n"
            
            return output
            
        except Exception as e:
            error_msg = f"Error performing news search: {str(e)}"
            self.console.print(f"[red]{error_msg}[/red]")
            return error_msg
    
    def fetch_url_content(self, url: str, max_length: int = 5000) -> str:
        """Fetch and extract clean text content from a URL.
        
        Args:
            url: URL to fetch content from
            max_length: Maximum character length (500-20000)
            
        Returns:
            Extracted text content
        """
        if not url.startswith(('http://', 'https://')):
            return "Error: Invalid URL format. URL must start with http:// or https://"
        
        try:
            max_length = max(500, min(max_length, 20000))
            
            config = use_config()
            config.set("DEFAULT", "EXTRACTION_TIMEOUT", "10")
            
            self.console.print(f"[dim]Fetching content from {url[:60]}...[/dim]")
            
            downloaded = trafilatura.fetch_url(url)
            
            if not downloaded:
                # Fallback to requests
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    response = requests.get(url, timeout=10, headers=headers)
                    response.raise_for_status()
                    downloaded = response.text
                except Exception as req_error:
                    return f"Error fetching URL: Could not download content. Error: {str(req_error)}"
            
            content = trafilatura.extract(
                downloaded,
                include_formatting=False,
                include_links=False,
                include_images=False,
                include_tables=True,
                config=config,
                favor_recall=True
            )
            
            if not content:
                return f"Error: Could not extract readable content from {url}"
            
            if len(content) > max_length:
                content = content[:max_length] + f"\n\n[Content truncated at {max_length} characters. Original: {len(content)} chars]"
            
            output = f"Content extracted from {url}:\n"
            output += "=" * 60 + "\n\n"
            output += content
            output += "\n\n" + "=" * 60
            output += f"\n[Extracted {len(content)} characters]"
            
            return output
            
        except Exception as e:
            error_msg = f"Error fetching content from {url}: {str(e)}"
            self.console.print(f"[red]{error_msg}[/red]")
            return error_msg
    
    def calculate(self, expression: str) -> str:
        """Perform mathematical calculation.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Calculation result
        """
        try:
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
    
    def get_current_time(self) -> str:
        """Get current date and time.
        
        Returns:
            Formatted current date and time
        """
        now = datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
