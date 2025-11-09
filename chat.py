"""Core chatbot implementation."""

from datetime import datetime
from typing import List, Dict, Any
import json
import ollama
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.markup import escape

from models import Message, ContextUsage
from config_manager import ConfigManager
from tools import get_tool_definitions, ToolExecutor
try:
    from utils.gemini_provider import GeminiProvider
    from tools.adapter import LangChainToolAdapter
except Exception:
    GeminiProvider = None
    LangChainToolAdapter = None
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
        # Lazy init provider for Gemini when selected
        self._gemini_provider = None  # type: ignore
        self.url_cache: Dict[str, str] = {}
        
        # Initialize RAG if enabled
        self.rag_retriever = None
        self.web_search_rag = None
        if self.config.get('rag_enabled', False):
            self._initialize_rag()
        
        # Initialize tool executor (after RAG initialization)
        self.tool_executor = ToolExecutor(self.config.get_all(), self.console, self.web_search_rag)

        # Proactive checks for common misconfigurations
        self._warn_common_misconfigurations()

    def _warn_common_misconfigurations(self) -> None:
        """Print helpful hints if config looks inconsistent.

        - If provider is Ollama but `model` looks like a Gemini model, suggest switching provider or model.
        - If provider is Gemini but no API key, surface an actionable message early.
        """
        try:
            provider = self.config.llm_provider
            model = self.config.model
            gemini_model = getattr(self.config, "gemini_model", "")

            if provider == "ollama" and model.lower().startswith("gemini"):
                self.console.print(
                    "[yellow]\nWarning: Your provider is set to 'ollama' but the model is '"
                    + escape(model)
                    + "', which looks like a Google Gemini model.\n"
                    "• To use Gemini, set llm_provider to 'gemini' and put the model under 'gemini_model'.\n"
                    "• Or keep llm_provider 'ollama' and change 'model' to a local model (e.g., 'qwen3', 'llama3.1').\n"
                    "Edit config.json accordingly.\n[/yellow]"
                )

            if provider == "gemini":
                api_key = self.config.gemini_api_key
                if not api_key:
                    self.console.print(
                        "[yellow]\nWarning: llm_provider is 'gemini' but GOOGLE_API_KEY isn't set.\n"
                        "Set it in your environment or .env file, e.g.:\n"
                        "  GOOGLE_API_KEY=your_key_here\n[/yellow]"
                    )
                # Nudge to correct common model name variants
                if gemini_model and gemini_model.lower() in {"gemini-flash", "gemini-flash-latest", "gemini"}:
                    self.console.print(
                        "[dim]Tip: Prefer explicit Gemini ids like 'gemini-1.5-flash' or 'gemini-1.5-pro'.[/dim]"
                    )
        except Exception:
            # Never block startup due to diagnostics
            pass
    
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
        
        provider_note = ""
        try:
            if self.config.llm_provider == "gemini":
                provider_note = (
                    "\n\nTool use with Gemini:\n"
                    "- You are bound to a set of tools with named parameters. Prefer calling tools directly when needed.\n"
                    "- If the runtime does not accept native tool calls, output a fenced JSON block exactly in this format and nothing else on that line:\n"
                    "```tool_calls\n"
                    "[{\n  \"name\": \"web_search\", \n  \"arguments\": {\"query\": \"...\", \"iterations\": 1}\n}]\n"
                    "```\n"
                    "- Use the exact tool names and argument keys from the schema above.\n"
                )
        except Exception:
            pass

        return (
            "You are a helpful AI assistant with access to various tools. "
            f"The current date is {datetime.now().strftime('%Y-%m-%d')}. "
            "\n\nAvailable Search Tools:\n"
            "- web_search: General web search using DuckDuckGo (use for most queries)\n"
            "- news_search: Search recent news articles with date filtering (use for current events, breaking news)\n"
            "- fetch_url_content: Extract full text content from any URL (use to read full articles from search results)\n"
            "\n\nUncertainty and Web Search Guidance:\n"
            "Use this decision rule when you are not fully confident you have enough information:\n"
            "- If the user's request is ambiguous, missing crucial constraints, or admits multiple valid interpretations,\n"
            "  first ask up to 2 concise clarifying question(s) before using tools.\n"
            "- If the task clearly requires external facts, current data, or verification, call web_search directly.\n"
            "- If you must proceed without clarification (e.g., time-sensitive), state the minimal assumptions you are making.\n"
            "- Prefer RAG retrieval when it likely contains the answer; otherwise search the web and cite sources.\n"
            "\nCall web_search when any of the following are true:\n"
            "- You lack necessary data or need to fact-check claims\n"
            "- The topic is time-sensitive or likely to have changed (news, prices, releases, sports)\n"
            "- The user asks for sources, citations, or comparisons across sites\n"
            "- The query involves proper nouns, current events, or niche technical docs\n"
            "\nSearch craft (keep queries short and precise):\n"
            "- Include key entities and constraints (year, version, OS, region)\n"
            "- Use operators when helpful: quotes for exact match, site:example.com, filetype:pdf, OR for alternatives\n"
            "- Iterate if needed; each iteration should refine based on prior results\n"
            "\nAnswer structure (keep it concise and sourced):\n"
            "- Start with the direct answer or summary\n"
            "- Cite 2–4 best sources with titles and URLs; quote short snippets when decisive\n"
            "- Include an 'Assumptions' section when you had to assume anything\n"
            "- Include 'Open questions' if clarification is still needed\n"
            "\n\nBest Practices:\n"
            "1. Use web_search for general information and research\n"
            "2. Use news_search when the user asks about recent events, news, or current affairs\n"
            "3. The system automatically fetches relevant article content for you\n"
            "4. Chain tools together: search -> use fetched content for synthesis\n"
            "5. If you don't have the context or information use web_search\n"
            "6. Give concise, accurate, and well-sourced answers based on tool results using snippets, quotes, and excerpts when needed from the sources\n"
            f"{fetch_info}"
            f"{rag_info}"
            "\nThe web_search tool supports an 'iterations' parameter (1-5). "
            "For complex questions requiring multiple searches:\n"
            "- Call web_search(query='...', iterations=3) to request 3 search cycles\n"
            "- After each search, you'll get results and guidance to refine your next search\n"
            "- Each iteration should build on previous results with more specific queries\n"
            "- Use this for deep research, fact-checking, or gathering comprehensive information"
            f"{provider_note}"
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
                assistant_msg: Dict[str, Any] = {
                    'role': 'assistant',
                    'content': full_response if full_response else ''
                }
                if tool_calls:
                    assistant_msg['tool_calls'] = tool_calls
                ollama_messages.append(assistant_msg)
                
                # Execute tools
                for tool_call in tool_calls:
                    # Handle both dict and ToolCall object formats
                    if isinstance(tool_call, dict):
                        function_name = tool_call['function']['name']
                        arguments = tool_call['function']['arguments']
                        tool_call_id = tool_call.get('id')
                    else:
                        # ToolCall object from Ollama
                        function_name = tool_call.function.name
                        arguments = tool_call.function.arguments
                        tool_call_id = None
                    
                    self.display.show_tool_call(function_name, arguments, iteration)
                    
                    # Execute tool
                    tool_result = self.tool_executor.execute(function_name, arguments)
                    self.display.show_tool_result(tool_result)
                    
                    # Auto-fetch URLs from search results if applicable
                    tool_result = self._auto_fetch_urls(function_name, tool_result, user_input)
                    
                    # Add result to messages; for Gemini include tool_call_id for proper linkage
                    if self.config.llm_provider == "gemini" and tool_call_id:
                        ollama_messages.append({
                            'role': 'tool',
                            'content': tool_result,
                            'tool_call_id': tool_call_id
                        })
                    else:
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
            self.messages.append(Message(
                role='assistant',
                content=full_response,
                thinking=full_thinking if full_thinking else None
            ))
            
            return full_response
            
        except Exception as e:
            # Provide enriched troubleshooting if this looks like a provider/model issue
            enriched = self._enrich_error_message(e)
            self.console.print(f"\n[red]ERROR:[/red] {escape(str(e))}")
            if enriched:
                self.console.print(enriched)
            return f"Error: {str(e)}"
    
    def _stream_response(self, messages: List[Dict[str, Any]]) -> tuple[str, str, list]:
        """Stream response from Ollama.
        
        Args:
            messages: Messages to send to Ollama
            
        Returns:
            Tuple of (thinking_text, response_text, tool_calls)
        """
        # Provider switch: Gemini or Ollama
        if self.config.llm_provider == "gemini":
            if GeminiProvider is None:
                raise RuntimeError("Gemini provider unavailable; install langchain-google-genai.")
            if self._gemini_provider is None:
                self._gemini_provider = GeminiProvider(
                    model=self.config.gemini_model,
                    api_key=self.config.gemini_api_key,
                    temperature=self.config.temperature,
                    minimal_safety=self.config.gemini_safety_minimal,
                )
                if self.config.tools_enabled and LangChainToolAdapter:
                    adapter = LangChainToolAdapter(self.tool_executor)
                    lc_tools = adapter.build_tools(self.tools)
                    self._gemini_provider.bind_tools(lc_tools)
            # Stream from Gemini
            full_thinking = ""  # Gemini doesn't provide separate thinking field
            full_response = ""
            tool_calls: List[Dict[str, Any]] = []
            for chunk in self._gemini_provider.stream(messages):  # type: ignore[attr-defined]
                content_chunk = chunk.get("content")
                if content_chunk:
                    if not full_response:
                        self.console.print("\n[bold green]Assistant (Gemini):[/bold green]")
                    full_response += content_chunk
                if chunk.get("tool_calls"):
                    tool_calls = chunk["tool_calls"]  # overwrite with latest collected calls
            # Fallback: if no structured tool_calls emitted, try to parse an explicit tool_calls block from text
            if not tool_calls and "```tool_calls" in full_response:
                cleaned, parsed_calls = self._parse_tool_calls_from_text(full_response)
                if parsed_calls:
                    full_response = cleaned
                    tool_calls = parsed_calls
            return full_thinking, full_response, tool_calls

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

    def _parse_tool_calls_from_text(self, text: str) -> tuple[str, list]:
        """Extract tool calls from a fenced code block labeled 'tool_calls'.

        Format expected:
        ```tool_calls
        [ { "name": "web_search", "arguments": { ... } }, ... ]
        ```

        Returns a tuple of (text_without_block, tool_calls_list)
        where tool_calls_list are dicts shaped like {"function": {"name": str, "arguments": dict}}
        """
        try:
            import re, json
            pattern = r"```tool_calls\s*(.*?)\s*```"
            m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            if not m:
                return text, []
            block = m.group(1).strip()
            calls = json.loads(block)
            parsed = []
            if isinstance(calls, dict):
                calls = [calls]
            if isinstance(calls, list):
                for idx, c in enumerate(calls, start=1):
                    name = (c or {}).get("name")
                    args = (c or {}).get("arguments", {})
                    if name and isinstance(args, dict):
                        parsed.append({
                            "id": f"tc-{idx}",
                            "function": {"name": name, "arguments": args}
                        })
            # Remove the block from the text
            new_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            return new_text, parsed
        except Exception:
            return text, []

    def _enrich_error_message(self, err: Exception) -> str:
        """Return an actionable troubleshooting message for common failures.

        Covers:
        - Ollama 404 'model not found' with Gemini-like model names.
        - Gemini provider issues (missing API key or bad model id).
        """
        try:
            msg = str(err).lower()
            provider = self.config.llm_provider
            cfg = self.config.get_all()
            model = cfg.get("model")
            gemini_model = cfg.get("gemini_model")

            lines: list[str] = []

            # Case 1: Ollama cannot find model
            if ("model" in msg and "not found" in msg) or ("status code: 404" in msg):
                if provider == "ollama":
                    # Detect if user put a Gemini-like id into Ollama 'model'
                    if model and str(model).lower().startswith("gemini"):
                        lines.append("[bold]It looks like you're trying to use a Gemini model with the Ollama provider.[/bold]")
                        lines.append("Fix options:")
                        lines.append("- Use Gemini: set in config.json → 'llm_provider': 'gemini', and set 'gemini_model': 'gemini-1.5-flash' (or 'gemini-1.5-pro').")
                        lines.append("  Also set GOOGLE_API_KEY in your environment or .env file.")
                        lines.append("- Stay on Ollama: change 'model' to a local model id (e.g., 'qwen3', 'llama3.1', 'mistral') and pull it if needed:")
                        lines.append("  ollama pull qwen3")
                    else:
                        lines.append("[bold]Ollama can't find the requested model.[/bold]")
                        lines.append(f"Requested: '{model}'")
                        lines.append("Try pulling or switching models. Examples:")
                        lines.append("- ollama pull qwen3  (then set 'model': 'qwen3')")
                        lines.append("- ollama pull llama3.1  (then set 'model': 'llama3.1')")
                else:
                    lines.append("The backend reported 'model not found', but provider isn't Ollama. Double-check config.json.")

            # Case 2: Gemini provider problems
            if provider == "gemini":
                # Missing API key
                if "api key" in msg or "permission" in msg or "unauthorized" in msg:
                    lines.append("[bold]Gemini requires GOOGLE_API_KEY.[/bold] Set it in your environment or .env:")
                    lines.append("GOOGLE_API_KEY=your_key_here")
                # Common wrong model ids
                if gemini_model and gemini_model.lower() in {"gemini", "gemini-flash", "gemini-flash-latest"}:
                    lines.append("Use explicit Gemini ids like 'gemini-1.5-flash' or 'gemini-1.5-pro'.")

            if not lines:
                return ""

            # Format as a friendly Rich block
            header = "\n[bold yellow]Troubleshooting tips[/bold yellow]\n"
            bullet = "\n".join(f" • {escape(line)}" for line in lines)
            where = (
                "\n[dim]Edit settings in config.json. Current values: "
                f"llm_provider='{escape(str(provider))}', model='{escape(str(model))}', gemini_model='{escape(str(gemini_model))}'.[/dim]"
            )
            return header + bullet + where
        except Exception:
            return ""
    
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
            embedding_model = self.config.get('embedding_model', 'embeddinggemma')
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
                show_chunk_previews = self.config.get('show_chunk_previews', True)
                
                # Provide preview_printer callback to write chunk previews using console
                self.web_search_rag = WebSearchRAG(
                    self.rag_retriever,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    auto_index=auto_index,
                    show_chunk_previews=show_chunk_previews,
                    preview_printer=lambda text: self.console.print(text)
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
            'embedding_model': self.config.get('embedding_model', 'embeddinggemma')
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

    def rag_hard_delete(self) -> None:
        """Hard delete the RAG collection (drop & recreate)."""
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
