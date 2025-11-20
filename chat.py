"""Core chatbot implementation."""

from datetime import datetime
from typing import List, Dict, Any, Generator
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
from utils import ContextCalculator, ChatLogger, DisplayHelper, SessionIndex


class ChatBot:
    """Core chatbot with AI conversation and tool use capabilities."""
    
    def __init__(self, config_path: str = "config.json", logs_dir: str = "chat_logs"):
        """Initialize the chatbot.
        
        Args:
            config_path: Path to configuration file
            logs_dir: Directory for chat logs
        """
        self.console = Console(force_terminal=True, force_interactive=True)
        self.config = ConfigManager(config_path)
        self.context_calc = ContextCalculator(self.config.model)
        self.logger = ChatLogger(logs_dir)
        self.display = DisplayHelper(self.console)
        self.session_index = SessionIndex("chat_logs/index.json")
        
        self.messages: List[Message] = []
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Track whether the session has any substantive interaction to avoid
        # writing empty chat logs on exit.
        self._dirty: bool = False
        self.tools = get_tool_definitions()
        # Token counters for calibration & index metadata
        self.total_prompt_tokens: int = 0
        self.total_eval_tokens: int = 0
        # Lazy init provider for Gemini when selected
        self._gemini_provider = None  # type: ignore
        self.url_cache: Dict[str, str] = {}
        # Context & calibration settings
        self._last_prompt_eval_count: int | None = None
        self._warn_threshold: float = 0.85
        self._truncate_threshold: float = 0.90
        self._keep_last_turns: int = 6
        # Token accounting (populated from Ollama response metadata when available)
        self.total_prompt_tokens: int = 0
        self.total_eval_tokens: int = 0
        self.last_prompt_eval_count: int = 0
        self.last_eval_count: int = 0
        
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
            "- search_and_fetch: Ranked web search with automatic content fetching and RAG indexing\n"
            "- news_search: Search recent news articles with date filtering (use for current events, breaking news)\n"
            "- search_wikipedia: Search Wikipedia for encyclopedic facts, definitions, historical info, biographies\n"
            "- search_arxiv: Search arXiv.org for academic papers, research articles, technical documentation\n"
            "- fetch_url_content: Extract full text content from any URL (use to read full articles from search results)\n"
            "- search_vector_db: Search your indexed knowledge base (RAG) for previously fetched content\n"
            "\n\nTool Selection Strategy:\n"
            "Choose the right tool based on the information need:\n"
            "• search_wikipedia: Encyclopedic facts, definitions, historical events, biographies, scientific concepts, geographic data\n"
            "  - Best for: 'What is quantum computing?', 'Who was Marie Curie?', 'History of Python programming language'\n"
            "  - Returns: Article summaries with inline URL citations, auto-indexed into RAG\n"
            "• search_arxiv: Academic papers, research articles, technical papers, algorithm documentation, scientific studies\n"
            "  - Best for: 'Latest transformer architecture papers', 'Quantum machine learning research', 'Graph neural networks'\n"
            "  - Returns: Paper metadata (title, authors, abstract, publication date) with inline citations, auto-indexed into RAG\n"
            "  - Optional: Can fetch full PDF content for deep research (slower)\n"
            "• web_search: General queries, current information, product reviews, how-to guides, comparisons\n"
            "  - Best for: Most general queries not covered by Wikipedia or arXiv\n"
            "• news_search: Breaking news, current events, recent developments, time-sensitive updates\n"
            "  - Best for: 'Latest AI developments', 'Recent political events'\n"
            "• search_vector_db: Previously indexed content from Wikipedia, arXiv, web, or news\n"
            "  - Use first to check if information is already available before fetching new content\n"
            "\n\nCitation Requirements:\n"
            "- Wikipedia: Use inline citations like 'Quantum computing uses qubits [1]...' where [1] is the Wikipedia article\n"
            "- arXiv: Use academic citations like 'Transformers achieve SOTA [1]...' where [1] is the paper\n"
            "- Web/News: Use inline citations with source titles and URLs\n"
            "- All tools automatically chunk and index content into RAG for later retrieval\n"
            "\n\nUncertainty and Web Search Guidance:\n"
            "Use this decision rule when you are not fully confident you have enough information:\n"
            "- If the user's request is ambiguous, missing crucial constraints, or admits multiple valid interpretations,\n"
            "  first ask up to 2 concise clarifying question(s) before using tools.\n"
            "- If the task clearly requires external facts, current data, or verification, call the appropriate search tool directly.\n"
            "- If you must proceed without clarification (e.g., time-sensitive), state the minimal assumptions you are making.\n"
            "When asked questions that may have been covered in recent searches or news:\n"
            "- FIRST use search_vector_db to check if relevant information is already indexed\n"
            "- If RAG has useful context, synthesize answer from it and cite sources\n"
            "- If RAG lacks info or for breaking news, then use the appropriate search tool (Wikipedia, arXiv, web, or news)\n"
            "- Prefer RAG retrieval when it likely contains the answer; otherwise search and cite sources.\n"
            "\nCall the appropriate search tool when any of the following are true:\n"
            "- You lack necessary data or need to fact-check claims\n"
            "- The topic is time-sensitive or likely to have changed (news, prices, releases, sports)\n"
            "- The user asks for sources, citations, or comparisons across sites\n"
            "- The query involves proper nouns, current events, or niche technical docs\n"
            "\nSearch craft (keep queries short and precise):\n"
            "- Include key entities and constraints (year, version, OS, region)\n"
            "- Use operators when helpful: quotes for exact match, site:example.com, filetype:pdf, OR for alternatives\n"
            "- For Wikipedia: Use specific proper nouns and topic names\n"
            "- For arXiv: Use technical terms, algorithm names, research topics\n"
            "- Chain tools: search -> search_vector_db to retrieve indexed chunks -> synthesize answer\n"
            "\nAnswer structure (keep it concise and sourced):\n"
            "- Start with the direct answer or summary\n"
            "- Cite 2–4 best sources with inline citations (numbers in brackets)\n"
            "- Use the citation format provided by each tool\n"
            "- For academic content (arXiv), use proper citation format: Author et al. (Year). Title.\n"
            "- Include an 'Assumptions' section when you had to assume anything\n"
            "- Include 'Open questions' if clarification is still needed\n"
            "\n\nBest Practices:\n"
            "1. Choose the right search tool: Wikipedia for facts, arXiv for research, web_search for general, news_search for current events\n"
            "2. Always check search_vector_db first to avoid redundant fetches\n"
            "3. All search tools automatically chunk and index content into RAG (ChromaDB)\n"
            "4. Chain tools together: search -> search_vector_db -> synthesize with inline citations\n"
            "5. Use inline URL citations as provided by the tools\n"
            "6. Be critical and verify information from multiple sources and break your thought process and assumptions\n"
            "7. Give concise, accurate, and well-sourced answers based on tool results using snippets, quotes, and excerpts when needed from the sources\n"
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
    
    def save_chat_log(self) -> bool:
        """Save current chat session to file."""
        # Skip saving if nothing meaningful happened in this session.
        if not self._dirty:
            self.console.print("[dim]Nothing to save (no messages sent). Skipping log write.[/dim]")
            return False

        log_path = self.logger.save_session(
            self.current_session,
            self.messages,
            self.config.get_all()
        )
        # Update the session index (best effort)
        try:
            # Title from the first user message, if any
            title = ""
            for m in self.messages:
                if getattr(m, 'role', '') == 'user' and getattr(m, 'content', '').strip():
                    title = " ".join(m.content.strip().split()[:8])
                    break
            self.session_index.add_session(
                session_id=self.current_session,
                file_path=str(log_path),
                title=title or f"Session {self.current_session}",
                tokens_in=self.total_prompt_tokens,
                tokens_out=self.total_eval_tokens,
            )
        except Exception:
            pass
        return True

    def start_new_session(self, save_current: bool = True) -> None:
        """Start a new chat session, optionally saving the current one first.

        Args:
            save_current: If True, save the current session when it has content.
        """
        if save_current:
            self.save_chat_log()
        self.messages = []
        self._dirty = False
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def clear_history(self) -> None:
        """Clear chat message history."""
        self.messages = []
        self._dirty = False
    
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
        # Mark session as having content after first real user input
        if user_input.strip():
            self._dirty = True
        
        # Prepare messages for Ollama
        ollama_messages = [{'role': 'system', 'content': self._get_system_prompt()}]
        for msg in self.messages[:-1]:
            ollama_messages.append(msg.to_ollama_format())
        ollama_messages.append({'role': 'user', 'content': user_input})
        
        # Show context usage
        context_usage = self.context_calc.calculate_usage(ollama_messages, self._get_system_prompt())
        estimated_before = context_usage.current_tokens
        # Warn and conditionally condense if approaching/exceeding limits
        if context_usage.percentage >= self._warn_threshold * 100:
            self.console.print("[yellow]Approaching context limit. Older turns may be summarized if needed.[/yellow]")
        if context_usage.percentage >= self._truncate_threshold * 100:
            self._condense_history()
            # Rebuild messages after condensation
            ollama_messages = [{'role': 'system', 'content': self._get_system_prompt()}]
            for msg in self.messages[:-1]:
                ollama_messages.append(msg.to_ollama_format())
            ollama_messages.append({'role': 'user', 'content': user_input})
            context_usage = self.context_calc.calculate_usage(ollama_messages, self._get_system_prompt())
            estimated_before = context_usage.current_tokens
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
                    
                    # Truncate massive tool results to prevent context overflow and slow streaming
                    if len(tool_result) > 5000:
                        tool_result = tool_result[:5000] + f"\n\n[Truncated {len(tool_result) - 5000} chars...]"
                    
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
                # Update calibration if we have actual prompt tokens
                try:
                    if isinstance(self._last_prompt_eval_count, int):
                        # Estimate for this follow-up call based on current composed messages
                        follow_usage = self.context_calc.calculate_usage(ollama_messages, self._get_system_prompt())
                        self.context_calc.register_actual(self._last_prompt_eval_count, follow_usage.current_tokens)
                        self._last_prompt_eval_count = None
                except Exception:
                    pass
                
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
            # A non-empty assistant message should also mark the session dirty
            if (full_response and full_response.strip()) or (full_thinking and full_thinking.strip()):
                self._dirty = True
            
            # Calibrate after first response
            try:
                if isinstance(self._last_prompt_eval_count, int):
                    self.context_calc.register_actual(self._last_prompt_eval_count, estimated_before)
            finally:
                self._last_prompt_eval_count = None
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
        
        last_prompt_eval = None
        last_eval = None
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

            # Capture token usage metadata when present (on final chunk)
            if 'prompt_eval_count' in chunk or 'eval_count' in chunk:
                try:
                    pe = int(chunk.get('prompt_eval_count') or 0)
                    ec = int(chunk.get('eval_count') or 0)
                    self.last_prompt_eval_count = pe
                    self.last_eval_count = ec
                    # Accumulate totals for the session
                    self.total_prompt_tokens += pe
                    self.total_eval_tokens += ec
                except Exception:
                    pass
        
            # Capture token counts from final chunk if present
            if chunk.get('done') is True:
                if 'prompt_eval_count' in chunk:
                    last_prompt_eval = chunk.get('prompt_eval_count')
                if 'eval_count' in chunk:
                    last_eval = chunk.get('eval_count')

        # Update rolling totals for session and expose last prompt count for calibration
        try:
            if isinstance(last_prompt_eval, int):
                self.total_prompt_tokens += max(0, last_prompt_eval)
                self._last_prompt_eval_count = last_prompt_eval
            if isinstance(last_eval, int):
                self.total_eval_tokens += max(0, last_eval)
        except Exception:
            pass

        return full_thinking, full_response, tool_calls

    def _initialize_rag(self) -> None:
        """Initialize RAG components."""
        try:
            from rag.web_search_rag import WebSearchRAG
            from rag.retriever import RAGRetriever
            from rag.embeddings import OllamaEmbeddingsWrapper
            from rag.vector_store import ChromaVectorStore
            
            # Fix: Look for embedding_model in root config first, then rag dict, then default
            embedding_model = self.config.get('embedding_model') or \
                              self.config.get('rag', {}).get('embedding_model', 'all-MiniLM-L6-v2')
            
            embeddings = OllamaEmbeddingsWrapper(model=embedding_model)
            store = ChromaVectorStore(
                path=self.config.get('chroma_db_path') or self.config.get('rag', {}).get('chroma_db_path', './chroma_db'),
                collection_name=self.config.get('rag_collection') or self.config.get('rag', {}).get('collection_name', 'rag_documents')
            )
            self.rag_retriever = RAGRetriever(embeddings, store)
            self.web_search_rag = WebSearchRAG(self.rag_retriever, auto_index=self.config.get('web_search_rag_enabled', True))
        except Exception as e:
            self.console.print(f"[yellow]Warning: RAG initialization failed: {e}[/yellow]")
            self.web_search_rag = None
            self.rag_retriever = None

    def get_rag_status(self) -> Dict[str, Any]:
        """Get RAG system status."""
        if not self.rag_retriever:
            return {"status": "disabled", "count": 0}
        return {
            "status": "active",
            "count": self.rag_retriever.store.count(),
            "collection": self.rag_retriever.store.collection_name
        }

    def rag_index_file(self, file_path: str) -> int:
        """Index a local file into RAG."""
        if not self.web_search_rag:
            raise ValueError("RAG is not enabled")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use index_single_page as a proxy for indexing text content
            return self.web_search_rag.index_single_page(
                url=f"file://{file_path}",
                content=content,
                title=file_path.split('/')[-1]
            )
        except Exception as e:
            self.console.print(f"[red]Error indexing file: {e}[/red]")
            raise

    def rag_search(self, query: str) -> List[Dict[str, Any]]:
        """Search RAG database directly."""
        if not self.rag_retriever:
            return []
        return self.rag_retriever.retrieve(query)

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all chat sessions."""
        return self.session_index.list_sessions()

    def load_session(self, session_id: str) -> bool:
        """Load a specific session."""
        session = self.session_index.get_session(session_id)
        if not session:
            return False
        
        try:
            file_path = session['file']
            
            # If it's a markdown file, parse it directly
            if file_path.endswith('.md'):
                return self._load_markdown_session(file_path, session_id)
            # Fallback for any legacy JSON files
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.current_session = session_id
                self.messages = []
                for msg in data.get('messages', []):
                    self.messages.append(Message(
                        role=msg['role'],
                        content=msg['content'],
                        thinking=msg.get('thinking'),
                        tool_calls=msg.get('tool_calls')
                    ))
                self._dirty = False
                return True
            
            return False
        except Exception as e:
            self.console.print(f"[red]Error loading session: {e}[/red]")
            return False

    def _load_markdown_session(self, file_path: str, session_id: str) -> bool:
        """Legacy support: Load session from markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.current_session = session_id
            self.messages = []
            
            # Very basic parsing - split by "## ROLE [TIMESTAMP]"
            # This is brittle but better than nothing for legacy files
            import re
            parts = re.split(r'## (USER|ASSISTANT|SYSTEM|TOOL) \[.*?\]\n\n', content)
            
            # parts[0] is header, then alternating role and content
            if len(parts) < 2:
                return False
                
            for i in range(1, len(parts), 2):
                role = parts[i].lower()
                text = parts[i+1].strip()
                
                # Extract thinking if present
                thinking = None
                if "### Thinking Process" in text:
                    think_match = re.search(r'### Thinking Process\n\n```\n(.*?)\n```\n\n', text, re.DOTALL)
                    if think_match:
                        thinking = think_match.group(1)
                        text = text.replace(think_match.group(0), "")
                
                # Remove sources footer if present
                if "**Sources:**" in text:
                    text = text.split("**Sources:**")[0].strip()
                
                # Remove separator
                text = text.replace("\n\n---", "").strip()
                
                self.messages.append(Message(
                    role=role,
                    content=text,
                    thinking=thinking
                ))
            
            self._dirty = False
            return True
        except Exception as e:
            self.console.print(f"[red]Error parsing markdown session: {e}[/red]")
            return False

    def _auto_fetch_urls(self, tool_name: str, tool_result: str, user_query: str) -> str:
        """Automatically fetch URLs from search results if enabled."""
        if not self.config.get('auto_fetch_urls', True):
            return tool_result
            
        if tool_name not in ['web_search', 'search_and_fetch', 'news_search']:
            return tool_result
            
        try:
            urls = self.display.extract_and_rank_urls(tool_result, user_query, threshold=0.6)
            if not urls:
                return tool_result
                
            fetched_content = []
            for item in urls[:2]: # Limit to top 2
                url = item['url']
                if url in self.url_cache:
                    continue
                
                self.console.print(f"[dim]Auto-fetching: {url}[/dim]")
                content = self.tool_executor.fetch_url_content(url)
                self.url_cache[url] = content
                fetched_content.append(f"--- Content from {url} ---\n{content}\n--- End of {url} ---")
            
            if fetched_content:
                return tool_result + "\n\n" + "\n".join(fetched_content)
        except Exception:
            pass
            
        return tool_result

    def _condense_history(self) -> None:
        """Condense message history to save context."""
        if len(self.messages) <= 2:
            return
            
        self.console.print("[yellow]Condensing history...[/yellow]")
        # Keep system prompt (implicit) and last few messages
        keep_count = self._keep_last_turns * 2 # user + assistant
        if len(self.messages) > keep_count:
            removed = len(self.messages) - keep_count
            self.messages = self.messages[-keep_count:]
            self.console.print(f"[dim]Removed {removed} old messages.[/dim]")

    def _parse_tool_calls_from_text(self, text: str) -> tuple[str, list]:
        """Parse tool calls from text block."""
        import re
        import json
        
        pattern = r"```tool_calls\s*(\[.*?\])\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1)
                tool_calls = json.loads(json_str)
                # Remove the block from text
                clean_text = text.replace(match.group(0), "").strip()
                return clean_text, tool_calls
            except Exception:
                pass
        return text, []

    def _enrich_error_message(self, error: Exception) -> str | None:
        """Add helpful context to errors."""
        msg = str(error).lower()
        if "connection refused" in msg:
            return "Is Ollama running? Try 'ollama serve' or check if the service is active."
        if "model" in msg and "not found" in msg:
            return f"Model '{self.config.model}' not found. Try 'ollama pull {self.config.model}'."
        return None

    def chat_stream(self, user_input: str) -> Generator[Dict[str, Any], None, None]:
        """Stream chat response as events instead of printing to console.
        
        Yields:
            Dict with keys: type, content, metadata
        """
        # Add user message
        user_msg = Message(role='user', content=user_input)
        self.messages.append(user_msg)
        if user_input.strip():
            self._dirty = True
        
        # Prepare messages
        ollama_messages = [{'role': 'system', 'content': self._get_system_prompt()}]
        for msg in self.messages[:-1]:
            ollama_messages.append(msg.to_ollama_format())
        ollama_messages.append({'role': 'user', 'content': user_input})
        
        # Context usage
        context_usage = self.context_calc.calculate_usage(ollama_messages, self._get_system_prompt())
        yield {
            "type": "context_update",
            "data": {
                "current_tokens": context_usage.current_tokens,
                "max_tokens": context_usage.max_tokens,
                "percentage": context_usage.percentage,
                "status": "red" if context_usage.percentage >= self._warn_threshold * 100 else "yellow" if context_usage.percentage >= 0.7 * 100 else "green"
            }
        }
        
        # Check limits
        if context_usage.percentage >= self._truncate_threshold * 100:
            self._condense_history()
            # Rebuild messages
            ollama_messages = [{'role': 'system', 'content': self._get_system_prompt()}]
            for msg in self.messages[:-1]:
                ollama_messages.append(msg.to_ollama_format())
            ollama_messages.append({'role': 'user', 'content': user_input})
        
        # Initial response
        full_thinking = ""
        full_response = ""
        tool_calls = []
        
        # Stream generator
        for event in self._stream_generator(ollama_messages):
            yield event
            if event["type"] == "token":
                full_response += event["content"]
            elif event["type"] == "thinking":
                full_thinking += event["content"]
            elif event["type"] == "tool_calls":
                tool_calls = event["data"]
        
        # Tool loop
        max_iterations = self.config.max_tool_iterations
        iteration = 0
        
        while tool_calls and iteration < max_iterations:
            iteration += 1
            
            # Add assistant message
            assistant_msg = {
                'role': 'assistant',
                'content': full_response if full_response else ''
            }
            if tool_calls:
                assistant_msg['tool_calls'] = tool_calls
            ollama_messages.append(assistant_msg)
            
            # Execute tools
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    function_name = tool_call['function']['name']
                    arguments = tool_call['function']['arguments']
                    tool_call_id = tool_call.get('id')
                else:
                    function_name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    tool_call_id = None
                
                yield {"type": "tool_start", "name": function_name, "args": arguments}
                
                tool_result = self.tool_executor.execute(function_name, arguments)
                
                # Auto-fetch
                tool_result = self._auto_fetch_urls(function_name, tool_result, user_input)
                
                yield {"type": "tool_result", "name": function_name, "result": tool_result}
                
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
            
            # Next response
            full_response = "" # Reset for next turn
            tool_calls = []
            
            for event in self._stream_generator(ollama_messages):
                yield event
                if event["type"] == "token":
                    full_response += event["content"]
                elif event["type"] == "thinking":
                    full_thinking += event["content"]
                elif event["type"] == "tool_calls":
                    tool_calls = event["data"]
        
        # Save message
        self.messages.append(Message(
            role='assistant',
            content=full_response,
            thinking=full_thinking if full_thinking else None
        ))
        if (full_response and full_response.strip()) or (full_thinking and full_thinking.strip()):
            self._dirty = True

    def _stream_generator(self, messages: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        """Generator for streaming response events."""
        if self.config.llm_provider == "gemini":
            if GeminiProvider is None:
                yield {"type": "error", "content": "Gemini provider unavailable"}
                return
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
            
            full_response = ""
            tool_calls = []
            for chunk in self._gemini_provider.stream(messages):
                content_chunk = chunk.get("content")
                if content_chunk:
                    full_response += content_chunk
                    yield {"type": "token", "content": content_chunk}
                if chunk.get("tool_calls"):
                    tool_calls = chunk["tool_calls"]
            
            if tool_calls:
                yield {"type": "tool_calls", "data": tool_calls}
            elif "```tool_calls" in full_response:
                cleaned, parsed_calls = self._parse_tool_calls_from_text(full_response)
                if parsed_calls:
                    # We might need to correct the response if we want to hide the block
                    # But for now let's just yield the tool calls
                    yield {"type": "tool_calls", "data": parsed_calls}

        else:
            response = ollama.chat(
                model=self.config.model,
                messages=messages,
                tools=self.tools if self.config.tools_enabled else None,
                stream=True,
                options={'temperature': self.config.temperature}
            )
            
            for chunk in response:
                message = chunk.get('message', {})
                if 'thinking' in message and message['thinking']:
                    yield {"type": "thinking", "content": message['thinking']}
                
                if 'content' in message and message['content']:
                    yield {"type": "token", "content": message['content']}
                
                if 'tool_calls' in message:
                    raw_tool_calls = message['tool_calls']
                    tool_calls = []
                    for tc in raw_tool_calls:
                        if isinstance(tc, dict):
                            tool_calls.append(tc)
                        else:
                            tool_calls.append({
                                'function': {
                                    'name': tc.function.name,
                                    'arguments': tc.function.arguments
                                }
                            })
                    yield {"type": "tool_calls", "data": tool_calls}
                
                # Metadata
                if 'prompt_eval_count' in chunk or 'eval_count' in chunk:
                    try:
                        pe = int(chunk.get('prompt_eval_count') or 0)
                        ec = int(chunk.get('eval_count') or 0)
                        self.total_prompt_tokens += pe
                        self.total_eval_tokens += ec
                    except Exception:
                        pass
