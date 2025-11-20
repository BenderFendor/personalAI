# GitHub Copilot Instructions for PersonalAI

use .venv for the virtual environment for python.
use uv for pip installing.

## Project Overview
Personal AI Chatbot is a **modular RAG-powered CLI chatbot** using Ollama for LLM inference, with web search, tool calling, and vector-based retrieval. The architecture follows SOLID principles with clear separation between core logic, tools, RAG, and utilities.

## Key Architecture Patterns

### 1. Modular Tool System (`tools/`)
- **Tool Definition**: `tools/definitions.py` defines function schemas for Ollama's native function calling
- **Tool Execution**: `tools/implementations.py` contains `ToolExecutor` class with registry pattern
- **Auto-Indexing Pattern**: Tools like `web_search` and `fetch_url_content` automatically index results into RAG if `web_search_rag_enabled: true`
- **Adding New Tools**: 
  1. Add schema to `get_tool_definitions()` in `definitions.py`
  2. Implement method in `ToolExecutor` class
  3. Register in `_register_tools()` dict mapping

Example:
```python
# In definitions.py - add to returned list
{
    'type': 'function',
    'function': {
        'name': 'my_tool',
        'description': 'Clear, actionable description for the LLM',
        'parameters': {'type': 'object', 'required': ['param'], 'properties': {...}}
    }
}

# In implementations.py - add method and register
def my_tool(self, param: str) -> str:
    """Implementation."""
    return result

# Add to _register_tools()
'my_tool': self.my_tool
```

### 2. RAG Architecture (`rag/`)
- **Two-Layer Indexing**: Web search results and fetched pages are automatically chunked and indexed via `WebSearchRAG.index_search_results()` and `index_single_page()`
- **Chunking Strategy**: Uses LangChain `RecursiveCharacterTextSplitter` with `chunk_size=500` and `chunk_overlap=100` (configurable in `config.json`)
- **Vector Store**: ChromaDB with `sentence-transformers/all-MiniLM-L6-v2` embeddings (see `config.json` → `rag.embedding_model`)
- **Retrieval Tool**: `search_vector_db` tool available to LLM for querying indexed content
- **Auto-Index Flow**: `web_search` → `WebSearchRAG.index_search_results()` → ChromaDB; `fetch_url_content` → `WebSearchRAG.index_single_page()` → ChromaDB

### 3. Agentic Tool Loop (chat.py)
The chatbot uses an **iterative tool execution pattern** (max 5 iterations by default):
1. LLM generates response with optional tool calls
2. Execute all tools in parallel and add results to messages
3. LLM processes tool results and may request more tools
4. Loop continues until LLM returns final text response or max iterations reached

This allows multi-hop reasoning: `web_search` → analyze results → `fetch_url_content` → synthesize answer.

### 4. System Prompt Strategy
The system prompt is **dynamically generated** in `ChatBot._get_system_prompt()`:
- **Clarify-First Rule**: LLM asked to pose up to 2 clarifying questions for ambiguous queries before using tools
- **Citation Requirements**: Web search results must cite 2-4 sources in answers
- **Tool Selection Guidance**: Explicit when to use `web_search` vs `news_search`, iterative search patterns
- **RAG Integration**: Automatically injects RAG context info (doc count, instructions to cite sources)

When modifying prompt behavior, edit `_get_system_prompt()` in `chat.py`.

### 5. Context Window Management (`utils/context.py`)
- **Auto-Detection**: `ContextCalculator` detects model context limits via Ollama API
- **Token Estimation**: Approximates tokens as `len(text) / 4` (fast heuristic)
- **Visual Feedback**: Color-coded progress bars (green → yellow → red) shown before each LLM call
- Models like `qwen3` (32K context) are detected automatically

### 6. URL Auto-Fetch with Semantic Ranking (`utils/display.py`)
**Smart Content Fetching**:
- `DisplayHelper.extract_and_rank_urls()` uses `sentence-transformers` to rank search result URLs by semantic similarity to user query
- Only URLs with similarity ≥ 0.6 (configurable via `auto_fetch_threshold`) are fetched
- Avoids unsupported content types (videos, PDFs, audio) with explicit checks in `fetch_url_content()`
- Cached results stored in `ChatBot.url_cache` to avoid redundant fetches

Pattern for adding semantic features: Load `SentenceTransformer('all-MiniLM-L6-v2')` in `__init__`, use `util.pytorch_cos_sim()` for similarity.

### 7. Configuration Management (`config_manager.py`)
- **Type-Safe Access**: `ConfigManager` provides property-based access to config values with defaults
- **Hot Reload**: CLI commands like `/toggle-tools` modify config and call `config.save()` for persistence
- **Nested Config**: RAG settings nested under `config.json` → `rag` key
- **Environment-Specific**: Local `config.json` not tracked in git (add to `.gitignore`)

## Common Development Workflows

### Running & Testing
```bash
# Activate venv first (already done based on context)
source .venv/bin/activate

# Run chatbot
python main.py  # or python chat.py (legacy entry point)

# Test specific tool
python -c "from tools.implementations import ToolExecutor; from rich.console import Console; \
executor = ToolExecutor({}, Console()); print(executor.web_search('test query'))"
```

### Adding a New RAG Source
To index custom documents beyond web search:
```python
# In CLI or new script
chatbot = ChatBot()
chunk_count = chatbot.rag_index_file('/path/to/document.txt')
# Uses WebSearchRAG chunking under the hood
```

### Debugging Tool Calls
- Tool execution logs show in colored output: `[bold cyan]>> Tool Call: tool_name`
- Multi-iteration searches display iteration guidance: `"Iteration 1 of 3 - Analyze and refine..."`
- Tool results abbreviated to 200 chars in console (full results passed to LLM)

### Modifying LLM Behavior
1. **Change model**: Edit `config.json` → `model` (e.g., `"llama3.1"`, `"mixtral"`)
2. **Adjust temperature**: `config.json` → `temperature` (0.0-1.0)
3. **Disable tools**: Set `tools_enabled: false` or use `/toggle-tools` at runtime
4. **Control thinking**: `thinking_enabled: true` + `show_thinking: true` for reasoning traces

## Project-Specific Conventions

### Error Handling
- **Rich Markup Safety**: Always use `rich.markup.escape()` on user-provided text before printing to avoid markup injection
- **Tool Errors**: Return descriptive error strings (e.g., `"Error: Tool 'x' not found"`) rather than raising exceptions
- **Graceful Degradation**: If RAG/embeddings unavailable, system warns but continues with web search only

### Message Flow
Messages use `models.Message` dataclass with `to_ollama_format()` for API compatibility:
- User/Assistant messages: `role` + `content`
- Tool results: `role: 'tool'` + `content` (Ollama format)
- System prompt: Always first message, dynamically generated

### Type Hints
All functions use type hints. Key types:
- `List[Dict[str, Any]]` for Ollama message arrays
- `Message` for internal representation
- `ContextUsage` for context tracking

### Testing Strategy
No formal test suite yet, but manual testing workflow:
1. Test tools individually via `ToolExecutor`
2. Test RAG indexing via `/rag-index` and `/rag-search` commands
3. Test multi-turn conversations with tool chaining
4. Verify context limits don't break with long chats

## Using exa-code for Development

**exa-code** is a search tool that provides high-quality, up-to-date code examples and documentation for programming tasks. Use it to:

1. **Find Implementation Patterns**: When adding new features (e.g., "LangChain ChromaDB retrieval patterns"), exa-code returns relevant, real-world code snippets
2. **Resolve Library Issues**: Search for specific SDK usage (e.g., "Ollama function calling Python examples")
3. **Architecture Guidance**: Query best practices (e.g., "RAG chunking strategies with RecursiveCharacterTextSplitter")

**How to Use exa-code**:
- Call `get_code_context_exa(query="your detailed query", tokensNum=5000)` for code examples
- Call `web_search_exa(query="your query", numResults=5)` for broader web search
- Use specific queries with library names, versions, and frameworks for best results
- Adjust `tokensNum` based on complexity: 5000 for focused examples, 10000-15000 for comprehensive docs

**Example Usage**:
```python
# When implementing a new RAG feature
get_code_context_exa(
    query="ChromaDB vector store with LangChain text splitters for document indexing",
    tokensNum=8000
)

# When debugging Ollama integration
get_code_context_exa(
    query="Ollama Python streaming responses with tool calling and function definitions",
    tokensNum=5000
)
```

**Best Practices**:
- Use exa-code when official docs are unclear or you need working examples
- Prefer specific queries over generic ones ("LangChain ChromaDB metadata filtering" > "vector database")
- Cross-reference exa-code results with project patterns to maintain consistency
- Store useful patterns in comments for future reference

## Integration Points

### Ollama Dependency
- **Local Server**: Requires Ollama running on `localhost:11434` (default)
- **Model Management**: Models must be pre-pulled (`ollama pull qwen3`)
- **Function Calling**: Uses Ollama's native tool calling (not all models support this - use tool-capable models like `qwen2`, `llama3.1`)

### ChromaDB Persistence
- **Location**: `./chroma_db` directory (configurable via `chroma_db_path`)
- **Collection Naming**: `rag_documents` (configurable via `rag_collection`)
- **Schema**: Documents stored with metadata: `source`, `title`, `chunk_index`, `total_chunks`, `type`

### DuckDuckGo Search
- Uses `ddgs` library (synchronous API)
- Rate limits apply - tool auto-handles errors and returns descriptive messages
- `max_search_results: 20` default (configurable)

## Known Quirks & Gotchas

1. **Tool Iteration Parameter**: `web_search` accepts `iterations: 1-5` but each iteration is a separate LLM call - expensive for large iteration counts
2. **Context Overflow**: Long conversations + tool results can exceed context window. Use `/clear` or implement auto-truncation
3. **URL Caching**: `url_cache` is in-memory only - cleared on restart. Consider persistent cache for production
4. **Markdown Rendering**: Toggle with `/toggle-markdown` - some terminals may not render Rich markdown properly
5. **RAG Rebuild Placeholder**: `/rag-rebuild` currently only returns doc count - implement actual rebuild logic for document sources
6. **PDF Support**: `fetch_url_content` explicitly blocks PDFs - add PDF extraction if needed (e.g., using `pdfplumber`)

## Quick Reference

**Key Files**:
- `chat.py`: Core chatbot orchestration and agentic loop
- `tools/implementations.py`: All tool logic and `ToolExecutor`
- `rag/web_search_rag.py`: Auto-indexing for web content
- `config.json`: All runtime configuration
- `utils/display.py`: Rich console formatting and semantic ranking

**Key Commands (CLI)**:
- `/toggle-tools`, `/toggle-thinking`, `/toggle-markdown`: Runtime toggles
- `/rag-status`, `/rag-search <query>`, `/rag-index <file>`: RAG management
- `/context`: Check token usage
- `/history` or `Ctrl+]`: Browse past chats

**Dependencies**:
- `ollama`: LLM client (function calling support)
- `duckduckgo_search`: Web search
- `trafilatura`: Clean web content extraction
- `chromadb`: Vector store
- `langchain`, `langchain-text-splitters`: Text chunking
- `sentence-transformers`: Embeddings and semantic similarity
- `rich`: Terminal UI

When in doubt, check `README.md` for architectural overview or use exa-code to find similar implementations in other projects.
