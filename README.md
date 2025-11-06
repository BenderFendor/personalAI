# Personal AI Chatbot - Refactored

A modular, clean implementation of a personal AI chatbot using Ollama with web search capabilities, thinking models, and tool use.

## 📁 Project Structure

```
personalAI/
├── main.py                 # Entry point for the application
├── chatbot.py             # Core chatbot logic
├── cli.py                 # Command-line interface
├── config_manager.py      # Configuration management
├── models.py              # Data models (Message, ContextUsage, etc.)
├── config.json            # Configuration file
├── requirements.txt       # Python dependencies
│
├── tools/                 # Tool system
│   ├── __init__.py
│   ├── definitions.py     # Tool definitions for Ollama
│   └── implementations.py # Tool implementations (web search, news, etc.)
│
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── context.py         # Context window calculations
│   ├── logger.py          # Chat logging
│   ├── display.py         # Rich console display helpers
│   ├── keyboard.py        # Keyboard input handling
│   └── sidebar.py         # Chat history sidebar UI
│
└── chat_logs/             # Saved chat sessions
    └── chat_*.md
```

## 🏗️ Architecture

### Core Components

1. **ChatBot (`chatbot.py`)**
   - Main chatbot orchestration
   - Message handling and conversation flow
   - Streaming responses from Ollama
   - Agentic tool loop for iterative tool use

2. **ChatCLI (`cli.py`)**
   - Interactive command-line interface
   - Command handling (/quit, /save, /config, etc.)
   - User input processing

3. **ConfigManager (`config_manager.py`)**
   - Configuration loading and saving
   - Type-safe config access
   - Default values management

4. **Models (`models.py`)**
   - `Message`: Chat message with metadata
   - `ContextUsage`: Context window tracking
   - `SearchResult`: Web search results
   - `ToolCall`: Tool invocation data

### Modules

#### Tools Module (`tools/`)
- **definitions.py**: Tool schemas for Ollama function calling
- **implementations.py**: Actual tool implementations
  - `web_search`: DuckDuckGo web search with iterations
  - `news_search`: Recent news article search
  - `fetch_url_content`: Extract content from URLs
  - `calculate`: Mathematical calculations
  - `get_current_time`: Current date/time

#### Utils Module (`utils/`)
- **context.py**: Context window size detection and token estimation
- **logger.py**: Save chat sessions to markdown files
- **display.py**: Rich console formatting and display helpers
- **keyboard.py**: Keyboard input handling for special keys
- **sidebar.py**: Interactive chat history sidebar UI

## 🚀 Usage

### Running the Chatbot

```bash
python main.py
```

or use the original entry point:

```bash
python chat.py
```

### Available Commands

- `/quit` or `/exit` - Exit the chat
- `/save` - Save chat log
- `/clear` - Clear chat history
- `/config` - Show configuration
- `/context` - Show context window usage
- `/history` - View past chat sessions sidebar
- `/toggle-tools` - Toggle tool use on/off
- `/toggle-thinking` - Toggle thinking display
- `/toggle-markdown` - Toggle markdown rendering
- `/help` - Show help message

### Keyboard Shortcuts

- **Ctrl+]** - Toggle chat history sidebar (browse and load past sessions)
- **Ctrl+C** - Interrupt current operation

### Chat History Sidebar

Press **Ctrl+]** or type `/history` to open an interactive sidebar showing all past chat sessions:
- Use **↑/↓** arrow keys to navigate through sessions
- Press **Enter** to view the selected session
- Press **Esc** or **Ctrl+]** again to close the sidebar

The sidebar displays:
- Date and time of each session
- Preview of the first user message
- Easy navigation to review past conversations

## 🔧 Configuration

Edit `config.json` to customize:

```json
{
  "model": "qwen3",
  "temperature": 0.7,
  "web_search_enabled": true,
  "max_search_results": 20,
  "thinking_enabled": true,
  "show_thinking": true,
  "tools_enabled": true,
  "markdown_rendering": true,
  "max_tool_iterations": 5
}
```

## 🎯 Key Features

### Modular Design
- **Separation of Concerns**: Each module has a single responsibility
- **Easy Testing**: Components can be tested independently
- **Maintainable**: Changes to one module don't affect others
- **Extensible**: Easy to add new tools or features

### Tool System
- **Pluggable**: Add new tools by updating `tools/` module
- **Iterative**: Support for multi-step tool use
- **Agentic Loop**: AI can chain multiple tool calls

### Context Management
- **Auto-detection**: Automatically detects model context window size
- **Visual Tracking**: Progress bar showing context usage
- **Warnings**: Alerts when context is filling up

### Rich Display
- **Markdown Rendering**: Beautiful formatted output
- **Thinking Display**: Optional display of AI reasoning
- **Color Coding**: Context usage color-coded by level
- **Structured Output**: Clean, organized information display

### Clarify-first and Sourced Answers
- The system prompt instructs the model to ask up to two short clarifying questions when a query is ambiguous.
- When external facts are required, it uses `web_search` and cites 2–4 sources in answers.
- If assumptions are needed (e.g., time-sensitive requests), the model lists them explicitly and notes any open questions.

## 📝 Adding New Tools

1. Add tool definition in `tools/definitions.py`:
```python
{
    'type': 'function',
    'function': {
        'name': 'my_new_tool',
        'description': 'What this tool does',
        'parameters': {
            'type': 'object',
            'required': ['param1'],
            'properties': {
                'param1': {
                    'type': 'string',
                    'description': 'Parameter description'
                }
            }
        }
    }
}
```

2. Implement the tool in `tools/implementations.py`:
```python
def my_new_tool(self, param1: str) -> str:
    """Tool implementation."""
    # Your logic here
    return result
```

3. Register in `ToolExecutor._register_tools()`:
```python
'my_new_tool': self.my_new_tool
```

## 🎨 Design Principles

1. **Clean Code**: Clear naming, proper typing, docstrings
2. **SOLID Principles**: Single responsibility, open/closed, etc.
3. **DRY**: Don't repeat yourself - shared logic extracted
4. **Type Safety**: Type hints throughout for better IDE support
5. **Error Handling**: Graceful error handling and user feedback

## 🔄 Migration from Original

The original monolithic `chat.py` has been refactored into:
- Configuration → `config_manager.py`
- Data structures → `models.py`
- Tools → `tools/` module
- Utilities → `utils/` module
- Core logic → `chatbot.py`
- CLI → `cli.py`

The original `chat.py` still works but is now superseded by the modular structure.

## 📦 Dependencies

See `requirements.txt` for the full list. Core libraries include:
- `ollama` – Local LLM inference and tool-calling
- `duckduckgo_search` – Web search API
- `trafilatura` – Clean article text extraction
- `rich` – Terminal UI / formatting
- `requests`, `beautifulsoup4`, `lxml` – HTTP and HTML parsing helpers
- `langchain-google-genai` – Gemini integration (optional; only needed when `llm_provider` is `gemini`)

### Optional: Gemini Provider
To use Google Gemini instead of Ollama:
1. Install extras: `pip install langchain-google-genai`
2. Set environment variable (or add to `.env`):
    ```
    GOOGLE_API_KEY=your_key_here
    ```
3. In `config.json` set:
    ```json
    {
      "llm_provider": "gemini",
      "gemini_model": "gemini-1.5-flash"
    }
    ```

If the API key is missing you'll get a clear startup warning.

### Ollama Models
Ensure you pull local models before use, e.g.:
```
ollama pull qwen3
ollama pull llama3.1
```
Then set `"model": "qwen3"` (or whichever you pulled) in `config.json` when `llm_provider` is `ollama`.

## 🛠 Troubleshooting

### Model Not Found (Ollama 404)
You might see:
```
ERROR: model 'gemini-flash-latest' not found (status code: 404)
```
This usually means a Gemini model name was placed under the Ollama `model` key.

Fix options:
1. Use Gemini:
    - Set `"llm_provider": "gemini"`
    - Set `"gemini_model": "gemini-1.5-flash"` (or `gemini-1.5-pro`)
    - Export `GOOGLE_API_KEY`
2. Stay with Ollama:
    - Change `"model"` to a local model id (e.g. `qwen3`, `llama3.1`, `mistral`)
    - Pull it with `ollama pull <model>` if not already present.

### Gemini Errors
- Missing API key → set `GOOGLE_API_KEY`
- Ambiguous model id (e.g. `gemini-flash-latest`) → prefer explicit versions like `gemini-1.5-flash` or `gemini-1.5-pro`.

### No Tool Calls With Gemini
Some model variants may have limited tool support. Ensure you installed `langchain-google-genai` and restarted.

### Context Overflow
If long sessions approach the context window, use `/clear` or trim earlier messages. Future enhancement: automatic summarization.

---

## 🤝 Contributing

To extend or modify:
1. Follow the existing module structure
2. Add type hints to all functions
3. Include docstrings
4. Keep modules focused and single-purpose
5. Test individual components
