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
│   └── display.py         # Rich console display helpers
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
- `/toggle-tools` - Toggle tool use on/off
- `/toggle-thinking` - Toggle thinking display
- `/toggle-markdown` - Toggle markdown rendering
- `/help` - Show help message

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

See `requirements.txt`:
- `ollama` - Ollama Python client
- `duckduckgo_search` - Web search
- `rich` - Terminal formatting
- `trafilatura` - Content extraction
- `requests` - HTTP client
- `beautifulsoup4` - HTML parsing
- `lxml` - XML/HTML parsing

## 🤝 Contributing

To extend or modify:
1. Follow the existing module structure
2. Add type hints to all functions
3. Include docstrings
4. Keep modules focused and single-purpose
5. Test individual components
