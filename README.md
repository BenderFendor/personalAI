# Personal AI

Local-first AI chatbot with CLI, API, and web UI. Default provider is Ollama with optional Gemini. Supports tool calls, web search, and RAG indexing.

## Structure
- `main.py`: CLI entry point
- `chat.py`: core ChatBot loop
- `api.py`: FastAPI server with SSE streaming
- `tools/`: tool schemas and implementations
- `rag/`: RAG indexing and retrieval helpers
- `frontend/`: Next.js UI

## Run
Backend:
- `uv pip install -r requirements.txt`
- `python main.py`

API and UI:
- `uvicorn api:app --reload --host 0.0.0.0 --port 8000`
- `cd frontend && npm install && npm run dev`
- `./startserver.sh` runs both and expects a `.venv`

For Ollama, pull your model with `ollama pull <model>`.

## Commands
- `/help`, `/save`, `/new`, `/clear`, `/history`, `/quit`
- `/config`, `/context`
- `/toggle-tools`, `/toggle-thinking`, `/toggle-markdown`, `/toggle-chunk-previews`
- `/rag-status`, `/rag-index <path>`, `/rag-search <query>`, `/rag-clear`, `/rag-rebuild`, `/rag-hard-delete`

## Tools
- `web_search`, `search_and_fetch`, `news_search`
- `fetch_url_content`, `search_vector_db`
- `search_wikipedia`, `search_arxiv`, `search_academic`, `search_pubmed`
- `deep_research`, `calculate`, `get_current_time`

## Config
Edit `config.json` for provider, model, and feature flags. Key fields include `llm_provider`, `model`, `gemini_model`, `rag_enabled`, `web_search_enabled`, and `auto_fetch_urls`.

## Gemini
Set `llm_provider` to `gemini`, set `gemini_model`, and export `GOOGLE_API_KEY`.
