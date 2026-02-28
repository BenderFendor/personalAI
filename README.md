# Personal AI

Local-first AI chatbot with CLI, API, and web UI. Uses llama.cpp server for local inference. Supports tool calls and web search.

## Structure
- `main.py`: CLI entry point
- `chat.py`: core ChatBot loop
- `api.py`: FastAPI server with SSE streaming
- `tools/`: tool schemas and implementations
- `frontend/`: Next.js UI

## Run
Backend:
- `uv pip install -r requirements.txt`
- Start llama.cpp server: `llama-server -m models/your-model.gguf --port 8080`
- `python main.py`

API and UI:
- `uvicorn api:app --reload --host 0.0.0.0 --port 8000`
- `cd frontend && npm install && npm run dev`
- `./startserver.sh` runs both and expects a `.venv`

## Commands
- `/help`, `/save`, `/new`, `/clear`, `/history`, `/quit`
- `/config`, `/context`
- `/toggle-tools`, `/toggle-thinking`, `/toggle-markdown`

## Tools
- `web_search`, `search_and_fetch`, `news_search`
- `fetch_url_content`
- `search_wikipedia`, `search_arxiv`, `search_academic`, `search_pubmed`
- `deep_research`, `calculate`, `get_current_time`

## Config
Edit `config.json`. Key fields: `llm_provider`, `gemini_model`, `web_search_enabled`, `auto_fetch_urls`.
