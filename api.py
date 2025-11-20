from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import shutil
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from chat import ChatBot
from models import Message

app = FastAPI(title="Personal AI API")

# Create a thread pool for running synchronous tasks
executor = ThreadPoolExecutor()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChatBot
chatbot = ChatBot()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ConfigUpdate(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    tools_enabled: Optional[bool] = None
    thinking_enabled: Optional[bool] = None
    show_thinking: Optional[bool] = None
    web_search_enabled: Optional[bool] = None
    auto_fetch_urls: Optional[bool] = None

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if request.session_id and request.session_id != chatbot.current_session:
        chatbot.load_session(request.session_id)
    
    async def event_generator():
        stream = chatbot.chat_stream(request.message)
        loop = asyncio.get_event_loop()
        while True:
            try:
                # Run next(stream) in thread pool to avoid blocking the event loop
                event = await loop.run_in_executor(executor, next, stream)
                yield f"data: {json.dumps(event)}\n\n"
            except StopIteration:
                break
            except Exception as e:
                print(f"Error in stream: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                break
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/chat/new")
async def new_chat():
    chatbot.start_new_session(save_current=True)
    return {"session_id": chatbot.current_session, "messages": []}

@app.get("/api/history")
async def get_history():
    return [msg.to_ollama_format() for msg in chatbot.messages]

@app.post("/api/history/clear")
async def clear_history():
    chatbot.clear_history()
    return {"status": "cleared"}

@app.get("/api/config")
async def get_config():
    return chatbot.config.get_all()

@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    updates = config.model_dump(exclude_unset=True)
    
    for key, value in updates.items():
        chatbot.config.set(key, value)
        
    chatbot.config.save()
    return chatbot.config.get_all()

@app.get("/api/rag/status")
async def rag_status():
    return chatbot.get_rag_status()

@app.post("/api/rag/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save to temp file
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Index
        chunks = chatbot.rag_index_file(file_path)
        
        # Cleanup
        os.remove(file_path)
        
        return {"status": "success", "chunks_indexed": chunks, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SearchRequest(BaseModel):
    query: str

@app.post("/api/rag/search")
async def rag_search(request: SearchRequest):
    return chatbot.rag_search(request.query)

@app.get("/api/sessions")
async def list_sessions():
    return chatbot.get_all_sessions()

@app.post("/api/sessions/rebuild")
async def rebuild_sessions():
    """Rebuild session index from chat_logs directory."""
    count = chatbot.rebuild_session_index()
    return {"status": "success", "sessions_indexed": count}

@app.post("/api/sessions/{session_id}/load")
async def load_session(session_id: str):
    success = chatbot.load_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Return the messages so frontend can update immediately
    return {
        "status": "loaded", 
        "session_id": session_id,
        "messages": [msg.to_ollama_format() for msg in chatbot.messages]
    }
