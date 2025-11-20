from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import shutil
import os
from chat import ChatBot
from models import Message

app = FastAPI(title="Personal AI API")

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
        # For now, we just use the current session or start a new one if needed
        # But ChatBot is stateful.
        pass
    
    async def event_generator():
        for event in chatbot.chat_stream(request.message):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
