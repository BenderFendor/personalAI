#!/bin/bash
# Activate the virtual environment
source .venv/bin/activate
# Start the Python backend in the background and redirect output (optional)
uvicorn api:app --reload --host 0.0.0.0 --port 8000 &

# Start the npm frontend in the current foreground
cd frontend && npm run dev

# Wait for all background processes to complete (optional, ensures script waits for the frontend process if the backend finishes first)
wait
