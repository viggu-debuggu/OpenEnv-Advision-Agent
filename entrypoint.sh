#!/bin/bash
set -e

# If evaluator injects OPENENV_URL → run as agent
if [ -n "$OPENENV_URL" ]; then
    echo "[entrypoint] Evaluator mode detected → running inference.py"
    exec python inference.py
else
    # Normal HF Space mode → run the FastAPI + Gradio server
    echo "[entrypoint] Server mode detected → starting uvicorn"
    exec uvicorn server.app:app --host 0.0.0.0 --port 7860
fi
