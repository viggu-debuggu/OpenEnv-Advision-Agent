#!/bin/bash
# entrypoint.sh — Smart entrypoint for AdVision Docker container.
set -e

if [ -n "$OPENENV_URL" ]; then
    echo "[entrypoint] Evaluator mode detected (OPENENV_URL=$OPENENV_URL)"
    echo "[entrypoint] Running agent: python inference.py"
    exec python inference.py
else
    echo "[entrypoint] Server mode detected — starting Gradio + FastAPI server"
    exec uvicorn server.app:app --host 0.0.0.0 --port 7860
fi
