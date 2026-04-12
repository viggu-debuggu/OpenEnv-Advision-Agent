#!/bin/bash
set -e

# Load the environment server in the background for ALL modes
# This ensures that even in 'evaluator mode', the server is reachable at localhost:7860
# inside the same container.
echo "[entrypoint] Starting AdVision server in background..."
uvicorn server.app:app --host 0.0.0.0 --port 7860 &

# If evaluator injects OPENENV_URL → run as agent
if [ -n "$OPENENV_URL" ]; then
    echo "[entrypoint] Evaluator mode detected → waiting for server..."
    # Give the server a few seconds to warm up
    sleep 3
    echo "[entrypoint] Running inference.py against $OPENENV_URL"
    exec python inference.py
else
    # Normal HF Space mode → just wait for the background server
    echo "[entrypoint] Server mode detected → waiting for background process"
    wait -n
fi
