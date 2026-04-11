#!/bin/bash
# entrypoint.sh — Smart entrypoint for AdVision Docker container.
#
# When OPENENV_URL is set by the evaluator:
#   → We are in AGENT mode: run inference.py to solve the benchmark task.
#
# Otherwise:
#   → We are in SERVER mode: run the Gradio + FastAPI UI for human evaluation on HF Spaces.

set -e

if [ -n "$OPENENV_URL" ]; then
    echo "[entrypoint] AGENT mode detected (OPENENV_URL=$OPENENV_URL)"
    exec python inference.py
else
    echo "[entrypoint] SERVER mode detected — starting Gradio + FastAPI server"
    exec uvicorn server.app:app --host 0.0.0.0 --port 7860
fi
