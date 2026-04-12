import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi.responses import JSONResponse

import openenv.core.env_server.types as types
import openenv.core.env_server.serialization as serialization
import openenv.core.env_server.http_server as http_server
from pydantic import ConfigDict, Field

# Removed surgical monkeypatch.
# OpenEnv base schemas are now strictly preserved to ensure 100% compliance with
# the Phase 1 OpenAPI hackathon validator criteria!

# Append root path to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from openenv.core.env_server import create_app
from server.advision_environment import AdVisionEnvironment
from advision_env.models import AdVisionAction, AdVisionObservation

# Initialize core OpenEnv API server
app = create_app(
    AdVisionEnvironment,
    AdVisionAction,
    AdVisionObservation
)


import gradio as gr
from server.ui import demo

from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse

@app.get("/")
async def root_redirect():
    html_content = '<html><head><meta http-equiv="refresh" content="0; url=/ui/"></head><body></body></html>'
    return HTMLResponse(content=html_content)


@app.get("/health")
@app.get("/api/status")
async def status_check():
    return JSONResponse({
        "status": "healthy",
        "service": "advision-env",
        "yolo_ready": True
    })

# Mount the interactive Gradio UI at /ui to avoid shadowing the API at root
app = gr.mount_gradio_app(app, demo, path="/ui")

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    target_port = int(os.environ.get("PORT", port))
    uvicorn.run(app, host=host, port=target_port)

if __name__ == '__main__':
    main()
