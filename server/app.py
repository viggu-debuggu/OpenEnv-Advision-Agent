import os
import sys
import gradio as gr
from pathlib import Path
from fastapi.responses import JSONResponse

# Append root path to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from openenv.core.env_server import create_app
from server.advision_environment import AdVisionEnvironment
from advision_env.models import AdVisionAction, AdVisionObservation
from server.ui import demo as ui_demo

# This automatically handles all /reset, /step, /state, /schema endpoints
app = create_app(
    AdVisionEnvironment,
    AdVisionAction,
    AdVisionObservation
)

# ── Mount Gradio UI: required for Phase 3 human evaluation ───────────────────
app = gr.mount_gradio_app(app, ui_demo, path="/")

# ── Root route: required for HuggingFace Spaces health checks ─────────────────
# Note: Since Gradio is mounted at /, this endpoint might be shadowed,
# but HF health checks usually follow redirects or handle Gradio. 
# We keep it as a fallback /health is also available.
@app.get("/health")
@app.get("/api/status")
async def status_check():
    return JSONResponse({
        "name": "AdVision AI",
        "description": "OpenEnv-compliant In-Content Ad Placement Environment",
        "status": "running",
        "endpoints": ["/health", "/reset", "/step", "/state", "/schema", "/docs", "/"]
    })

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    target_port = int(os.environ.get("PORT", port))
    uvicorn.run(app, host=host, port=target_port)

if __name__ == '__main__':
    main()
