import os
import sys
from pathlib import Path
from fastapi.responses import JSONResponse

# Append root path to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from openenv.core.env_server import create_app
from server.advision_environment import AdVisionEnvironment
from advision_env.models import AdVisionAction, AdVisionObservation

# Initialize core OpenEnv API server first to ensure route precedence
app = create_app(
    AdVisionEnvironment,
    AdVisionAction,
    AdVisionObservation
)


import gradio as gr
from server.ui import demo

@app.get("/health")
@app.get("/api/status")
async def status_check():
    return JSONResponse({
        "name": "AdVision AI",
        "description": "OpenEnv-compliant In-Content Ad Placement Environment",
        "status": "running",
        "endpoints": ["/health", "/reset", "/step", "/state", "/schema", "/docs", "/"]
    })

# Mount the interactive Gradio UI at the root path for human judges
app = gr.mount_gradio_app(app, demo, path="/")

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    target_port = int(os.environ.get("PORT", port))
    uvicorn.run(app, host=host, port=target_port)

if __name__ == '__main__':
    main()
