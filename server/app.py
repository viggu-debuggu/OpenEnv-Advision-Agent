import os
import sys
from pathlib import Path
from fastapi.responses import JSONResponse

# Append root path to Python path so server can import advision_env
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from openenv.core.env_server import create_app
from server.advision_environment import AdVisionEnvironment
from models import AdVisionAction, AdVisionObservation

# This automatically handles all /reset, /step, /state, /schema endpoints
app = create_app(
    AdVisionEnvironment,
    AdVisionAction,
    AdVisionObservation
)

# ── Root route: required for HuggingFace Spaces health checks ─────────────────
@app.get("/")
async def root():
    return JSONResponse({
        "name": "AdVision AI",
        "description": "OpenEnv-compliant In-Content Ad Placement Environment",
        "status": "running",
        "endpoints": ["/health", "/reset", "/step", "/state", "/schema", "/docs"]
    })

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    target_port = int(os.environ.get("PORT", port))
    uvicorn.run(app, host=host, port=target_port)

if __name__ == '__main__':
    main()
