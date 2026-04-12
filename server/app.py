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

# --- SURGICAL MONKEYPATCH FOR OPENENV-CORE v0.2.3 ---
# This fixes a bug where metadata and custom observation fields are stripped by the server.

# 1. Allow extra fields in the base Observation model
types.Observation.model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)

# 2. Re-map Response models to include metadata at the top level
class PatchedResetResponse(types.ResetResponse):
    model_config = ConfigDict(extra="allow")
    metadata: Dict[str, Any] = Field(default_factory=dict)
types.ResetResponse = PatchedResetResponse
http_server.ResetResponse = PatchedResetResponse

class PatchedStepResponse(types.StepResponse):
    model_config = ConfigDict(extra="allow")
    metadata: Dict[str, Any] = Field(default_factory=dict)
types.StepResponse = PatchedStepResponse
http_server.StepResponse = PatchedStepResponse

# 3. Override serialization to fill-in the missing metadata
def patched_serialize_observation(observation):
    # Dumps the model while preserving our extra fields (due to extra="allow")
    obs_dict = observation.model_dump(exclude={"reward", "done", "metadata"})
    metadata = getattr(observation, 'metadata', {})
    
    return {
        "observation": obs_dict,
        "reward": float(getattr(observation, 'reward', 0.0)),
        "done": bool(getattr(observation, 'done', False)),
        "metadata": metadata
    }
serialization.serialize_observation = patched_serialize_observation
# ----------------------------------------------------

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

from fastapi.responses import JSONResponse, RedirectResponse


@app.get("/health")
@app.get("/api/status")
async def status_check():
    return JSONResponse({
        "name": "AdVision AI",
        "description": "OpenEnv-compliant In-Content Ad Placement Environment",
        "status": "running",
        "endpoints": ["/", "/health", "/reset", "/step", "/state", "/schema", "/docs", "/ui"]
    })

# Mount the interactive Gradio UI directly at the root
app = gr.mount_gradio_app(app, demo, path="/")

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    target_port = int(os.environ.get("PORT", port))
    uvicorn.run(app, host=host, port=target_port)

if __name__ == '__main__':
    main()
