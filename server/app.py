import os
import sys
import time
import logging
import asyncio
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Append root path to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server.advision_environment import AdVisionEnvironment
from advision_env.models import AdVisionAction, AdVisionObservation, AdVisionState

# Initialize the environment singleton
env = AdVisionEnvironment()

# Initialize FastAPI with strict compliance settings
app = FastAPI(
    title="AdVision OpenEnv Server",
    description="High-performance Ad Placement Environment for OpenEnv Hackathon",
    version="1.0.0",
    redirect_slashes=False  # Problem 4: Prevent 307 redirects
)

# Problem 6: Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response Models for strict compliance
class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float]
    done: bool

# --- API Endpoints ---

@app.get("/")
async def root_redirect():
    html_content = '<html><head><meta http-equiv="refresh" content="0; url=/ui/"></head><body></body></html>'
    return HTMLResponse(content=html_content)

@app.get("/health")
@app.get("/api/status")
async def health_check():
    # Problem 3 & 5: Descriptive and compliant health check
    return {
        "status": "healthy",
        "service": "advision-env",
        "yolo_ready": True,
        "episode_id": env.state.episode_id
    }

@app.post("/reset", response_model=ResetResponse)
async def reset(request: Dict[str, Any] = Body(default={})):
    # Problem 7: Null reward on reset protocol
    seed = request.get("seed")
    obs = env.reset(seed=seed, **request)
    
    # Problem 1 & 2: Explicitly serialize to include metadata
    return {
        "observation": obs.model_dump(),
        "reward": None,
        "done": False
    }

@app.post("/step", response_model=StepResponse)
async def step(payload: Dict[str, Any] = Body(...)):
    # Extract action from wrapper if present (standard OpenEnv protocol)
    action_data = payload.get("action", payload)
    
    try:
        # Validate action using our Pydantic model
        action_obj = AdVisionAction(**action_data)
        obs = env.step(action_obj)
        
        # Problem 1 & 2: Ensure metadata and custom fields survive
        return {
            "observation": obs.model_dump(),
            "reward": float(obs.reward) if obs.reward is not None else 0.0,
            "done": bool(obs.done)
        }
    except Exception as e:
        logging.error(f"Step error: {e}")
        # Standard OpenEnv 422/500 formatting
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/state")
async def get_state():
    # Problem 5: Ensure /state reflects the singleton instance correctly
    return env.state.model_dump()

@app.get("/schema")
async def get_schema():
    # Problem 1: Return correct subclass schemas
    return {
        "action": AdVisionAction.model_json_schema(),
        "observation": AdVisionObservation.model_json_schema(),
        "state": AdVisionState.model_json_schema()
    }

@app.get("/metadata")
async def get_metadata():
    return {
        "name": "AdVision Environment",
        "description": "RL-Env for In-Content Ad Placement",
        "version": "1.0.0",
        "author": "AdVision Team"
    }

# --- Gradio UI ---
import gradio as gr
from server.ui import demo
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
