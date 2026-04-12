import os
import sys
from typing import Any, Dict

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# --- PATH SETUP (Must be before internal imports) ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server.advision_environment import AdVisionEnvironment  # noqa: E402
from advision_env.models import AdVisionAction  # noqa: E402

# Initialize the environment singleton
env = AdVisionEnvironment()

# Initialize FastAPI with strict compliance settings
app = FastAPI(
    title="AdVision OpenEnv Server",
    description="High-performance Ad Placement Environment for OpenEnv Hackathon",
    version="1.0.0"
)

# Enable CORS for metadata and UI accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# CORE API ENDPOINTS (OpenEnv Specification v0.2.3)
# ------------------------------------------------------------------------------

@app.get("/health", tags=["System"])
@app.get("/", include_in_schema=False)
def health():
    """Liveness check for container orchestrators and validators."""
    return {"status": "ok", "service": "advision-openenv", "message": "POST to /reset to start"}

@app.get("/metadata", tags=["Environment"])
def metadata():
    """Returns environment specification metadata."""
    return env.metadata()

@app.get("/schema", tags=["Environment"])
def get_schema():
    """Returns the JSON schema for Observations, Actions, and States."""
    return env.schema()

@app.post("/reset", tags=["Environment"])
def reset(payload: Dict[str, Any] = Body(default_factory=dict)):
    """Resets the environment and returns the initial observation."""
    seed = payload.get("seed")
    # OpenEnv spec uses episode_id, AdVision previously used options. We support both.
    episode_id = payload.get("episode_id") or payload.get("options", {}).get("episode_id")
    
    obs = env.reset(seed=seed, episode_id=episode_id)
    
    # Extract info if available, otherwise return empty dict
    info = getattr(obs, "info", {})
    return {"observation": obs, "info": info}

@app.post("/step", tags=["Environment"])
def step(action: AdVisionAction):
    """Executes a single environment step."""
    obs = env.step(action)
    return {
        "observation": obs,
        "reward": float(obs.reward) if obs.reward is not None else 0.0,
        "done": bool(obs.done),
        "terminated": bool(obs.done),
        "truncated": False,
        "info": getattr(obs, "info", {})
    }

@app.get("/state", tags=["Environment"])
def get_state():
    """Returns the current verifiable state for evaluation consistency."""
    return env.state()

# ------------------------------------------------------------------------------
# EXTRA UTILITIES
# ------------------------------------------------------------------------------

@app.get("/ui", tags=["UI"])
def ui_redirect():
    """Standard landing page for human evaluation."""
    return HTMLResponse(content="""
    <html>
        <head>
            <title>AdVision - UI Redirect</title>
            <meta http-equiv="refresh" content="0; url=/ui/">
        </head>
        <body>
            <p>Redirecting to <a href="/ui/">Human Evaluation UI</a>...</p>
        </body>
    </html>
    """, status_code=200)

# --- Gradio UI ---
import gradio as gr  # noqa: E402
from server.ui import demo  # noqa: E402
app = gr.mount_gradio_app(app, demo, path="/ui")

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    # Note: Using reload=False in production for performance
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
