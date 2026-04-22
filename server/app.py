import os
import sys
import tempfile
from typing import Any, Dict

from fastapi import FastAPI, Body, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import cv2
import numpy as np

# --- PATH SETUP (Must be before internal imports) ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server.advision_environment import AdVisionEnvironment  # noqa: E402
from advision_env.models import AdVisionAction  # noqa: E402
from advision_env.pipeline.placement_engine import PlacementConfig # noqa: E402
from server.ui_utils import run_processing_pipeline # noqa: E402

# Initialize the environment singleton
env = AdVisionEnvironment()

# Initialize FastAPI
app = FastAPI(
    title="AdVision OpenEnv Server",
    description="High-performance Ad Placement Environment for OpenEnv Hackathon",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ------------------------------------------------------------------------------
# PREMIUM UI & API
# ------------------------------------------------------------------------------

@app.get("/", tags=["UI"], response_class=HTMLResponse)
def root_ui():
    """Serves the premium custom UI."""
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "UI files not found. Please run the setup."

@app.post("/api/process", tags=["API"])
async def api_process(
    video: UploadFile = File(...),
    ad: UploadFile = File(...),
    scale: float = Form(1.4),
    rotation: float = Form(0.0),
    tilt: float = Form(0.0),
    alpha: float = Form(0.97),
    feather: float = Form(22.0),
    shadow: float = Form(0.4)
):
    """API endpoint for the custom UI to process videos."""
    # Save uploaded files to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        shutil.copyfileobj(video.file, tmp_video)
        video_path = tmp_video.name

    # Read ad image
    ad_bytes = await ad.read()
    nparr = np.frombuffer(ad_bytes, np.uint8)
    ad_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cfg = PlacementConfig(
        scale=scale,
        rotation_deg=rotation,
        perspective_tilt=tilt,
        alpha=alpha,
        feather_px=int(feather),
        shadow_strength=shadow,
        enable_shadow=(shadow > 0)
    )

    try:
        out_path, metrics, frames = run_processing_pipeline(video_path, ad_bgr, cfg)
        
        # In a real HF space, we might want to serve this via a route
        # For now, we'll return a path that our static server can reach
        # But since out_path is outside static, we'll copy it to static/temp
        temp_dir = os.path.join(STATIC_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        final_video_name = os.path.basename(out_path)
        final_video_path = os.path.join(temp_dir, final_video_name)
        shutil.copy(out_path, final_video_path)

        return {
            "video_url": f"/static/temp/{final_video_name}",
            "metrics": metrics,
            "frames": frames
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

# ------------------------------------------------------------------------------
# CORE API ENDPOINTS (OpenEnv Specification v0.2.3)
# ------------------------------------------------------------------------------

@app.get("/health", tags=["System"])
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
    options = payload.get("options")
    obs, info = env.reset(seed=seed, options=options)
    return {"observation": obs, "info": info}

@app.post("/step", tags=["Environment"])
def step(action: AdVisionAction):
    """Executes a single environment step."""
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return {
        "observation": obs,
        "reward": float(reward) if reward is not None else 0.0,
        "done": done,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    }

@app.get("/state", tags=["Environment"])
def get_state():
    """Returns the current verifiable state for evaluation consistency."""
    return env.state()

# ------------------------------------------------------------------------------
# GRADIO BACKUP
# ------------------------------------------------------------------------------

@app.get("/ui", tags=["UI"])
def ui_redirect():
    """Legacy redirect."""
    return HTMLResponse(content="""
    <html><head><meta http-equiv="refresh" content="0; url=/"></head>
    <body>Redirecting to <a href="/">Premium UI</a>...</body></html>
    """, status_code=200)

# --- Gradio UI ---
import gradio as gr  # noqa: E402
from server.ui import demo  # noqa: E402
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

