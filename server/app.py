import os
import sys
from typing import Any, Dict

from fastapi import Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from openenv.core.env_server import create_fastapi_app

# --- PATH SETUP (Must be before internal imports) ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server.advision_environment import AdVisionEnvironment  # noqa: E402
from advision_env.models import AdVisionAction, AdVisionObservation  # noqa: E402

# Initialize the environment singleton
env = AdVisionEnvironment()

# ------------------------------------------------------------------------------
# CORE API INITIALIZATION (Official OpenEnv Helper)
# ------------------------------------------------------------------------------
# Initialize FastAPI with the official OpenEnv Helper (passing CLASSES, not instances)
app = create_fastapi_app(AdVisionEnvironment, AdVisionAction, AdVisionObservation)

# Initialize FastAPI with customized identity
app.title = "AdVision OpenEnv Server"
app.description = "High-performance Ad Placement Environment for OpenEnv Hackathon"
app.version = "1.1.0"

# Enable CORS for metadata and UI accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# EXTRA UTILITIES & UI
# ------------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Liveness check for container orchestrators and validators."""
    return {"status": "ok", "service": "advision-openenv", "protocol": "REST + WebSocket"}

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
