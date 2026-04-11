FROM python:3.10-slim

# ── System dependencies for OpenCV, FFmpeg ──────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Environment variables ─────────────────────────────────────────────────────
ENV ADVISION_YOLO_SIZE=n
ENV ADVISION_USE_MIDAS=0
# Disable OpenCV display (headless container)
ENV DISPLAY=""
# Module 3 deployment standards
ENV WORKERS=4
ENV MAX_CONCURRENT_ENVS=100

# ── Phase 1: Agent-only lightweight deps (needed for inference.py) ───────────
# These are installed first so they are cached even if the server deps change.
# inference.py ONLY needs: openai, openenv-core, python-dotenv
RUN pip install --no-cache-dir \
    "openenv-core>=0.2.3" \
    "openai>=1.0.0" \
    "pydantic>=2.0,<3.0" \
    "python-dotenv>=1.0.0"

# ── Phase 2: CPU-only PyTorch (server needs it, agent does NOT import it) ────
# Force CPU to prevent RAM overflow — GPU variant (+cuda) would be ~3GB.
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu torchvision==0.16.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Phase 3: Remaining server / environment dependencies ─────────────────────
RUN pip install --no-cache-dir \
    pyyaml>=6.0 \
    gymnasium>=0.29.0 \
    "ultralytics>=8.0.0" \
    "opencv-python-headless>=4.8.0" \
    "fastapi>=0.100.0" \
    "uvicorn>=0.24.0" \
    "matplotlib>=3.7.0" \
    "scipy>=1.10.0" \
    "Pillow>=10.0.0" \
    "imageio[ffmpeg]>=2.37.3" \
    "stable-baselines3>=2.0.0" \
    "requests>=2.28.0" \
    "numpy>=1.24.0" \
    gradio

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . /app

# ── Install advision package ──────────────────────────────────────────────────
RUN pip install --no-cache-dir -e .

# ── Pre-download YOLO weights (avoids 1st-run download inside container) ──────
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# ── Create required data dirs ─────────────────────────────────────────────────
RUN mkdir -p /app/data/input_videos /app/data/ad_images

# ── Expose port for Gradio / FastAPI UI ──────────────────────────────────────
EXPOSE 7860

# ── Smart entrypoint ──────────────────────────────────────────────────────────
# If OPENENV_URL is set by the evaluator → agent mode (python inference.py)
# Otherwise → server mode (uvicorn FastAPI + Gradio)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
