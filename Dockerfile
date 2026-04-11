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

# ── Python dependencies ───────────────────────────────────────────────────────
# Install in two phases: heavy ML first (so cache layer is reused), then app deps
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    openenv-core \
    openai \
    "pydantic>=2.0" \
    pyyaml \
    gymnasium \
    ultralytics \
    opencv-python-headless \
    python-dotenv \
    fastapi \
    uvicorn \
    matplotlib \
    scipy \
    Pillow \
    "imageio[ffmpeg]" \
    stable-baselines3

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . /app

# ── Install advision package ──────────────────────────────────────────────────
RUN pip install --no-cache-dir -e .

# ── Pre-download YOLO weights (avoids 1st-run download inside container) ──────
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# ── Create required data dirs ─────────────────────────────────────────────────
RUN mkdir -p /app/data/input_videos /app/data/ad_images

# ── Expose port for Gradio UI ─────────────────────────────────────────────────
EXPOSE 7860

# ── Default: run the FastAPI server (deployment standard) ───────────────────
# We use server.app:app because the FastAPI instance is located in server/app.py
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
