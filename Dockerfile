FROM python:3.10-slim

# ── System dependencies ───────────────────────────────────────────────────────
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
ENV DISPLAY="" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# ── Layer 1: Agent-only deps (inference.py needs ONLY these) ─────────────────
RUN pip install \
    "openenv-core>=0.2.3" \
    "openai>=1.0.0" \
    "pydantic>=2.0,<3.0" \
    "python-dotenv>=1.0.0" \
    "requests>=2.28.0" \
    "anthropic>=0.18.0"

# ── Layer 2: CPU-only PyTorch (NO CUDA = saves ~2GB RAM) ─────────────────────
# This is critical for staying within the 8000MB evaluator limit.
RUN pip install \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Layer 3: Server & Vision deps (Includes Gradio for the HF Space UI) ──────
RUN pip install \
    "ultralytics>=8.0.0" \
    "opencv-python-headless>=4.8.0" \
    "fastapi>=0.100.0" \
    "uvicorn>=0.24.0" \
    "scipy>=1.10.0" \
    "Pillow>=10.0.0" \
    "imageio[ffmpeg]>=2.37.3" \
    "numpy>=1.24.0" \
    "pyyaml>=6.0" \
    "gradio"

# ── Copy project files ────────────────────────────────────────────────────────
COPY . /app

# ── Install project package (editable) ───────────────────────────────────────
RUN pip install -e .

# ── Pre-warm YOLO weights ─────────────────────────────────────────────────────
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# ── Create required data directories ─────────────────────────────────────────
RUN mkdir -p /app/data/input_videos /app/data/ad_images

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Runtime ───────────────────────────────────────────────────────────────────
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]
