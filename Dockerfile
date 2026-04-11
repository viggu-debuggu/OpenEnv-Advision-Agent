FROM python:3.10-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV DISPLAY="" PYTHONUNBUFFERED=1

# ── Layer 1: Agent-only dependencies (slim runtime) ───────────────────────────
# inference.py only needs these to talk to the evaluator.
RUN pip install --no-cache-dir \
    "openenv-core>=0.2.3" \
    "openai>=1.0.0" \
    "pydantic>=2.0,<3.0" \
    "python-dotenv>=1.0.0" \
    "requests>=2.28.0"

# ── Layer 2: Heavy ML Stack (separated to prevent RAM/build issues) ───────────
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu torchvision==0.16.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Layer 3: Application Server & Extra packages ──────────────────────────────
RUN pip install --no-cache-dir \
    "ultralytics>=8.0.0" \
    "opencv-python-headless>=4.8.0" \
    "fastapi>=0.100.0" \
    "uvicorn>=0.24.0" \
    "scipy>=1.10.0" \
    "Pillow>=10.0.0" \
    "imageio[ffmpeg]>=2.37.3" \
    "numpy>=1.24.0" \
    "pyyaml>=6.0" \
    gradio

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . /app

# ── Metadata ──────────────────────────────────────────────────────────────────
# Ensure the package is installed in editable mode for model visibility
RUN pip install --no-cache-dir -e .

# Pre-warm YOLO weights
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true
RUN mkdir -p /app/data/input_videos /app/data/ad_images

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 7860
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
