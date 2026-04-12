---
title: AdVision AI — OpenEnv RL Environment
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - computer-vision
  - reinforcement-learning
  - meta-ai-hackathon
  - ad-placement
  - spatial-ai
---

# 🎯 AdVision AI — OpenEnv RL Environment

### Meta PyTorch OpenEnv Hackathon | Real-World Spatial AI | In-Content Ad Placement

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://openenv.ai)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/vignesh93917/OpenEnv_AdVision_Agent)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue)](https://hub.docker.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AdVision AI** is a fully OpenEnv-compliant Reinforcement Learning environment for training agents to place advertisements realistically inside video content — on walls, billboards, and surfaces — without interrupting the viewer experience.

🌍 **Live Space:** https://huggingface.co/spaces/vignesh93917/OpenEnv_AdVision_Agent  
🎬 **Demo Video:** https://www.youtube.com/watch?v=8dDctULc_Yw  
📦 **GitHub:** https://github.com/viggu-debuggu/OpenEnv-Advision-Agent

---

## 🛑 The Problem: The Ad-Skipping Epidemic

In the digital era, traditional advertisement breaks are failing:

- **90%+ skip rate** — viewers skip or ignore commercial breaks on TV and YouTube
- **High Costs** — traditional product placement requires physical presence on set and costs millions
- **Low ROI** — mass-market ads fail in hyper-local regional markets (e.g., a Hindi ad in a Telugu serial)
- **Brand Erosion** — brands are losing visibility in an increasingly cluttered market

## 💡 The Solution: AdVision AI

AdVision turns every video frame into a potential ad surface. Instead of interrupting the show, we **augment** it:

- **Unskippable** — ads are placed *inside* the content on walls, billboards, shelves
- **Spatially Aware** — ORB Homography keeps the ad world-locked through camera motion
- **Hyper-Local** — swap ads dynamically by region without re-filming
- **RL-Trained** — an agent learns optimal placement for maximum visibility and realism

### Real-World Case Study: Telugu Serial Targeting

A popular Telugu serial broadcasts across Andhra Pradesh and Telangana. The AI detects a blank wall behind actors in a kitchen scene. In rural AP it surfaces a **Santoor soap** billboard; in urban Hyderabad it shows a **Zomato** banner. Same video content — multi-regional brand targeting — 100% viewer retention.

---

## ⚙️ Environment Overview

AdVision is a **multi-step RL environment** following the OpenEnv `reset → step → done` loop. An agent observes the current video frame features and detected surfaces, then outputs a 7-dimensional continuous action to place an advertisement. The environment returns a scalar reward based on placement quality, realism, and temporal stability.

### Environment Spec

| Property | Value |
|---|---|
| **Type** | Continuous action, image-based observation |
| **Action Space** | 7-dimensional continuous (Box) |
| **Observation Space** | 28-dimensional float vector + surface metadata |
| **Reward Range** | [0.0, 1.0] per step |
| **Max Steps** | 10 per episode |
| **Tasks** | 3 (Basic Placement, Realistic Blend, Temporal Consistency) |
| **OpenEnv Protocol** | `reset/step/state/health` via FastAPI + WebSocket |

---

## 🎮 Action Space

The agent outputs a 7-dimensional action controlling the full spatial placement:

| Field | Type | Range | Description |
|---|---|---|---|
| `x_position` | float | -0.5 to 0.5 | Horizontal shift on the surface |
| `y_position` | float | -0.5 to 0.5 | Vertical shift on the surface |
| `scale` | float | 0.5 to 1.5 | Ad size relative to surface |
| `rotation` | float | -30.0 to 30.0 | Clockwise rotation in degrees |
| `tilt` | float | 0.0 to 1.0 | Perspective tilt for 3D effect |
| `ad_selection` | float | 0.0 to 1.0 | Which ad variant to display |
| `alpha` | float | 0.0 to 1.0 | Blend opacity (0.97 = near-opaque) |

---

## 👁️ Observation Space

The 28-dimensional observation captures scene dynamics per frame:

| Field | Dimensions | Description |
|---|---|---|
| `mean_b`, `mean_g`, `mean_r` | 3 | Mean BGR colour channels |
| `brightness` | 1 | Overall scene luminance |
| `edge_density` | 1 | Proportion of edge pixels |
| `sharpness` | 1 | Laplacian variance (focus metric) |
| `dominant_angles` | 5 | Histogram of dominant edge orientations |
| `surface_centroids` | 10 | X/Y centroid of top-5 detected surfaces |
| `surface_depths` | 5 | MiDaS estimated depth per surface |
| `detected_surfaces` | metadata | List of surface dicts with area, confidence, bounding box |
| `scene_type` | str | Detected scene category (indoor/outdoor/urban) |
| `placement_score` | float | Current composite reward score |

---

## 🏆 Reward Structure

AdVision uses a composite reward combining three components:

| Component | Weight | Metric | Description |
|---|---|---|---|
| **Placement Reward** | 0.4 | YOLO Confidence × Centroid Alignment | Is the ad on a valid surface? |
| **Realism Reward** | 0.35 | LAB ΔE Color Match + Edge Feathering | Does the ad look realistic? |
| **Temporal Stability** | 0.25 | L2 Drift Penalty across frames | Does the ad stay world-locked? |

**Total reward = 0.4 × placement + 0.35 × realism + 0.25 × temporal**

### Task Thresholds

| Task | Difficulty | Success Threshold | Primary Metric |
|---|---|---|---|
| `task1_easy` | Easy | reward ≥ 0.70 | Placement on any detected surface |
| `task2_medium` | Medium | reward ≥ 0.60 | Correct scale + lighting blend |
| `task3_hard` | Hard | reward ≥ 0.80 | World-locked across moving video |

---

## 📊 Baseline Scores

Evaluated using the Claude Haiku Agent (Seed 42) on a 30-frame reference video.

| Task ID | Difficulty | Score | Passed | Mean Reward |
|---|---|---|---|---|
| `task1_basic_placement` | Easy | **1.0000** | ✅ PASS | 0.50 |
| `task2_realistic_blend` | Medium | **0.0000** | ❌ FAIL | 0.50 |
| `task3_temporal_consistency` | Hard | **0.9167** | ❌ FAIL | 0.50 |

**Average Baseline Score: 0.6389**

---

---

## 🚀 Quick Start

### Install the Client

```bash
pip install openenv-core
```

### Connect and Run (Async)

```python
import asyncio
from advision_env.client import AdVisionEnv

async def main():
    async with AdVisionEnv(base_url="wss://vignesh93917-openenv-advision-agent.hf.space") as env:
        result = await env.reset()
        obs = result.observation

        for step in range(10):
            action = {
                "x_position": 0.0,
                "y_position": 0.0,
                "scale": 1.0,
                "rotation": 0.0,
                "tilt": 0.0,
                "ad_selection": 0.0,
                "alpha": 0.97,
            }
            result = await env.step(action)
            print(f"Step {step+1}: reward={result.reward:.2f} done={result.done}")
            if result.done:
                break

asyncio.run(main())
```

### Connect and Run (Sync)

```python
from advision_env.client import AdVisionEnv

env = AdVisionEnv(base_url="wss://vignesh93917-openenv-advision-agent.hf.space").sync()

with env:
    result = env.reset()
    obs = result.observation

    for step in range(10):
        action = {
            "x_position": 0.1,
            "y_position": -0.1,
            "scale": 1.2,
            "rotation": 2.0,
            "tilt": 0.1,
            "ad_selection": 0.5,
            "alpha": 0.97,
        }
        result = env.step(action)
        print(f"Step {step+1}: reward={result.reward:.2f} done={result.done}")
        if result.done:
            break
```

### Run the Inference Agent

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export TASK_NAME=task1_easy

python inference.py
```

---

## 📊 Sample Evaluator Output

```
[START] task=task1_easy env=advision_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=place(surf=0,x=0.00,y=0.00,scale=1.00,alpha=0.97) reward=0.45 done=false error=null
[STEP] step=2 action=place(surf=3,x=0.10,y=-0.05,scale=1.10,alpha=0.97) reward=0.58 done=false error=null
[STEP] step=3 action=place(surf=5,x=0.12,y=-0.08,scale=1.15,alpha=0.97) reward=0.67 done=false error=null
[STEP] step=4 action=place(surf=5,x=0.14,y=-0.10,scale=1.20,alpha=0.97) reward=0.71 done=false error=null
[STEP] step=5 action=place(surf=5,x=0.14,y=-0.10,scale=1.20,alpha=0.97) reward=0.73 done=true error=null
[END] success=true steps=5 rewards=0.45,0.58,0.67,0.71,0.73
```

---

## 🖥️ Server Setup

### Docker (Recommended)

```bash
docker build -t advision-env:latest .
docker run --rm -p 7860:7860 advision-env:latest
curl http://localhost:7860/health
```

Health response:
```json
{"status": "healthy", "service": "advision-env", "yolo_ready": true}
```

### Without Docker

```bash
git clone https://huggingface.co/spaces/vignesh93917/OpenEnv_AdVision_Agent
cd OpenEnv_AdVision_Agent
pip install -r requirements.txt
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — confirms YOLO model is loaded |
| `POST` | `/reset` | Start a new episode, returns initial observation |
| `POST` | `/step` | Submit action, returns observation + reward + done |
| `GET` | `/state` | Current episode state (frame index, surfaces, scores) |
| `GET` | `/schema` | Action and observation JSON schema |
| `GET` | `/docs` | Swagger UI for interactive API testing |
| `GET` | `/` | Gradio interactive UI for human evaluation |

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ Mandatory | — | Hugging Face API token for LLM calls |
| `API_BASE_URL` | Optional | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | Optional | `Qwen/Qwen2.5-72B-Instruct` | LLM model for agent decisions |
| `TASK_NAME` | Optional | `task1_easy` | Which task to run (`task1_easy`, `task2_medium`, `task3_hard`) |
| `OPENENV_URL` | Injected by evaluator | — | Evaluator's environment server URL |
| `ADVISION_YOLO_SIZE` | Optional | `n` | YOLO model size: n/s/m/l |
| `ADVISION_USE_MIDAS` | Optional | `0` | Enable MiDaS depth estimation (1=on) |

---

## 📂 Project Architecture

```
AdVision/
├── advision_env/                  # Core OpenEnv environment package
│   ├── client.py                  # EnvClient subclass (AdVisionEnv)
│   ├── env/                       # Gymnasium environment + reward logic
│   │   └── ad_placement_env.py    # AdPlacementEnv (v5 Jitter Penalty)
│   ├── models/                    # YOLOv8, MiDaS depth, scene segmenter
│   ├── pipeline/                  # ORB Homography + LAB colour matching
│   ├── openenv/
│   │   └── tasks.py               # Task1_BasicPlacement, Task2, Task3
│   └── openenv_wrapper.py         # OpenEnv glue layer
├── server/
│   ├── app.py                     # FastAPI server (Gradio UI + evaluator API)
│   └── advision_environment.py    # OpenEnv Environment base class impl
├── scripts/
│   └── generate_sample_assets.py  # Auto-generate sample brand ad images
├── data/
│   ├── ad_images/                 # Brand graphics (.png)
│   └── input_videos/              # Scene videos (.mp4)
├── inference.py                   # ✅ OpenEnv evaluator-compatible agent
├── models.py                      # Pydantic models: Action, Observation, Reward
├── openenv.yaml                   # OpenEnv environment specification
├── entrypoint.sh                  # Smart entrypoint: agent mode vs server mode
├── Dockerfile                     # CPU-optimised Docker (fits 8GB RAM)
├── pyproject.toml                 # Package metadata
├── tests/                         # ✅ Unit and validation tests
│   └── test_advision_env.py       # Core loop sanity checks
```

---

## 🧪 Testing & Validation

The `tests/` directory contains unit tests verifying the OpenEnv interface and basic loop execution:
```bash
pytest tests/
```

Before submission, you **must validate** the environment layout and APIs:
```bash
openenv validate .
```

---

## 📄 OpenEnv Specification (`openenv.yaml`)

Our environment configuration dictates exactly how agents and evaluators interact with the system:

```yaml
name: advision
version: 1.0.0
description: >
  AdVisionEnv is a real-world spatial AI environment for evaluating agents
  on advertisement placement in video scenes. Agents must detect surfaces,
  select optimal placement parameters, and maintain temporal consistency
  across frames — modeling a genuine media-production workflow.

tags:
  - computer-vision
  - advertising
  - spatial-reasoning
  - video-processing
  - real-world

spec_version: "1"
type: environment
runtime: docker
app: server.app:app
port: 7860

environment:
  python:
    entrypoint: server.advision_environment:AdVisionEnvironment
    baseline_script: inference.py

  episode:
    max_steps: 30
    reward_range: [0.0, 1.0]

  action_space:
    type: object
    description: "7-dimensional continuous action space for ad placement control"
    fields:
      x_position:   {type: float, range: [-0.5, 0.5],   description: "Horizontal shift fraction"}
      y_position:   {type: float, range: [-0.5, 0.5],   description: "Vertical shift fraction"}
      scale:        {type: float, range: [0.5, 1.5],    description: "Ad size relative to surface"}
      rotation:     {type: float, range: [-30.0, 30.0], description: "Clockwise rotation (degrees)"}
      tilt:         {type: float, range: [0.0, 1.0],    description: "Perspective tilt severity"}
      ad_selection: {type: float, range: [0.0, 1.0],    description: "Ad variant selector"}
      alpha:        {type: float, range: [0.0, 1.0],    description: "Blend opacity (0.97 = near-opaque)"}

  observation_space:
    type: object
    description: "Rich scene observation combining detected surfaces, scene features, and placement score"
    fields:
      detected_surfaces: {type: array,  description: "Detected placeable surfaces with bbox/centroid/depth"}
      scene_type:        {type: string, description: "Scene classification: indoor / outdoor / urban"}
      placement_score:   {type: float,  description: "Current step placement quality [0,1]"}
      frame_features:    {type: object, description: "Brightness, edge density, sharpness, mean BGR"}
      raw_obs:           {type: array,  description: "28-dim numerical feature vector"}

tasks:
  - id: task1_easy
    name: "Basic Surface Placement"
    difficulty: easy
    description: >
      Place the advertisement on any valid detected surface in the scene.
      Success requires the ad to predominantly overlap a geometric plane structure
      with placement reward > 0.5 in at least 70% of frames.
    grader: advision_env.openenv_wrapper:grade_task1
    success_threshold: 0.70
    min_frames: 10

  - id: task2_medium
    name: "Realistic Blend Placement"
    difficulty: medium
    description: >
      Place the ad with correct scale, perspective mapping, and lighting match.
      The grader evaluates alignment + lighting distributions, requiring both
      to exceed 0.60 in at least 60% of frames.
    grader: advision_env.openenv_wrapper:grade_task2
    success_threshold: 0.60
    min_frames: 15

  - id: task3_hard
    name: "Photorealistic Temporal Tracking"
    difficulty: hard
    description: >
      World-lock the advertisement across a moving camera scene with full
      temporal consistency. Requires ORB-homography tracking to keep corners
      pinned to the same real-world surface point. Penalises L2 drift and flickering.
      Temporal stability must exceed 0.70 in 80%+ of frames.
    grader: advision_env.openenv_wrapper:grade_task3
    success_threshold: 0.80
    min_frames: 30

reward:
  type: weighted_sum
  components:
    realism:    {weight: 0.22, description: "Edge continuity at ad boundary"}
    alignment:  {weight: 0.22, description: "IoU between ad mask and surface mask"}
    lighting:   {weight: 0.18, description: "LAB luminance match to scene surroundings"}
    occlusion:  {weight: 0.18, description: "Depth-aware occlusion correctness"}
    visibility: {weight: 0.10, description: "Frame coverage ratio"}
    temporal:   {weight: 0.10, description: "Corner stability with explicit jitter penalty"}

docker:
  image: advision-env
  build: "docker build -t advision-env ."
  run:   "docker run --rm -e HF_TOKEN=$HF_TOKEN -e TASK_NAME=$TASK_NAME advision-env python inference.py"
```

---

## ⚙️ How the Pipeline Works

1. **Ingestion** — Video frame is extracted and passed to the detection pipeline
2. **Detection** — YOLOv8n identifies placeable surfaces (walls, boards, floors)
3. **Observation** — 28-dim state vector is constructed from frame features + surfaces
4. **Agent Decision** — LLM or RL agent outputs a 7-dim placement action
5. **Placement** — ORB Homography locks the ad to the detected surface
6. **Visual Blending** — GrabCut + LAB Color Matching + Gaussian Feathering create realism
7. **Reward** — Composite score (placement + realism + temporal stability) is returned
8. **Iteration** — Agent refines placement over up to 10 steps per episode

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| **Surface Detection** | Ultralytics YOLOv8n |
| **Spatial Tracking** | OpenCV ORB + Homography |
| **Depth Estimation** | MiDaS (optional) |
| **Segmentation** | GrabCut + HSV Masking |
| **RL Framework** | Stable Baselines3 + Gymnasium |
| **LLM Agent** | OpenAI-compatible client (Qwen2.5-72B) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Gradio 6.0 |
| **Deployment** | Docker on HF Spaces (CPU-only) |
| **OpenEnv** | openenv-core >= 0.2.3 |

---

## 📈 Business Impact

| Metric | Value |
|---|---|
| **Market Size** | ₹1 Lakh Crore Indian AdEx market |
| **Skip Rate Problem** | 90%+ of viewers skip traditional ad breaks |
| **Comparable Solutions** | Mirriad, Ryff (enterprise, $millions/year) |
| **AdVision Advantage** | Automated, CPU-optimized, hyper-local, open-source |

---

## 🗺️ Future Roadmap

- **Real-time Processing** — Live TV/broadcast ad insertion at 30fps
- **Multi-Region Targeting** — Automatic ad swapping based on viewer IP/location
- **Product Replacement** — Replace a generic bottle with a branded one in existing footage
- **Advanced Analytics** — Heatmaps of highest-visibility surfaces across a show
- **Regional Language OCR** — Detect text in regional languages to avoid placement overlap

---

## 🤝 Contributing & License

Contributions are welcome! Please follow the OpenEnv contribution guidelines.  
This project is licensed under the **MIT License**.

---

> **Note to Meta AI Hackathon judges:** AdVision is fully OpenEnv compliant.
> The environment exposes `/reset`, `/step`, `/state`, `/health`, and `/schema` endpoints.
> Run `python inference.py` with `HF_TOKEN` set to execute a full evaluation episode.
> The `entrypoint.sh` automatically switches between agent mode (when `OPENENV_URL` is injected by the evaluator) and server mode (normal HF Space operation).
