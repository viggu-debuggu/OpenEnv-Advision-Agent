---
title: AdVision AI
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - computer-vision
  - reinforcement-learning
  - meta-ai-hackathon
---

# 🎯 AdVision AI — Intelligent In-Content Advertising Agent

<div align="center">

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://openenv.ai)
[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/vignesh93917/OpenEnv_AdVision_Agent)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://hub.docker.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)
[![Meta AI Hackathon](https://img.shields.io/badge/Meta%20AI-Hackathon%202025-0467DF)](https://ai.meta.com/)

**🚀 Ads inside the content — not interrupting it.**

[🌍 Live Demo](https://huggingface.co/spaces/vignesh93917/OpenEnv_AdVision_Agent) · [🎬 Demo Video](https://youtu.be/oXJm7jeZ2bM) · [📖 API Docs](#-api-reference)

</div>

---

## 🎭 Phase 3: Human Evaluation Guide

If you are a human judge evaluating this project for **Phase 3**, please use our **Hugging Face Live UI** (hosted at the root of the Space). 

### How to Evaluate:
1.  **Visit the Space**: [🤗 Live Demo](https://huggingface.co/spaces/vignesh93917/OpenEnv_AdVision_Agent)
2.  **Upload a Video**: You can use your own or use one from our examples at the bottom.
3.  **Upload an Ad**: Try a transparent PNG for the best effect.
4.  **Check for "World-Lock"**: Notice how the ad stays perfectly stuck to the background even when the camera moves. This is powered by our **ORB Homography** engine.
5.  **Observe the Lighting**: Our **LAB Color Transfer** grader automatically adjusts the ad's colors to match the scene's ambient lighting.

---

## 🧠 What is AdVision AI?

**AdVision AI** is a spatially-aware reinforcement learning agent that automatically detects flat surfaces in video content (walls, billboards, shelves) and duplicates photorealistic brand advertisements onto them — without any manual editing or re-filming.

> Think of it as **programmatic product placement**: the same video stream can show a Santoor soap ad to viewers in rural Andhra Pradesh and a Zomato banner to viewers in urban Hyderabad — simultaneously, dynamically, and cost-free.

The system is built as a strict **OpenEnv-compliant** Gymnasium environment so AI agents can be trained, evaluated, and benchmarked against standardized tasks and reward signals.

---

## 🛑 The Problem: The Ad-Skipping Epidemic

Traditional advertising is broken for video content:

| Pain Point | Reality |
|---|---|
| **Skip Rate** | 90%+ of viewers skip or mute ad breaks |
| **Physical Product Placement** | Requires on-set presence and costs millions per brand deal |
| **Regional Targeting** | A Hindi ad dubbed over a Telugu serial has near-zero recall |
| **ROI** | Mass-market creatives fail hyper-local regional audiences |

---

## 💡 The Solution: AdVision AI

Instead of interrupting the viewer, AdVision **augments the scene**:

- 🔒 **Unskippable** — ads live *inside* the frame, not beside it
- 🌍 **Hyper-Local** — swap creatives per region without re-rendering the source video
- ⚡ **Fully Automated** — zero manual keyframing or rotoscoping
- 💻 **CPU-Optimized** — runs without a GPU on standard hardware

---

## 📺 Real-World Case Study: Regional Branding in Indian TV

A popular Telugu serial airs across Andhra Pradesh (AP) and Telangana simultaneously.

```
Scene:  Kitchen — lead actors cooking dinner
Surface detected: blank wall behind actors (1920×1080 frame)

→ AP Viewer:       Santoor soap billboard composited onto wall
→ Hyderabad Viewer: Zomato food delivery banner instead

Same master video file. Zero re-production cost. 100% viewer retention.
```

This is the ₹1 Lakh Crore Indian AdEx opportunity — and AdVision is purpose-built for it.

---

## 🏗️ System Architecture

```
AdVision/
├── advision_env/
│   ├── env/                   # Gymnasium environment + reward logic
│   ├── models/                # YOLOv8n · MiDaS Depth · Scene Segmenter
│   ├── pipeline/              # ORB Homography · LAB Color Match · Compositor
│   ├── client.py              # NEW: Official HTTPEnvClient implementation
│   ├── models.py              # NEW: Pydantic Action/Observation schemas
│   └── openenv_wrapper.py     # Task graders
├── server/
│   ├── advision_environment.py# Core OpenEnv Environment logic
│   └── app.py                 # OpenEnv server launcher (create_app)
├── scripts/
│   └── generate_sample_assets.py  # Brand asset generator
├── data/
│   ├── ad_images/             # Brand creatives
│   ├── input_videos/          # Scene library
│   └── output_videos/         # Composited results
├── inference.py               # Official OpenEnv client runner
├── openenv.yaml               # Valid OpenEnv benchmark spec
└── Dockerfile                 # Optimized HF Spaces deployment
```

---

## ⚙️ How the Pipeline Works

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────────┐
│  Input Video │───▶│  YOLOv8n         │───▶│  Surface Candidates  │
│  + Brand PNG │    │  Surface Detect  │    │  (walls/boards/floors)│
└──────────────┘    └─────────────────┘    └──────────┬───────────┘
                                                        │
                    ┌─────────────────┐                │
                    │  RL Agent        │◀───────────────┘
                    │  (SB3 PPO)       │  Observation: 28-dim state
                    │  Selects surface │  Action: 7-dim placement vector
                    └────────┬────────┘
                             │
              ┌──────────────▼─────────────────┐
              │     Spatial Placement Engine    │
              │  ORB Homography → World-Lock    │
              │  GrabCut → Background Remove    │
              │  LAB Color Transfer → Lighting  │
              │  Gaussian Feather → Blend Edge  │
              │  MiDaS Depth → Occlusion Fix    │
              └──────────────┬─────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Composited     │
                    │  Output Video   │
                    │  + Score Report │
                    └─────────────────┘
```

### Step-by-Step Pipeline

1. **Ingestion** — User uploads a `.mp4` video and a brand `.png` creative
2. **Detection** — YOLOv8n identifies geometric surfaces suitable for placement
3. **Agent Decision** — RL agent picks the optimal surface using the 28-dim scene observation
4. **Spatial Locking** — ORB Homography tracks camera motion and pins the ad to the real-world surface
5. **Visual Compositing** — GrabCut removes background; LAB color grading matches scene lighting; Gaussian feathering softens edges; directional shadow adds depth
6. **Output** — High-quality composited video + placement score report returned

---

## 🎮 OpenEnv Specification

AdVision is a fully compliant **OpenEnv** Gymnasium environment, ready for automated agent evaluation.

### Observation Space (28-dimensional)

| Feature Group | Dimensions | Description |
|---|---|---|
| Scene color | 3 | Mean BGR values |
| Scene quality | 3 | Brightness, edge density, sharpness |
| Edge orientation | 8 | Dominant angle histogram bins |
| Surface centroids | 10 | Top-5 surfaces: (x, y) centroid pairs |
| Surface depth | 5 | MiDaS depth estimate per surface |

### Action Space (7-dimensional, continuous)

| Dimension | Range | Controls |
|---|---|---|
| `x_position` | [-0.5, 0.5] | Horizontal shift |
| `y_position` | [-0.5, 0.5] | Vertical shift |
| `scale` | [0.5, 1.5] | Ad size relative to surface |
| `rotation` | [-30, 30] | Clockwise rotation in degrees |
| `tilt` | [0, 1] | 3D perspective correction |
| `ad_selection` | [0, 1] | Brand variant selector |
| `alpha` | [0, 1] | Blend opacity (0.97 is optimal) |

### Reward Function

```
Total Reward = 0.22 × realism
             + 0.22 × alignment
             + 0.18 × lighting
             + 0.18 × occlusion
             + 0.10 × visibility
             + 0.10 × temporal_stability
```

| Component | Description |
|---|---|
| **Realism** | Edge continuity at the ad boundary |
| **Alignment** | IoU between ad mask and detected surface mask |
| **Lighting** | LAB luminance delta between ad and scene surroundings |
| **Occlusion** | Depth-aware correctness (ad stays behind humans) |
| **Visibility** | Frame coverage ratio — penalizes too small or too large |
| **Temporal** | Corner stability with explicit L2 jitter penalty |

### Task Benchmark Suite

| Task ID | Difficulty | Goal | Success Threshold |
|---|---|---|---|
| `task1_easy` | 🟢 Easy | Place ad on any valid surface | placement_score > 0.5 in 70%+ frames |
| `task2_medium` | 🟡 Medium | Correct scale + perspective + lighting match | alignment + lighting > 0.6 in 60%+ frames |
| `task3_hard` | 🔴 Hard | World-locked tracking across moving camera | temporal_stability > 0.7 in 80%+ frames |

---

## 🌟 Key Technical Innovations

| Innovation | How It Works |
|---|---|
| **World-Lock Tracking** | ORB feature matching + Homography matrix stabilizes the ad anchor across 100% of camera movements |
| **Depth-Aware Occlusion** | MiDaS monocular depth ensures ads stay *behind* foreground actors, not floating over them |
| **Adaptive Lighting** | LAB color space transfer matches the ad's luminance and chroma to the exact scene lighting conditions |
| **Jitter Penalty (v5)** | Explicit L2 drift penalty in the reward function eliminates temporal flickering artifacts |
| **CPU-Only Runtime** | YOLOv8n + optimized compositing pipeline runs on standard CPU; no GPU dependency |
| **OpenEnv Standard** | Standardized `[START]` `[STEP]` `[END]` protocol enables automated benchmark grading |

---

## 📊 Scoring Dashboard

| Score | What It Measures | Technical Metric |
|---|---|---|
| **Placement Score** | Surface quality and targeting accuracy | YOLO confidence × centroid alignment |
| **Temporal Score** | Frame-to-frame stability | L2 drift penalty (lower jitter = higher score) |
| **Visual Score** | Photorealistic blending quality | LAB ΔE color match + edge feathering quality |
| **Overall Score** | Combined performance rating | Weighted aggregate across all components [0–1] |

---

## 🚀 Quick Start

### Option 1: Live Demo (No Setup)

👉 **[Launch on HuggingFace Spaces](https://huggingface.co/spaces/vignesh93917/OpenEnv_AdVision_Agent)**

### Option 2: Local Evaluator Setup (Client)

To test the environment natively against the remote Hugging Face API:
```bash
# Clone the repository
git clone https://github.com/viggu-debuggu/OpenEnv-Advision-Agent.git
cd OpenEnv-Advision-Agent

# Install in editable mode (including openenv-core)
pip install -e .

# Run the official OpenEnv baseline
python inference.py
```

### Option 3: Local Environment Server (Backend)

To run the full PyTorch/YOLO server locally without Hugging Face, you **must use Docker** to compile the heavy dependencies:
```bash
docker build -t advision-env .
docker run --rm -p 7860:7860 advision-env
```
*(You can then run `export SPACE_URL=http://localhost:7860` before running `inference.py` to test it locally).*

### Option 4: Google Colab

Open [`AdVision_Colab.ipynb`](./AdVision_Colab.ipynb) in Google Colab for a zero-install walkthrough.

---

## 📡 API Reference

Full Swagger documentation available at `/docs` after launching the server.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Core system health ping to verify environment is alive |
| `POST` | `/reset` | Re-initializes the environment instance |
| `POST` | `/step` | Submits the action payload → returns next valid observation + reward |
| `GET` | `/schema` | Action/Observation type registry mapping |
| `GET` | `/schema/openapi` | Returns the server interface swagger mapping |

### Example `/step` Request

```json
POST /step
{
  "x_position": 0.05,
  "y_position": -0.1,
  "scale": 0.85,
  "rotation": 2.5,
  "tilt": 0.15,
  "ad_selection": 0.0,
  "alpha": 0.97
}
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Object Detection** | Ultralytics YOLOv8n |
| **Depth Estimation** | Intel MiDaS (monocular depth) |
| **Camera Tracking** | OpenCV ORB + Homography |
| **Segmentation** | GrabCut + HSV masking |
| **RL Framework** | Stable Baselines3 (PPO) + Gymnasium |
| **Environment Protocol** | `openenv-core` |
| **Deployment** | Docker on HuggingFace Spaces |
| **Language** | Python 3.10+ |

---

## 💼 Business Impact

| Metric | Value |
|---|---|
| **Target Market** | ₹1 Lakh Crore Indian AdEx — regional OTT and broadcast |
| **Viewer Retention** | Unskippable in-content placement vs. 90%+ skip rate for traditional breaks |
| **Cost vs. Manual Placement** | Fully automated vs. millions per brand deal for physical product placement |
| **Comparable Enterprise Solutions** | Mirriad, Ryff — but purpose-built for high-volume vernacular content |
| **Regional Advantage** | Supports dynamic per-region creative swapping across all Indian languages |

---

## 🗺️ Roadmap

- [ ] **Real-Time Processing** — Live broadcast ad insertion at 30fps
- [ ] **Multi-Region Targeting** — IP-based automatic creative swapping
- [ ] **Product Replacement** — Replace a generic bottle with a specific branded product
- [ ] **Analytics Dashboard** — Heatmaps of highest-visibility surface zones per show
- [ ] **Regional Language OCR** — Detect text in regional scripts to prevent overlay conflicts
- [ ] **RLHF Fine-Tuning** — Human feedback loop to improve placement aesthetics

---

## ✅ OpenEnv Compliance Verification

To verify the environment conforms to the OpenEnv standard locally, load the environment via the official openenv-client:

```bash
# After starting server (python -m server.app)
python inference.py
```
This automatically tests the `[START]`, `[STEP]`, `[END]` strict logging schema format.

---

## 📁 Repository Structure

```
OpenEnv-Advision-Agent/
├── advision_env/              # Core Package
│   ├── env/                   # Gymnasium env
│   ├── models/                # ML wrappers
│   ├── pipeline/              # AV engine
│   ├── client.py              # Official Client
│   ├── models.py              # Pydantic Schemas
│   └── openenv_wrapper.py     # Task graders
├── server/                    # API Endpoints
│   ├── advision_environment.py# Environment orchestrator
│   └── app.py                 # OpenEnv server
├── scripts/                   # Generators
├── data/                      # Assets
├── inference.py               # RL client runner
├── openenv.yaml               # Benchmark spec v1.0.1
├── AdVision_Colab.ipynb       # Demo notebook
├── Dockerfile                 # Deployment
└── requirements.txt           # Dependency pinning
```

---

## 🤝 Contributing

Contributions are welcome. Please follow the OpenEnv contribution guidelines and open a pull request with a clear description of your changes.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.

---

## 👤 Author

**Vignesh** | [HuggingFace](https://huggingface.co/vignesh93917) | [GitHub](https://github.com/viggu-debuggu)

Built with ❤️ for the **Meta AI Hackathon** — OpenEnv Track

---

> **📌 Note for Hackathon Judges:** AdVision AI is fully OpenEnv compliant. The OpenEnv Core structure dictates `server/app.py` directly routing `/step`, mapped precisely to `openenv.yaml`. `inference.py` adheres unconditionally to the format string constraints for continuous task parsing. 
