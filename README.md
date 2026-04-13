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

**AdVision AI** is a state-of-the-art, OpenEnv-compliant Reinforcement Learning environment designed to bridge the gap between AI and media production. It allows agents to learn the complex art of **Contextual In-Content Advertising**—placing ads realistically on surfaces within video scenes without disrupting the user experience.

---

## 💡 The Vision: unskippable, Contextual, Spatial

In an era of ad-blockers and "Skip Ad" buttons, AdVision transforms every video frame into an interactive canvas. By leveraging Spatial AI, AdVision ensures that ads are:
- **Integrated**: Seamlessly blended into the scene geometry.
- **Dynamic**: Targeted and swappable based on metadata.
- **Stable**: World-locked across camera pans and zooms using ORB-homography.

---

## ⚙️ Environment Specification

AdVision follows the full **OpenEnv Specification**, providing a robust API for training and evaluation.

### Core Interface
| Property | Specification |
|:---|:---|
| **API** | `step(action)`, `reset(seed, episode_id)`, `state()` |
| **Action Space** | 7-Dimensional Continuous Box |
| **Observation** | 28-Dimensional Vector + Rich Surface Metadata |
| **Reward** | Composite [0.0, 1.0] with partial progress signal |
| **Max Steps** | 30 steps per episode |

### Action Space Detail
The agent controls the placement parameters with high precision:
- `x_position`, `y_position`: Centroid offset on the target surface `[-0.5, 0.5]`.
- `scale`: Relative size of the advertisement `[0.5, 1.5]`.
- `rotation`, `tilt`: Spatial orientation for perspective matching.
- `ad_selection`: Discrete choice of ad creative variants.
- `alpha`: Seamless blending opacity.

---

## 🏆 Evaluation Tasks

AdVision features three progressively difficult tasks, each with a deterministic agent grader.

````carousel
### Task 1: Basic Surface Placement
**Difficulty**: Easy
**Goal**: Place the ad on any valid detected surface with >0.5 placement reward.
**Success**: 70% of frames meeting criteria.
<!-- slide -->
### Task 2: Realistic Blend Placement
**Difficulty**: Medium
**Goal**: Align ad with correct scale, perspective, and lighting match.
**Success**: Alignment + Lighting > 0.60 in 60% of frames.
<!-- slide -->
### Task 3: Photorealistic Temporal Tracking
**Difficulty**: Hard
**Goal**: Pin the ad to a moving surface across a camera pan with zero drift.
**Success**: Temporal stability > 0.70 in 80% of frames.
````

---

## 📊 Baseline Performance

| Task | ID | Score | Status |
|:---|:---|:---|:---|
| 🟢 Easy | `task1_easy` | **1.00** | ✅ PASS |
| 🟡 Medium | `task2_medium` | **0.82** | ✅ PASS |
| 🔴 Hard | `task3_hard` | **0.75** | ⚠️ BORDERLINE |

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/viggu-debuggu/OpenEnv-Advision-Agent.git
cd AdVision
pip install -e .
```

### 2. Local Validation
```bash
openenv validate .
```

### 3. Running the Baseline Agent
The environment comes with a compliant `inference.py` script.
```bash
export HF_TOKEN="your_token"
export TASK_NAME="task1_easy"
python inference.py
```

---

## 📂 Project Structure

- `advision_env/`: Core environment logic and vision pipeline.
- `server/`: FastAPI server implementation for OpenEnv protocol.
- `inference.py`: Baseline inference script with [START]/[STEP]/[END] logging.
- `openenv.yaml`: Official environment metadata and task definitions.
- `Dockerfile`: Production-ready container configuration.

---

## 🤝 Acknowledgements

Developed for the Meta AI x PyTorch OpenEnv Hackathon. Special thanks to the OpenEnv team for the standardized framework.

---
© 2026 AdVision AI Team | Licensed under MIT.
