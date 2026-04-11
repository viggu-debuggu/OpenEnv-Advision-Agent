"""
inference.py - AdVision OpenEnv Baseline Inference Script (FULLY CORRECTED)
----------------------------------------------------------------------------
FIXES APPLIED:
  1. reward=:.2f (was :.1f) — evaluator requires 2 decimal places
  2. rewards=:.2f (was :.1f) — same
  3. Removed score= field from [END] line — not in spec
  4. Removed extra stdout prints (=====, task names, summaries) — moved to stderr
  5. HF_TOKEN raises ValueError if missing — strict compliance
  6. action= in [STEP] is a short readable string, not a full JSON blob
  7. env= in [START] uses 'advision' matching openenv.yaml name field
  8. Single task per run — reads TASK_NAME env var, runs exactly one episode
  9. [END] always emitted even on exception (try/finally)
 10. All [DEBUG] prints go to sys.stderr
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import openenv  # noqa: F401
except ImportError:
    pass


# FIX #6: Import path must match folder name on HF Space
# If your folder is advision/, change to: from advision_env.openenv_wrapper import ...
from advision_env.client import AdVisionEnv
from advision_env.openenv.tasks import (
    Task1_BasicPlacement,
    Task2_RealisticBlend,
    Task3_TemporalConsistency,
)
from models import Action, Reward

# ---------------------------------------------------------------------------
# Credentials — strictly following hackathon checklist
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

# FIX #5: Raise hard if HF_TOKEN is missing (spec requirement)
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

API_KEY = HF_TOKEN

# ---------------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------------
BENCHMARK             = "advision_env"   # FIX #7: matches openenv.yaml name field
MAX_STEPS             = 30
TASK_THRESHOLDS       = {"task1_easy": 0.70, "task2_medium": 0.60, "task3_hard": 0.80}
MAX_TOTAL_REWARD      = float(MAX_STEPS)

TASK_REGISTRY = {
    "task1_easy": {
        "id":     "task1_easy",
        "name":   "Basic Placement",
        "grader": Task1_BasicPlacement,
        "desc":   "Place ad on any detected surface (Placement reward > 0.5)",
    },
    "task2_medium": {
        "id":     "task2_medium",
        "name":   "Realistic Blend",
        "grader": Task2_RealisticBlend,
        "desc":   "Place ad with correct scale + lighting (Realism reward > 0.6)",
    },
    "task3_hard": {
        "id":     "task3_hard",
        "name":   "Temporal Consistency",
        "grader": Task3_TemporalConsistency,
        "desc":   "World-locked placement across moving video (Temporal > 0.7)",
    },
}

# FIX #8: Single task per run — read from env var
TASK_NAME = os.getenv("TASK_NAME", "task1_easy")

# ---------------------------------------------------------------------------
# Logging helpers — STRICTLY per hackathon spec
# FIX #1 & #2: reward and rewards formatted to 2 decimal places
# FIX #3: [END] has no score= field
# FIX #4: all extra info goes to stderr
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str):
    # ONLY this exact line to stdout
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    done_str = "true" if done else "false"
    err_str  = str(error).replace('\n', ' ') if error else "null"
    # FIX #1: reward=:.2f
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err_str}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    success_str  = "true" if success else "false"
    # FIX #2: rewards=:.2f
    rewards_str  = ",".join([f"{r:.2f}" for r in rewards])
    # FIX #3: no score= field — exactly: [END] success=... steps=... rewards=...
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# LLM action generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You control an ad placement agent. Output ONLY valid JSON:
{"x_position": float, "y_position": float, "scale": float,
 "rotation": float, "tilt": float, "ad_selection": float, "alpha": float}
"""

def _fallback_action(step: int, obs_dict: dict = None) -> str:
    return '{"x_position":0.0,"y_position":0.0,"scale":1.0,"rotation":0.0,"tilt":0.0,"ad_selection":0.0,"alpha":0.97}'


def action_to_str(act: dict) -> str:
    """FIX #6: Short readable action string, NOT a full JSON blob."""
    surf = int(act.get("ad_selection", 0) * 10)
    sc   = act.get("scale", 1.0)
    alp  = act.get("alpha", 0.97)
    xp   = act.get("x_position", 0.0)
    yp   = act.get("y_position", 0.0)
    return f"place(surf={surf},x={xp:.2f},y={yp:.2f},scale={sc:.2f},alpha={alp:.2f})"

def get_llm_action(client: OpenAI, step: int, obs_dict: dict,
                   last_reward: float, history: List[str], task_desc: str) -> Dict[str, float]:
    user_msg = (
        f"Task: {task_desc}\n"
        f"Step: {step}/{MAX_STEPS}\n"
        f"Last reward: {last_reward:.3f}\n"
        f"Scene type: {obs_dict.get('scene_type', 'unknown')}\n"
        f"Detected surfaces: {len(obs_dict.get('detected_surfaces', []))}\n"
        f"Placement score: {obs_dict.get('placement_score', 0.0):.3f}\n"
        f"History (last 3): {history[-3:]}\n\n"
        f"Choose action parameters to maximise the placement reward."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=200,
            timeout=15,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        # Verify it parses to dict, but return original string or reserialized string
        parsed = json.loads(text)
        required = ["x_position","y_position","scale","rotation","tilt","ad_selection","alpha"]
        for k in required:
            if k not in parsed:
                return _fallback_action(step, obs_dict)
                
        return json.dumps(parsed)
    except Exception as e:
        print(f"[DEBUG] LLM call failed (step {step}): {e} — using heuristic", file=sys.stderr)
        return _fallback_action(step, obs_dict)

# ---------------------------------------------------------------------------
# Single task runner
# FIX #8: runs exactly ONE episode
# FIX #9: [END] emitted in finally — always fires even on crash
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, env: AdVisionEnv, task: dict) -> None:
    task_id   = task["id"]
    grader_fn = task["grader"]
    task_desc = task["desc"]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    history: List[str]   = []
    episode_history: List[dict] = []
    steps_taken = 0
    last_reward = 0.0
    success     = False

    try:
        result = env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            act_str_json = get_llm_action(client, step, obs.model_dump(), last_reward, history, task_desc)
            
            # The evaluator checks the [STEP] line action string.
            # Convert JSON back to short format for logging:
            try:
                act_dict = json.loads(act_str_json)
                action_str = action_to_str(act_dict)
            except Exception:
                act_dict = json.loads(_fallback_action(step, obs.model_dump()))
                action_str = action_to_str(act_dict)

            err = None
            try:
                action = Action(**act_dict)
                result = env.step(action)

                obs = result.observation
                
                reward = float(result.reward)
                done   = bool(result.done)
                info   = result.info or {}
            except Exception as e:
                reward = 0.0
                done   = False
                info   = {}
                err    = str(e)

            last_reward = float(reward)
            rewards.append(last_reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=last_reward,
                     done=done, error=err)   # FIX #1 applied inside log_step

            history.append(f"Step {step}: reward={last_reward:+.3f}")
            episode_history.append({"step": step, "action": act_dict,
                                     "reward": last_reward, "info": info})
            if done:
                break

        # Score
        raw_val  = sum(rewards) / max(MAX_TOTAL_REWARD, 1e-9)
        raw_score = float(np.clip(raw_val, 0.0, 1.0))
        success  = raw_score >= TASK_THRESHOLDS.get(task_id, 0.5)

        # Grader
        grader_history = []
        for entry in episode_history:
            tr = entry["info"].get("typed_reward", None)
            if tr is None:
                rc = entry["info"].get("reward_components", {})
                tr = Reward(
                    placement_reward=rc.get("alignment", 0.0),
                    realism_reward=rc.get("realism", 0.0),
                    temporal_stability_reward=rc.get("temporal", 0.0),
                    occlusion_reward=rc.get("occlusion", 0.0),
                    penalty_for_flickering=0.0,
                )
            # Make sure we add reward_components to info for Tasks 2 and 3
            rc_dict = {
                "alignment": getattr(tr, 'placement_reward', 0.0),
                "lighting": getattr(tr, 'realism_reward', 0.0),
                "temporal": getattr(tr, 'temporal_stability_reward', 0.0),
                "occlusion": getattr(tr, 'occlusion_reward', 0.0)
            }
            info_dict = {"typed_reward": tr, "reward_components": rc_dict}
            grader_history.append({"info": info_dict})

        try:
            task_obj = grader_fn()          # instantiate the task class
            for entry in grader_history:
                tr = entry["info"].get("typed_reward")
                if tr:
                    if task_id == "task1_easy":
                        r = getattr(tr, 'placement_reward', 0.0)
                    elif task_id == "task2_medium":
                        r = (getattr(tr, 'placement_reward', 0.0) + getattr(tr, 'realism_reward', 0.0)) / 2.0
                    else:  # task3_hard
                        r = getattr(tr, 'temporal_stability_reward', 0.0)
                    task_obj.update(float(r), entry["info"])
            grader_score = float(task_obj.grade().score)
        except Exception as ge:
            print(f"[DEBUG] Grader error: {ge}", file=sys.stderr)
            grader_score = raw_score

        print(f"[DEBUG] Grader score: {grader_score:.4f}", file=sys.stderr)

    finally:
        try:
            if hasattr(env, 'close'):
                env.close()
        except Exception:
            pass

        # FIX #9: always emitted, even on exception
        # FIX #3: no score= field
        # FIX #2: rewards formatted to 2dp inside log_end
        log_end(success=success, steps=steps_taken, rewards=rewards)

# ---------------------------------------------------------------------------
# Main — FIX #8: single task per invocation
# ---------------------------------------------------------------------------

def main() -> None:
    task = TASK_REGISTRY.get(TASK_NAME)
    if task is None:
        print(f"[ERROR] Unknown TASK_NAME='{TASK_NAME}'. "
              f"Valid: {list(TASK_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    print(f"[DEBUG] Starting task={TASK_NAME} model={MODEL_NAME}", file=sys.stderr)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    SPACE_URL = os.getenv("SPACE_URL", "ws://localhost:7860")
    
    async_env = AdVisionEnv(base_url=SPACE_URL)
    env = async_env.sync()

    with env:
        run_task(client, env, task)


if __name__ == "__main__":
    main()
