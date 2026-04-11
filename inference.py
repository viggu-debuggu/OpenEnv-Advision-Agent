"""
inference.py - AdVision OpenEnv Baseline Inference Script
----------------------------------------------------------
* OFFICIAL OpenEnv pattern using typed HTTPEnvClient.
* NO raw requests. NO fake scores.
"""
import os
import json
from typing import List, Dict, Any

from openai import OpenAI
from advision_env import AdVisionEnv, AdVisionAction

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -- Credentials ----------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

API_KEY = HF_TOKEN

# -- Space URL Configuration ---------------------------------------------------
SPACE_URL = os.getenv("SPACE_URL", "https://vignesh93917-openenv-advision-agent.hf.space")

# -- Benchmark constants -------------------------------------------------------
BENCHMARK              = "AdVisionEnv"
MAX_STEPS              = 10

TASKS = [
    {
        "id":       "task1_easy",
        "name":     "Basic Placement",
        "desc":     "Place ad on any detected surface (Placement reward > 0.5)",
    },
    {
        "id":       "task2_medium",
        "name":     "Realistic Blend",
        "desc":     "Place ad with correct scale + lighting (Realism reward > 0.6)",
    },
    {
        "id":       "task3_hard",
        "name":     "Temporal Consistency",
        "desc":     "World-locked placement across moving video (Temporal > 0.7)",
    },
]

# -- Logging helpers ----------------------------------------------------------
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    done_str = "true" if done else "false"
    err_str = error.replace(" ", "_").replace("\n", "_")[:80] if error else "null"
    action_clean = action.replace(" ", "")
    print(f"[STEP]  step={step} action={action_clean} reward={reward:.2f} done={done_str} error={err_str}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    success_str = "true" if success else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END]   success={success_str} steps={steps} rewards={rewards_str}", flush=True)


# -- LLM action generation ----------------------------------------------------
SYSTEM_PROMPT = """You are an AI agent controlling an advertisement placement system.
You observe a video scene and must output action parameters to place an ad realistically.

Output ONLY valid JSON with these exact keys:
{
  "x_position":   float (-0.5 to 0.5),   // horizontal shift
  "y_position":   float (-0.5 to 0.5),   // vertical shift
  "scale":        float (0.5 to 1.5),    // ad size relative to surface
  "rotation":     float (-30.0 to 30.0), // clockwise rotation in degrees
  "tilt":         float (0.0 to 1.0),    // perspective tilt
  "ad_selection": float (0.0 to 1.0),    // which ad variant to use
  "alpha":        float (0.0 to 1.0)     // blend opacity
}
"""

def _fallback_action(obs: Any) -> AdVisionAction:
    act = AdVisionAction(
        x_position=0.0, y_position=0.0, scale=1.0,
        rotation=0.0, tilt=0.0, ad_selection=0.0, alpha=0.97
    )
    if hasattr(obs, "detected_surfaces") and obs.detected_surfaces:
        surfs = obs.detected_surfaces
        best = max(surfs, key=lambda s: s.get("area", 0))
        cx, cy = best.get("centroid", (320, 180))
        act.x_position = min(max((cx / 640.0) - 0.5, -0.5), 0.5)
        act.y_position = min(max((cy / 360.0) - 0.5, -0.5), 0.5)
    return act

def get_llm_action(client: OpenAI,
                   step: int,
                   obs: Any,
                   last_reward: float,
                   history: List[str],
                   task_desc: str) -> AdVisionAction:
    
    if API_KEY in ("dummy", "", None):
        return _fallback_action(obs)

    user_msg = (
        f"Task: {task_desc}\n"
        f"Step: {step}/{MAX_STEPS}\n"
        f"Last reward: {last_reward:.2f}\n"
        f"Scene: {getattr(obs, 'scene_type', 'unknown')}\n"
        f"Surfaces: {len(getattr(obs, 'detected_surfaces', []))}\n"
        f"Score: {getattr(obs, 'placement_score', 0.0):.2f}\n"
        f"History: {history[-3:]}\n\n"
        f"Choose action parameters."
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
        )
        text = resp.choices[0].message.content.strip()
        if "```" in text: text = text.split("```")[1].replace("json","")
        parsed = json.loads(text)
        return AdVisionAction(**parsed)
    except Exception:
        return _fallback_action(obs)


# -- Task Runner --------------------------------------------------------------
def run_task(llm_client: OpenAI, env_client: AdVisionEnv, task: dict):
    task_id   = task["id"]
    task_desc = task["desc"]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    history: List[str]   = []
    steps_taken = 0
    success = False

    try:
        obs = env_client.reset()
        
        for step in range(1, MAX_STEPS + 1):
            action = get_llm_action(llm_client, step, obs, 
                                   rewards[-1] if rewards else 0.0, 
                                   history, task_desc)
            
            action_json = action.model_dump_json()
            
            try:
                # Step the environment using the typed client
                obs, reward, done, info = env_client.step(action)
                err = None
            except Exception as e:
                reward = 0.0
                done = False
                err = str(e)

            rewards.append(float(reward))
            steps_taken = step
            log_step(step=step, action=action_json, reward=reward, done=done, error=err)
            history.append(f"Step {step}: {reward:+.2f}")
            
            if done: break

        # Success is determined by the final average reward or specific task threshold
        grader_score = sum(rewards) / max(len(rewards), 1)
        success = grader_score >= 0.5

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


def main():
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = AdVisionEnv(base_url=SPACE_URL)
    
    task_id = os.getenv("TASK_ID", "task1_easy")
    task = next((t for t in TASKS if t["id"] == task_id), TASKS[0])
    run_task(llm_client, env_client, task)

if __name__ == "__main__":
    main()
