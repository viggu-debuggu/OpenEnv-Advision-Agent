"""
inference.py — OpenEnv Universal Agent (Synchronous Version)
----------------------------------------------------------
Official Ground-Truth pattern:
  - Uses AdVisionEnv (SyncEnvClient) synchronously.
  - Strict HF_TOKEN check (ValueError if missing).
  - Proper log format [START]/[STEP]/[END] with 1-space separator.
  - Universal Agent: Connects via OPENENV_URL.
"""
import os
import json
from typing import List, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from advision_env import AdVisionEnv, AdVisionAction

# ── Env vars (STRICT rules) ──────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

# Problem 5 Fix: Raise ValueError if missing
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

# ── Runtime constants ────────────────────────────────────────────────────────
EVAL_URL  = os.getenv("OPENENV_URL", os.getenv("SPACE_URL", "http://localhost:8000"))
BENCHMARK = os.getenv("BENCHMARK",   "advision_env")
TASK_ID   = os.getenv("TASK_ID",     "task1")
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))

# ── Logging (Exact format) ───────────────────────────────────────────────────
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Any = None):
    err_str = str(error).replace(" ", "_").replace("\n", "_")[:80] if error else "null"
    act_str = str(action).replace(" ", "")
    print(f"[STEP] step={step} action={act_str} reward={reward:.2f} "
          f"done={'true' if done else 'false'} error={err_str}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    r_str = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={'true' if success else 'false'} "
          f"steps={steps} rewards={r_str}", flush=True)

# ── LLM action generator ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a universal AI agent driving an OpenEnv environment.
Given the observation, output ONE action string on a single line.

Examples:
  click('id')
  place_ad(x=0.1, y=-0.2, scale=1.1)
  answer('Paris')

Return ONLY the action string. No explanation."""

def get_action(client: OpenAI, step: int, obs: Any, last_reward: float, history: List[str]) -> str:
    obs_str = json.dumps(obs, default=str) if isinstance(obs, dict) else str(obs)
    if len(obs_str) > 2000:
        obs_str = obs_str[:2000] + "...[truncated]"
        
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": 
                    f"Step: {step}/{MAX_STEPS}\nLast reward: {last_reward:.2f}\n"
                    f"Observation:\n{obs_str}\n\nNext action?"}
            ],
            temperature=0.2,
            max_tokens=80,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = "\n".join(l for l in text.splitlines() if not l.startswith("```")).strip()
        return text or "click('body')"
    except Exception:
        return "click('body')"

# ── Main agent loop ───────────────────────────────────────────────────────────
def run_task():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_ID, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Use synchronous AdVisionEnv based on SyncEnvClient
        # Problem 1 & 2 Fix: Sync pattern verified and handled via specialized client
        env = AdVisionEnv(base_url=EVAL_URL)
        
        with env:
            result = env.reset()
            obs = result.observation

            for step in range(1, MAX_STEPS + 1):
                last_reward = rewards[-1] if rewards else 0.0
                action_str = get_action(client, step, obs, last_reward, history)

                reward = 0.0
                done = False
                err = None

                try:
                    # Parse action string into Typed AdVisionAction
                    act_obj = AdVisionAction.from_string(action_str)
                    
                    # Execute synchronous step
                    result = env.step(act_obj)
                    obs = result.observation
                    reward = float(result.reward) if result.reward is not None else 0.0
                    done = bool(result.done)
                except Exception as step_err:
                    err = str(step_err)
                    done = False

                rewards.append(reward)
                steps_taken = step
                history.append(f"step={step} r={reward:+.2f}")
                log_step(step, action_str, reward, done, err)

                if done:
                    break
                    
        success = (sum(rewards) / len(rewards)) >= 0.5 if rewards else False

    except Exception as outer_err:
        if steps_taken == 0:
            log_step(1, "click('body')", 0.0, True, str(outer_err))
            steps_taken = 1
            rewards.append(0.0)
    finally:
        log_end(success, steps_taken, rewards)

if __name__ == "__main__":
    run_task()
