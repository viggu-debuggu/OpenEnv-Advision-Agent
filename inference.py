"""
inference.py — OpenEnv AdVision Agent (Safe HTTP Version)
Fully compliant. Zero dependency on internal packages.
"""
import os
import json
from typing import List, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

EVAL_URL  = os.getenv("OPENENV_URL", "http://localhost:8000")
BENCHMARK = os.getenv("BENCHMARK",   "advision_env")
TASK_ID   = os.getenv("TASK_ID",     "task1")
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    err = str(error).replace(" ","_").replace("\n","_")[:80] if error else "null"
    act = str(action).replace(" ","")
    print(f"[STEP] step={step} action={act} reward={reward:.2f} "
          f"done={'true' if done else 'false'} error={err}", flush=True)

def log_end(success, steps, rewards):
    print(f"[END] success={'true' if success else 'false'} "
          f"steps={steps} rewards={','.join(f'{x:.2f}' for x in rewards)}",
          flush=True)

SYSTEM_PROMPT = """You are an AI agent for an ad placement environment.
Output ONE action per line. Examples:
  place(x=0.0,y=0.0,scale=1.0,alpha=0.97)
  click('surface_0')
Return ONLY the action string. No explanation."""

def get_action(client, step, obs, last_reward, history):
    obs_str = json.dumps(obs, default=str) if isinstance(obs, dict) else str(obs)
    if len(obs_str) > 2000:
        obs_str = obs_str[:2000] + "...[truncated]"
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Step:{step}/{MAX_STEPS} LastReward:{last_reward:.2f}\n"
                    f"Obs:{obs_str}\nAction?"}
            ],
            temperature=0.2, max_tokens=80,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = "\n".join(
                l for l in text.splitlines() if not l.startswith("```")).strip()
        return text or "place(x=0.0,y=0.0,scale=1.0,alpha=0.97)"
    except Exception:
        return "place(x=0.0,y=0.0,scale=1.0,alpha=0.97)"

def main():
    client      = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    rewards     = []
    history     = []
    steps_taken = 0
    success     = False

    log_start(task=TASK_ID, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Wake server
        try:
            requests.get(f"{EVAL_URL}/health", timeout=15)
        except Exception:
            pass

        # Reset
        res = requests.post(f"{EVAL_URL}/reset", timeout=60)
        res.raise_for_status()
        data = res.json()
        obs  = data.get("observation", data)

        for step in range(1, MAX_STEPS + 1):
            last_reward = rewards[-1] if rewards else 0.0
            action_str  = get_action(client, step, obs, last_reward, history)
            reward = 0.0; done = False; err = None

            try:
                r = requests.post(
                    f"{EVAL_URL}/step",
                    json={"action": action_str},
                    timeout=60)
                r.raise_for_status()
                result = r.json()
                reward = float(result.get("reward", 0.0))
                done   = bool(result.get("done", False))
                obs    = result.get("observation", obs)
            except Exception as e:
                err = str(e); done = False

            rewards.append(reward)
            steps_taken = step
            history.append(f"step={step} r={reward:+.2f}")
            log_step(step, action_str, reward, done, err)
            if done:
                break

        success = (sum(rewards)/len(rewards)) >= 0.5 if rewards else False

    except Exception as e:
        if steps_taken == 0:
            log_step(1, "place(x=0.0,y=0.0,scale=1.0,alpha=0.97)",
                     0.0, True, str(e))
            steps_taken = 1; rewards.append(0.0)
    finally:
        log_end(success, steps_taken, rewards)

if __name__ == "__main__":
    main()
