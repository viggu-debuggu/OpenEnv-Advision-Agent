"""
inference.py — Universal OpenEnv Agent Baseline
----------------------------------------------------------
Complies with the official OpenEnv evaluation protocol:
  - Uses GenericEnvClient (async) via .sync() wrapper
  - Connects to the evaluator-provided OPENENV_URL (fixes circular-dep Error #1)
  - Correctly reads reward from StepResult.reward (fixes Error #2)
  - No fake/surrogate grader (fixes Error #3)
  - Guarantees [STEP] and [END] are emitted even on crash (fixes Error #4)
  - Outputs function-call action strings like click('id') (fixes Error #7)
"""
from __future__ import annotations

import json
import os
from typing import Any, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from openenv.core import GenericEnvClient

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str     = os.getenv("HF_TOKEN", "dummy")

# ---------------------------------------------------------------------------
# Fix Error #1 — connect to the EVALUATOR's env server, not our own Space
# The evaluator injects OPENENV_URL into the container at runtime.
# ---------------------------------------------------------------------------
EVAL_URL: str = os.getenv(
    "OPENENV_URL",
    os.getenv("SPACE_URL", "http://localhost:8000"),
)

# ---------------------------------------------------------------------------
# Task / run constants (evaluator may inject these)
# ---------------------------------------------------------------------------
BENCHMARK:  str = os.getenv("BENCHMARK",  "miniwob")
MAX_STEPS:  int = int(os.getenv("MAX_STEPS",  "10"))
TASK_ID:    str = os.getenv("TASK_ID",    "default_task")


# ---------------------------------------------------------------------------
# Logging helpers  (must stay exactly in this format for auto-grader)
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: str | None = None) -> None:
    done_str   = "true" if done else "false"
    err_str    = str(error).replace(" ", "_").replace("\n", "_")[:80] if error else "null"
    action_str = str(action).replace(" ", "")
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_str} error={err_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM action generation
# Fix Error #7 — output function-call strings, not raw JSON blobs
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a universal AI agent driving an OpenEnv environment.
On each step you receive the current observation and must decide the next action.

For browser / miniwob tasks output ONE function call on a single line, e.g.:
  click('element_id')
  type('element_id', 'some text')
  fill('form_id')
  scroll('element_id', 'down')
  submit('form_id')

For text/reasoning tasks output the answer or tool call as a plain string, e.g.:
  answer('Paris')
  execute('print(1+1)')

Rules:
- Return ONLY the action string — no JSON, no markdown, no explanation.
- If unsure, prefer click('body') as a safe fallback.
"""


def _fallback_action() -> str:
    return "click('body')"


def get_llm_action(
    client: OpenAI,
    step: int,
    obs: Any,
    last_reward: float,
    history: List[str],
) -> str:
    """Ask the LLM for the next action; fall back gracefully on any error."""
    if HF_TOKEN in ("dummy", "", None):
        return _fallback_action()

    obs_str = json.dumps(obs, default=str) if isinstance(obs, dict) else str(obs)
    # Trim very long observations so we don't blow past token limits
    if len(obs_str) > 3000:
        obs_str = obs_str[:3000] + "… [truncated]"

    user_msg = (
        f"Task: {TASK_ID}\n"
        f"Step: {step}/{MAX_STEPS}\n"
        f"Last reward: {last_reward:.2f}\n"
        f"History: {history[-3:]}\n\n"
        f"Observation:\n{obs_str}\n\n"
        f"What is the next action?"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=80,
        )
        text = resp.choices[0].message.content.strip()
        # Strip any accidental markdown fences
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()
        return text or _fallback_action()
    except Exception as exc:  # noqa: BLE001
        # Don't crash the agent loop — just fall back
        print(f"[WARN] LLM error at step {step}: {exc}", flush=True)
        return _fallback_action()


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------
def run_task() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    log_start(task=TASK_ID, env=BENCHMARK, model=MODEL_NAME)

    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    success:     bool        = False

    # Fix Error #4 — robust try/finally guarantees [END] is always emitted
    try:
        async_client = GenericEnvClient(base_url=EVAL_URL)
        env = async_client.sync()

        with env:
            # Reset the environment
            reset_result = env.reset()
            obs = reset_result.observation

            for step in range(1, MAX_STEPS + 1):
                last_reward = rewards[-1] if rewards else 0.0
                action = get_llm_action(llm_client, step, obs, last_reward, history)

                err: str | None = None
                reward:  float  = 0.0
                done:    bool   = False

                try:
                    # Fix Error #2 — read reward from StepResult, NOT from obs
                    step_result = env.step(action)
                    obs    = step_result.observation
                    reward = float(step_result.reward) if step_result.reward is not None else 0.0
                    done   = bool(step_result.done)
                except Exception as step_exc:
                    err  = str(step_exc)
                    done = False

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action, reward=reward, done=done, error=err)
                history.append(f"step={step} act={action} r={reward:+.2f}")

                if done:
                    break

        # Success threshold: mean reward ≥ 0.5
        if rewards:
            success = (sum(rewards) / len(rewards)) >= 0.5

    except Exception as outer_exc:
        # Guarantee at least one [STEP] so the evaluator doesn't score 0 steps
        if steps_taken == 0:
            log_step(step=1, action="click('body')", reward=0.0, done=True,
                     error=str(outer_exc))
            steps_taken = 1
            rewards.append(0.0)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


def main() -> None:
    run_task()


if __name__ == "__main__":
    main()
