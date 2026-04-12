"""baseline_agent.py — Heuristic by default; Claude API if ANTHROPIC_API_KEY set.
   Updated for 7-dim action space: [surface_idx, x_offset, y_offset, scale,
                                     ad_idx, rotation_deg, perspective_tilt]
"""
from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from advision_env.env.ad_placement_env import AdPlacementEnv  # noqa: E402
from advision_env.openenv.tasks import TASKS  # noqa: E402
from advision_env.openenv.models import Observation, Action  # noqa: E402


try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

SYSTEM_PROMPT = """You are an expert ad placement agent.
Output ONLY valid JSON with ALL 7 keys:
{"surface_idx":0.8, "x_offset":0.0, "y_offset":0.0, "scale":1.0, "ad_idx":0.0, "rotation_deg":0.0, "perspective_tilt":0.0}
Rules:
- Prefer surfaces with depth>=0.5 and area>=0.15
- Keep offsets near 0 for stability (reduces jitter penalty)
- rotation_deg: use scene edge angle hint × 60 - 30 (range -30..30)
- perspective_tilt: 0 for frontal walls, 0.3-0.6 for floors/angled surfaces
"""


def _heuristic(obs):
    """
    7-dim heuristic agent.
    - Picks best surface by depth*0.6 + area*0.4
    - Infers rotation from scene edge angle (obs[26])
    - Infers perspective tilt from surface y-position (low surfaces = angled)
    """
    if not obs.surfaces or all(s.area == 0 for s in obs.surfaces):
        return Action(surface_idx=0.5, x_offset=0., y_offset=0., scale=1., ad_idx=0.,
                      rotation_deg=0., perspective_tilt=0.)

    scores = [s.depth*0.6+s.area*0.4 for s in obs.surfaces]
    best = int(np.argmax(scores))
    n_surfs = len(obs.surfaces)
    si_val = float(best)/max(1, n_surfs-1) if n_surfs > 1 else 0.5

    # Rotation hint from obs[26]: map 0..1 → -15..15 degrees (gentle rotation)
    edge_angle_hint = float(obs.raw_vector[26]) if len(obs.raw_vector) > 26 else 0.5
    rot = float(np.clip((edge_angle_hint - 0.5) * 30., -15., 15.))

    # Perspective tilt: surfaces in lower half of frame likely need tilt
    surf = obs.surfaces[best]
    tilt = float(np.clip((surf.centroid_y - 0.5) * 0.8, 0., 0.6))

    return Action(surface_idx=float(np.clip(si_val, 0, 1)),
                  x_offset=0., y_offset=0., scale=1., ad_idx=0.,
                  rotation_deg=rot, perspective_tilt=tilt)


def _claude_action(obs, client):
    surfs = "\n".join(f"  surf{i}: cx={s.centroid_x:.2f} cy={s.centroid_y:.2f} "
                      f"area={s.area:.2f} depth={s.depth:.2f}"
                      for i, s in enumerate(obs.surfaces))
    edge_hint = float(obs.raw_vector[26]) if len(obs.raw_vector) > 26 else 0.5
    prompt = (f"Frame {obs.frame_idx}: brightness={obs.frame_features.brightness:.2f} "
              f"edges={obs.frame_features.edge_density:.2f} edge_angle_hint={edge_hint:.2f}\n"
              f"Surfaces ({len(obs.surfaces)}):\n{surfs}\nOutput 7-key action JSON:")
    try:
        msg = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=140,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.startswith("```")).strip()
        data = json.loads(text)
        bounds = {"surface_idx": (0, 1), "x_offset": (-0.2, 0.2), "y_offset": (-0.2, 0.2),
                  "scale": (0.5, 1.5), "ad_idx": (0, 1), "rotation_deg": (-30, 30), "perspective_tilt": (0, 1)}
        defaults = {"surface_idx": 0.5, "x_offset": 0., "y_offset": 0., "scale": 1.,
                    "ad_idx": 0., "rotation_deg": 0., "perspective_tilt": 0.}
        for k, (lo, hi) in bounds.items():
            data[k] = float(np.clip(data.get(k, defaults[k]), lo, hi))
        return Action(**data)
    except Exception:
        return _heuristic(obs)


def run_task(task_id, n_frames=30, client=None, seed=42, video_path=None, ad_paths=None):
    env = AdPlacementEnv(
        max_frames=n_frames,
        seed=seed,
        video_path=video_path,
        ad_paths=ad_paths or []
    )
    task = TASKS[task_id]()
    task.reset()
    obs_vec, info = env.reset(seed=seed)
    obs = Observation.from_vector(obs_vec.tolist(), 0, info.get("n_persons", 0))
    rewards = []
    t0 = time.time()
    for step_idx in range(n_frames):
        action = _claude_action(obs, client) if client else _heuristic(obs)
        obs_vec, reward, term, trunc, info = env.step(np.array(action.to_vector(), np.float32))
        obs = Observation.from_vector(obs_vec.tolist(), step_idx + 1, info.get("n_persons", 0))
        task.update(reward, info)
        rewards.append(float(reward))
        if term or trunc:
            break
    result = task.grade()
    env.close()
    return {"task_id": task_id, "difficulty": TASKS[task_id].DIFFICULTY,
            "score": result.score, "passed": result.passed, "details": result.details,
            "mean_reward": round(float(np.mean(rewards)), 4),
            "min_reward": round(float(np.min(rewards)), 4),
            "max_reward": round(float(np.max(rewards)), 4),
            "n_frames": len(rewards), "elapsed_s": round(time.time()-t0, 2),
            "agent": "claude" if client else "heuristic", "seed": seed}


def main(seed=42, video_path=None, ad_paths=None):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    client = None
    if HAS_ANTHROPIC and api_key:
        client = anthropic.Anthropic(api_key=api_key)
        agent_name = "Claude Haiku Agent"
    else:
        agent_name = "Heuristic Agent (7-dim)"
    task_configs = [
        ("task1_basic_placement", 15, "Easy"),
        ("task2_realistic_blend", 25, "Medium"),
        ("task3_temporal_consistency", 35, "Hard")
    ]
    print("\n"+"="*60)
    print(f"ADVISION AI BASELINE | Agent: {agent_name} | Seed: {seed}")
    print("="*60)
    results = {}
    for tid, nf, diff in task_configs:
        r = run_task(tid, nf, client, seed, video_path, ad_paths)
        results[tid] = r
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{diff}] {tid}: score={r['score']:.4f} {status} | mean_r={r['mean_reward']:.4f}")
    scores = [results[t]["score"] for t, _, _ in task_configs]
    print("="*60)
    print(f"  Average: {np.mean(scores):.4f}")
    print("="*60)
    summary = {
        "agent": agent_name,
        "seed": seed,
        "tasks": results,
        "avg_score": round(float(np.mean(scores)), 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[v] Saved -> logs/baseline_results.json")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)

print('[v] baseline_agent.py v5 - 7-dim heuristic + Claude prompt updated')
