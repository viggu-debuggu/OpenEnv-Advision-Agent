"""
advision.openenv_wrapper
------------------------
Glue layer to provide AdVisionEnv and task graders for the OpenEnv baseline.
"""
from typing import List, Dict, Any
from .client import AdVisionEnv
from .openenv.tasks import Task1_BasicPlacement, Task2_RealisticBlend, Task3_TemporalConsistency

def grade_task1(history: List[Dict[str, Any]]) -> float:
    """Grade Task 1: Basic Surface Placement."""
    task = Task1_BasicPlacement()
    for entry in history:
        tr = entry["info"].get("typed_reward")
        if not tr: continue
        # Use placement_reward as the primary signal for this task
        reward = getattr(tr, 'placement_reward', 0.0)
        task.update(reward, entry["info"])
    return float(task.grade().score)

def grade_task2(history: List[Dict[str, Any]]) -> float:
    """Grade Task 2: Realistic Blend Placement."""
    task = Task2_RealisticBlend()
    for entry in history:
        tr = entry["info"].get("typed_reward")
        if not tr: continue
        # Task 2 expects alignment and lighting in info['reward_components']
        info = {
            "reward_components": {
                "alignment": getattr(tr, 'placement_reward', 0.0),
                "lighting": getattr(tr, 'realism_reward', 0.0),
                "temporal": getattr(tr, 'temporal_stability_reward', 0.0)
            }
        }
        reward = (getattr(tr, 'placement_reward', 0.0) + getattr(tr, 'realism_reward', 0.0)) / 2.0
        task.update(reward, info)
    return float(task.grade().score)

def grade_task3(history: List[Dict[str, Any]]) -> float:
    """Grade Task 3: Photorealistic Temporal Tracking."""
    task = Task3_TemporalConsistency()
    for entry in history:
        tr = entry["info"].get("typed_reward")
        if not tr: continue
        info = {
            "reward_components": {
                "temporal": getattr(tr, 'temporal_stability_reward', 0.0),
                "occlusion": getattr(tr, 'occlusion_reward', 0.0)
            }
        }
        reward = getattr(tr, 'temporal_stability_reward', 0.0)
        task.update(reward, info)
    return float(task.grade().score)
