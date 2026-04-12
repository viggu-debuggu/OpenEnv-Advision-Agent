"""
advision.openenv_wrapper
------------------------
Glue layer to provide AdVisionEnv and task graders for the OpenEnv baseline.
"""
from typing import List, Dict, Any
from .client import AdVisionEnv as AdVisionEnv
from .openenv.tasks import Task1_BasicPlacement, Task2_RealisticBlend, Task3_TemporalConsistency
from .models import Reward

def _get_tr(info: Dict[str, Any]) -> Reward:
    """Helper to get a Reward object from info dict."""
    tr = info.get("typed_reward")
    if tr and hasattr(tr, 'placement_reward'):
        return tr

    # Fallback: Reconstruct from reward_components
    rc = info.get("reward_components", {})
    return Reward(
        placement_reward=float(rc.get("alignment", rc.get("placement", 0.0))),
        realism_reward=float(rc.get("lighting", rc.get("realism", 0.0))),
        temporal_stability_reward=float(rc.get("temporal", 0.0)),
        occlusion_reward=float(rc.get("occlusion", 0.0)),
        penalty_for_flickering=0.0
    )

def grade_task1(history: List[Dict[str, Any]]) -> float:
    """Grade Task 1: Basic Surface Placement."""
    task = Task1_BasicPlacement()
    for entry in history:
        info = entry.get("info", {})
        tr = _get_tr(info)
        task.update(tr.placement_reward, info)
    return float(task.grade().score)

def grade_task2(history: List[Dict[str, Any]]) -> float:
    """Grade Task 2: Realistic Blend Placement."""
    task = Task2_RealisticBlend()
    for entry in history:
        info = entry.get("info", {})
        tr = _get_tr(info)
        # Task 2 grader expects certain keys in info['reward_components']
        rc = {
            "alignment": tr.placement_reward,
            "lighting": tr.realism_reward,
            "temporal": tr.temporal_stability_reward
        }
        task.update(tr.placement_reward, {"reward_components": rc})
    return float(task.grade().score)

def grade_task3(history: List[Dict[str, Any]]) -> float:
    """Grade Task 3: Photorealistic Temporal Tracking."""
    task = Task3_TemporalConsistency()
    for entry in history:
        info = entry.get("info", {})
        tr = _get_tr(info)
        rc = {
            "temporal": tr.temporal_stability_reward,
            "occlusion": tr.occlusion_reward
        }
        task.update(tr.temporal_stability_reward, {"reward_components": rc})
    return float(task.grade().score)
