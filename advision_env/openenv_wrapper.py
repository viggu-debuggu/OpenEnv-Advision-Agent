import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field

# OpenEnv base classes
try:
    from openenv import OpenEnv
except ImportError:
    class OpenEnv:
        pass

# Import the old internal gym env for reuse
from advision_env.env.ad_placement_env import AdPlacementEnv


# Import standard models from root
try:
    from models import Action, Observation, Reward
except ImportError:
    # Handle if root is not in sys.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models import Action, Observation, Reward


# --- OpenEnv Implementation ---

class AdVisionEnv(OpenEnv):
    def __init__(self):
        super().__init__()
        # Initialize internal gym environment
        dummy_video = os.path.join(os.path.dirname(__file__), '..', 'data', 'input_videos', 'test.mp4')
        if not os.path.exists(dummy_video):
            os.makedirs(os.path.dirname(dummy_video), exist_ok=True)
            # Create a 1 sec dummy video
            out = cv2.VideoWriter(dummy_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 360))
            for i in range(30): out.write(np.zeros((360, 640, 3), dtype=np.uint8))
            out.release()
            
        self.internal_env = AdPlacementEnv(video_path=dummy_video, max_frames=30)
        self.history = []

    def reset(self) -> Observation:
        obs_raw, info = self.internal_env.reset()
        self.history = []
        self.episode_id = f"ep_{int(os.times().elapsed * 1000)}"
        return self._make_obs(obs_raw, 0.0, False, info)

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        # Map 7-dim OpenEnv Action to 7-dim Internal Gym Action
        # Internal: [surface_idx, x_offset, y_offset, scale, ad_idx, rotation_deg, perspective_tilt]
        # Pass alpha through as an override attribute so PlacementEngine uses it
        self.internal_env._override_alpha = float(action.alpha)
        internal_action = np.array([
            0.0,                        # surface_idx locked to 0
            float(action.x_position),
            float(action.y_position),
            float(action.scale),
            float(action.ad_selection),
            float(action.rotation),
            float(action.tilt)
        ], dtype=np.float32)

        obs_raw, reward_float, term, trunc, info = self.internal_env.step(internal_action)
        done = term or trunc
        
        obs_model = self._make_obs(obs_raw, reward_float, done, info)
        rew_model = self._make_reward(info)
        
        self.history.append({
            'reward_components': info.get('reward_components', {})
        })
        
        info['typed_reward'] = rew_model
        
        return obs_model, reward_float, done, info

    def state(self) -> dict:
        return {
            "max_frames": self.internal_env.max_frames,
            "current_frame": getattr(self.internal_env, "_frame_idx", 0),
            "video_path": self.internal_env.video_path,
            "detected_surfaces_count": len(getattr(self.internal_env, "_surfaces", [])),
            "history_len": len(self.history)
        }



    def _make_obs(self, obs_raw, reward_val, done_val, info) -> Observation:
        scene = getattr(self.internal_env, '_scene', None)
        scene_type = scene.scene_type if scene else "unknown"
        
        # Surfaces
        surfaces = getattr(self.internal_env, '_surfaces', [])
        surf_dicts = [s.to_dict() for s in surfaces]
        
        total_score = info.get('reward_components', {}).get('total', 0.0)
        
        ff = {
            'mean_b': float(obs_raw[0]),
            'mean_g': float(obs_raw[1]),
            'mean_r': float(obs_raw[2]),
            'brightness': float(obs_raw[3]),
            'edge_density': float(obs_raw[4]),
            'sharpness': float(obs_raw[5]),
        }
        
        return Observation(
            detected_surfaces=surf_dicts,
            scene_type=scene_type,
            placement_score=total_score,
            frame_features=ff,
            reward=float(reward_val),
            done=bool(done_val),
            raw_obs=obs_raw.tolist()
        )

    def _make_reward(self, info) -> Reward:
        rc = info.get('reward_components', {})
        return Reward(
            placement_reward=rc.get('alignment', 0.0),
            realism_reward=rc.get('realism', 0.0),
            temporal_stability_reward=rc.get('temporal', 0.0),
            occlusion_reward=rc.get('occlusion', 0.0),
            penalty_for_flickering=0.0 # Handled in temporal internally
        )


# --- Graders for Tasks ---

def _clamp(v: float) -> float:
    # Round to 1 decimal, then stay strictly between 0 and 1 (0.1 to 0.9)
    val = round(float(v), 1)
    return float(np.clip(val, 0.1, 0.9))

def grade_task1(episode_history: List[dict]) -> float:
    # Easy: Place ad on detected surface (Placement reward > 0.5)
    if not episode_history: return _clamp(0.0)
    placements = [step.get('info', {}).get('typed_reward', {}).placement_reward for step in episode_history]
    if not placements: return _clamp(0.0)
    ratio = sum(1 for r in placements if r > 0.5) / len(placements)
    return _clamp(ratio)

def grade_task2(episode_history: List[dict]) -> float:
    # Medium: Place ad with correct scale and perspective (Realism > 0.6)
    if not episode_history: return _clamp(0.0)
    realisms = [step.get('info', {}).get('typed_reward', {}).realism_reward for step in episode_history]
    if not realisms: return _clamp(0.0)
    ratio = sum(1 for r in realisms if r > 0.6) / len(realisms)
    return _clamp(ratio)

def grade_task3(episode_history: List[dict]) -> float:
    # Hard: Photorealistic placement with tracking (Temporal Stability > 0.7)
    if not episode_history: return _clamp(0.0)
    temporals = [step.get('info', {}).get('typed_reward', {}).temporal_stability_reward for step in episode_history]
    if not temporals: return _clamp(0.0)
    ratio = sum(1 for r in temporals if r > 0.7) / len(temporals)
    return _clamp(ratio)

