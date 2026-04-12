import os
import cv2
import numpy as np
from typing import Optional

from openenv.core.env_server import Environment
from advision_env.models import AdVisionAction, AdVisionObservation, AdVisionState, Reward
from advision_env.env.ad_placement_env import AdPlacementEnv

class AdVisionEnvironment(Environment[AdVisionAction, AdVisionObservation, AdVisionState]):
    def __init__(self):
        super().__init__()
        
        # Initialize internal gym environment
        dummy_video = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'input_videos', 'test.mp4')
        if not os.path.exists(dummy_video):
            os.makedirs(os.path.dirname(dummy_video), exist_ok=True)
            # Create a 1 sec dummy video
            out = cv2.VideoWriter(dummy_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 360))
            for i in range(30): out.write(np.zeros((360, 640, 3), dtype=np.uint8))
            out.release()
            
        self.internal_env = AdPlacementEnv(video_path=dummy_video, max_frames=30)
        self.history = []

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> AdVisionObservation:
        obs_raw, info = self.internal_env.reset(seed=seed)
        self.history = []
        self.episode_id = episode_id or f"ep_{int(os.times().elapsed * 1000)}"
        return self._make_obs(obs_raw, 0.0, False, info)

    def step(self, action: AdVisionAction, timeout_s: Optional[float] = None, **kwargs) -> AdVisionObservation:
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
        
        # Keep history for State monitoring
        self.history.append({
            'reward_components': info.get('reward_components', {})
        })
        
        return obs_model

    def _make_obs(self, obs_raw, reward_val, done_val, info) -> AdVisionObservation:
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
        
        # Build the structured observation
        obs = AdVisionObservation(
            detected_surfaces=surf_dicts,
            scene_type=scene_type,
            placement_score=total_score,
            frame_id=info.get('frame_idx', 0),
            frame_features=ff,
            raw_obs=obs_raw.tolist() if hasattr(obs_raw, 'tolist') else list(obs_raw),
            
            # openenv-core Observation base properties
            reward=float(reward_val),
            done=bool(done_val),
            metadata={
                "info": info,
                "reward_components": info.get('reward_components', {})
            }
        )
        return obs

    @property
    def state(self) -> AdVisionState:
        return AdVisionState(
            episode_id=getattr(self, 'episode_id', "unknown"),
            step_count=self.internal_env._frame_idx if hasattr(self.internal_env, '_frame_idx') else 0,
            max_frames=self.internal_env.max_frames,
            video_path=self.internal_env.video_path,
            detected_surfaces_count=len(getattr(self.internal_env, "_surfaces", [])),
            history_len=len(self.history),
            done=all(h.get('done', False) for h in self.history[-1:]) if self.history else False
        )
