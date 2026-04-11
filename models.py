from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    # Fallbacks for scripts that don't have openenv-core
    class Action(BaseModel):
        pass
    class Observation(BaseModel):
        done: bool = False
        reward: Optional[float] = None
        metadata: dict = Field(default_factory=dict)

class AdVisionObservation(Observation):
    detected_surfaces: List[Dict[str, Any]] = Field(default_factory=list)
    scene_type: str = "unknown"
    placement_score: float = 0.0
    frame_features: Dict[str, float] = Field(default_factory=dict)
    raw_obs: List[float] = Field(default_factory=list)

class AdVisionAction(Action):
    x_position: float = Field(0.0)
    y_position: float = Field(0.0)
    scale: float = Field(1.0)
    rotation: float = Field(0.0)
    tilt: float = Field(0.0)
    ad_selection: float = Field(0.0)
    alpha: float = Field(0.97)

class Reward(BaseModel):
    placement_reward: float = 0.0
    realism_reward: float = 0.0
    temporal_stability_reward: float = 0.0
    occlusion_reward: float = 0.0
    penalty_for_flickering: float = 0.0
