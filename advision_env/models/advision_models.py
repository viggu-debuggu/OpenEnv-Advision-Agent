from typing import List, Dict, Any
from pydantic import Field, BaseModel
from openenv.core.env_server.types import Action, Observation, State

class AdVisionObservation(Observation):
    """
    Unified AdVision Observation model.
    Inherits reward, done, and metadata from OpenEnv Observation.
    WE MOVE CRITICAL METADATA TO TOP-LEVEL FIELDS TO PREVENT STRIPPING BY SERVER.
    """
    detected_surfaces: List[Dict[str, Any]] = Field(default_factory=list)
    scene_type: str = "unknown"
    placement_score: float = 0.0
    frame_id: int = 0
    frame_features: Dict[str, float] = Field(default_factory=dict)
    raw_obs: List[float] = Field(default_factory=list)

    # Critical info moved to top-level to survive serialization
    info: Dict[str, Any] = Field(default_factory=dict)
    reward_components: Dict[str, float] = Field(default_factory=dict)

class AdVisionAction(Action):
    """
    Unified AdVision Action model.
    """
    x_position: float = Field(0.0, ge=-0.5, le=0.5)
    y_position: float = Field(0.0, ge=-0.5, le=0.5)
    scale: float = Field(1.0, ge=0.5, le=1.5)
    rotation: float = Field(0.0, ge=-30.0, le=30.0)
    tilt: float = Field(0.0, ge=0.0, le=1.0)
    ad_selection: float = Field(0.0, ge=0.0, le=1.0)
    alpha: float = Field(0.97, ge=0.0, le=1.0)

class AdVisionState(State):
    """
    Unified AdVision State model.
    Inherits episode_id and step_count from OpenEnv State.
    """
    max_frames: int = 0
    video_path: str = ""
    detected_surfaces_count: int = 0
    history_len: int = 0
    done: bool = False

class Reward(BaseModel):
    """
    Detailed reward components model.
    """
    placement_reward: float = 0.0
    realism_reward: float = 0.0
    temporal_stability_reward: float = 0.0
    occlusion_reward: float = 0.0
    penalty_for_flickering: float = 0.0
