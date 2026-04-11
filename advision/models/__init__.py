from pydantic import BaseModel, Field
from typing import Optional, List, Any

class AdVisionState(BaseModel):
    frame_id: int = 0
    done: bool = False
    metadata: dict = {}

class AdVisionAction(BaseModel):
    x_position:   float = Field(0.0,  ge=-0.5, le=0.5)
    y_position:   float = Field(0.0,  ge=-0.5, le=0.5)
    scale:        float = Field(1.0,  ge=0.5,  le=1.5)
    rotation:     float = Field(0.0,  ge=-30,  le=30)
    tilt:         float = Field(0.0,  ge=0.0,  le=1.0)
    alpha:        float = Field(0.97, ge=0.0,  le=1.0)
    ad_selection: float = Field(0.0,  ge=0.0,  le=1.0)
    raw:          Optional[str] = None  # fallback for unparsed strings

    @classmethod
    def from_string(cls, s: str) -> "AdVisionAction":
        """Parse LLM output like place_ad(x=0.1, y=0.2) into AdVisionAction."""
        import re, json
        # Try JSON first
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                return cls(**data)
        except Exception:
            pass
            
        # Try key=value pairs or function call format
        pairs = dict(re.findall(r'(\w+)\s*=\s*([\d.\-]+)', s))
        if pairs:
            mapped = {
                "x_position":   float(pairs.get("x_position", pairs.get("x", 0.0))),
                "y_position":   float(pairs.get("y_position", pairs.get("y", 0.0))),
                "scale":        float(pairs.get("scale", 1.0)),
                "rotation":     float(pairs.get("rotation", 0.0)),
                "tilt":         float(pairs.get("tilt", 0.0)),
                "alpha":        float(pairs.get("alpha", 0.97)),
                "ad_selection": float(pairs.get("ad_selection", 0.0)),
            }
            return cls(**mapped)
            
        # Last resort: store as raw if it's a string
        return cls(raw=s)

class AdVisionObservation(BaseModel):
    scene_type:        str   = "unknown"
    placement_score:   float = 0.0
    detected_surfaces: List[Any] = []
    frame_id:          int   = 0
    metadata:          dict  = {}

class Reward(BaseModel):
    placement_reward: float = 0.0
    realism_reward: float = 0.0
    temporal_stability_reward: float = 0.0
    occlusion_reward: float = 0.0
    penalty_for_flickering: float = 0.0
