from __future__ import annotations
from typing import List, Dict, Any

from pydantic import BaseModel, Field

class SurfaceInfo(BaseModel):
    centroid_x: float = Field(ge=0., le=1.)
    centroid_y: float = Field(ge=0., le=1.)
    area: float = Field(ge=0., le=1.)
    depth: float = Field(default=0.5, ge=0., le=1.)
    confidence: float = Field(default=1.0, ge=0., le=1.)
    bbox: List[int] = Field(default_factory=list)
    corners: List[List[float]] = Field(default_factory=list)

class DetectedObject(BaseModel):
    label: str
    confidence: float = Field(ge=0., le=1.)
    bbox: List[int] = Field(default_factory=list)

class FrameFeatures(BaseModel):
    brightness: float = 0.5
    edge_density: float = 0.0
    sharpness: float = 0.0
    mean_bgr: List[float] = Field(default_factory=lambda: [0.5, 0.5, 0.5])

class Observation(BaseModel):
    image: str = "" # Base64 encoded frame
    frame_idx: int = 0
    n_surfaces: int = 0
    surfaces: List[SurfaceInfo] = Field(default_factory=list)
    persons: List[DetectedObject] = Field(default_factory=list)
    frame_features: FrameFeatures = Field(default_factory=FrameFeatures)
    raw_vector: List[float] = Field(default_factory=list)

    @staticmethod
    def from_vector(vec: List[float], frame_idx: int = 0, n_persons: int = 0) -> Observation:
        """Parses 28-dim environment vector into Observation model."""
        if len(vec) < 28:
            vec = vec + [0.0] * (28 - len(vec))
        
        feat = FrameFeatures(
            mean_bgr=vec[0:3],
            brightness=vec[3],
            edge_density=vec[4],
            sharpness=vec[5]
        )
        
        surfaces = []
        for i in range(5):
            base = 6 + i * 4
            if vec[base+2] > 0: # Check area
                surfaces.append(SurfaceInfo(
                    centroid_x=vec[base],
                    centroid_y=vec[base+1],
                    area=vec[base+2],
                    depth=vec[base+3]
                ))
        
        return Observation(
            frame_idx=frame_idx,
            n_surfaces=len(surfaces),
            surfaces=surfaces,
            frame_features=feat,
            raw_vector=vec
        )

class Action(BaseModel):
    """
    7-dimensional action space:
    - surface_idx: 0.0 to 1.0 (float mapping to categorical index)
    - x_offset: -0.2 to 0.2
    - y_offset: -0.2 to 0.2
    - scale: 0.5 to 1.5
    - ad_idx: 0.0 to 1.0
    - rotation_deg: -30 to 30
    - perspective_tilt: 0.0 to 1.0
    """
    surface_idx: float = Field(default=0.5, ge=0.0, le=1.0)
    x_offset: float = Field(default=0.0, ge=-0.2, le=0.2)
    y_offset: float = Field(default=0.0, ge=-0.2, le=0.2)
    scale: float = Field(default=1.0, ge=0.5, le=1.5)
    ad_idx: float = Field(default=0.0, ge=0.0, le=1.0)
    rotation_deg: float = Field(default=0.0, ge=-30.0, le=30.0)
    perspective_tilt: float = Field(default=0.0, ge=0.0, le=1.0)

    def to_vector(self) -> List[float]:
        return [
            self.surface_idx,
            self.x_offset,
            self.y_offset,
            self.scale,
            self.ad_idx,
            self.rotation_deg,
            self.perspective_tilt
        ]

class Reward(BaseModel):
    total: float = Field(ge=0., le=1.)
    realism: float = Field(ge=0., le=1.)
    alignment: float = Field(ge=0., le=1.)
    lighting: float = Field(ge=0., le=1.)
    occlusion: float = Field(ge=0., le=1.)
    visibility: float = Field(ge=0., le=1.)
    temporal: float = Field(ge=0., le=1.)

class EnvState(BaseModel):
    frame_idx: int
    n_surfaces: int
    n_ads: int
    n_persons: int
    last_reward: Dict[str, float] = Field(default_factory=dict)
    surfaces: List[Dict[str, Any]] = Field(default_factory=list)

print('[v] models.py v5 - aligned with AdPlacementEnv 7-dim space')
