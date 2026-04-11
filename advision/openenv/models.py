from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from pydantic import BaseModel, Field
import warnings

class SurfaceInfo(BaseModel):
    centroid_x: float=Field(ge=0.,le=1.); centroid_y: float=Field(ge=0.,le=1.)
    area: float=Field(ge=0.,le=1.); depth: float=Field(ge=0.,le=1.)
    quality: float=Field(ge=0.,le=1.,default=0.5)

class FrameFeatures(BaseModel):
    mean_b: float=Field(ge=0.,le=1.); mean_g: float=Field(ge=0.,le=1.)
    mean_r: float=Field(ge=0.,le=1.); brightness: float=Field(ge=0.,le=1.)
    edge_density: float=Field(ge=0.,le=1.); sharpness: float=Field(ge=0.,le=1.)

class Observation(BaseModel):
    frame_features: FrameFeatures
    surfaces: List[SurfaceInfo]=Field(default_factory=list)
    frame_idx: int=Field(ge=0); n_persons: int=Field(ge=0,default=0)
    raw_vector: List[float]

    @classmethod
    def from_vector(cls, vec, frame_idx, n_persons=0):
        vec=list(vec)
        # Support both 26-dim (old) and 28-dim (new) observation vectors
        if len(vec)<26:
            vec=vec+[0.]*(28-len(vec))
        elif len(vec)==26:
            vec=vec+[0.,0.]   # pad old obs to 28
        elif len(vec)>28:
            vec=vec[:28]
        ff=FrameFeatures(mean_b=float(np.clip(vec[0],0,1)),mean_g=float(np.clip(vec[1],0,1)),
                         mean_r=float(np.clip(vec[2],0,1)),brightness=float(np.clip(vec[3],0,1)),
                         edge_density=float(np.clip(vec[4],0,1)),sharpness=float(np.clip(vec[5],0,1)))
        surfs=[SurfaceInfo(centroid_x=float(np.clip(vec[6+i*4],0,1)),
                            centroid_y=float(np.clip(vec[7+i*4],0,1)),
                            area=float(np.clip(vec[8+i*4],0,1)),
                            depth=float(np.clip(vec[9+i*4],0,1))) for i in range(5)]
        return cls(frame_features=ff,surfaces=surfs,frame_idx=frame_idx,
                   n_persons=n_persons,raw_vector=vec)

class Action(BaseModel):
    """7-dim action matching the upgraded action space."""
    surface_idx:       float = Field(ge=0.,   le=1.)
    x_offset:          float = Field(ge=-0.2, le=0.2)
    y_offset:          float = Field(ge=-0.2, le=0.2)
    scale:             float = Field(ge=0.5,  le=1.5)
    ad_idx:            float = Field(ge=0.,   le=1.)
    rotation_deg:      float = Field(ge=-30., le=30.,  default=0.)   # NEW
    perspective_tilt:  float = Field(ge=0.,   le=1.,   default=0.)   # NEW

    def to_vector(self):
        return [self.surface_idx, self.x_offset, self.y_offset,
                self.scale, self.ad_idx, self.rotation_deg, self.perspective_tilt]

class Reward(BaseModel):
    total: float=Field(ge=0.,le=1.); realism: float=Field(ge=0.,le=1.)
    alignment: float=Field(ge=0.,le=1.); lighting: float=Field(ge=0.,le=1.)
    occlusion: float=Field(ge=0.,le=1.); visibility: float=Field(ge=0.,le=1.)
    temporal: float=Field(ge=0.,le=1.)

class EnvState(BaseModel):
    frame_idx: int; n_surfaces: int; n_ads: int; n_persons: int
    last_reward: Dict[str,float]=Field(default_factory=dict)
    surfaces: List[Dict[str,Any]]=Field(default_factory=list)

print('✅ models.py v5 — 7-dim Action (+ rotation_deg + perspective_tilt)')
