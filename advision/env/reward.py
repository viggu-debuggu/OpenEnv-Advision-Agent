from __future__ import annotations
import cv2, numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class RewardComponents:
    realism: float; alignment: float; lighting: float
    occlusion: float; visibility: float; temporal: float; total: float
    def to_dict(self):
        return {k:round(float(getattr(self,k)),4)
                for k in ['realism','alignment','lighting','occlusion','visibility','temporal','total']}


REWARD_WEIGHTS = dict(realism=0.22, alignment=0.22, lighting=0.18,
                      occlusion=0.18, visibility=0.10, temporal=0.10)


class RewardFunction:
    """
    Reward = weighted sum of 6 components.

    temporal() now includes an EXPLICIT JITTER PENALTY:
      movement = mean L2-distance that corners moved since last frame
      jitter_penalty = min(movement * 0.01, 0.30)
    A perfectly static ad (same corners every frame) gets zero penalty.
    Any camera-induced drift is penalised, incentivising the agent to
    pick stable placements and hold the scale/offset constant.
    """
    def __init__(self):
        self.prev_mask:    Optional[np.ndarray] = None
        self.prev_corners: Optional[np.ndarray] = None

    def _clamp(self, v: float) -> float:
        # Round to 1 decimal, then stay strictly between 0 and 1 (0.1 to 0.9)
        val = round(float(v), 1)
        return float(np.clip(val, 0.1, 0.9))

    def compute(self, frame_before, frame_after, ad_mask, surface_mask,
                surface_depth, persons, corners=None):
        r=self._clamp(self._realism(frame_before,frame_after,ad_mask))
        a=self._clamp(self._alignment(ad_mask,surface_mask))
        l=self._clamp(self._lighting(frame_before,frame_after,ad_mask))
        o=self._clamp(self._occlusion(surface_depth,persons,ad_mask))
        v=self._clamp(self._visibility(ad_mask,frame_before.shape[:2]))
        t=self._clamp(self._temporal(ad_mask,corners))
        W=REWARD_WEIGHTS
        total=(W['realism']*r+W['alignment']*a+W['lighting']*l+
               W['occlusion']*o+W['visibility']*v+W['temporal']*t)
        self.prev_mask    = ad_mask.copy()
        self.prev_corners = corners.copy() if corners is not None else None
        return RewardComponents(r,a,l,o,v,t,self._clamp(total))

    def _realism(self, before, after, mask):
        if mask.sum()==0: return 0.0
        k=np.ones((7,7),np.uint8)
        boundary=cv2.dilate(mask.astype(np.uint8),k)-mask.astype(np.uint8)
        if boundary.sum()==0: return 0.5
        edges=cv2.Canny(cv2.cvtColor(after,cv2.COLOR_BGR2GRAY),50,150).astype(np.float32)
        return float(max(0.,1.-(edges*boundary).sum()/(boundary.sum()+1e-6)/45))

    def _alignment(self, ad_mask, surface_mask):
        if ad_mask.sum()==0 or surface_mask.sum()==0: return 0.0
        sm=surface_mask
        if ad_mask.shape!=sm.shape:
            sm=cv2.resize(sm,(ad_mask.shape[1],ad_mask.shape[0]),interpolation=cv2.INTER_NEAREST)
        ab=(ad_mask>0).astype(np.float32); sb=(sm>0).astype(np.float32)
        return float((ab*sb).sum()/(np.clip(ab+sb,0,1).sum()+1e-6))

    def _lighting(self, before, after, mask):
        if mask.sum()==0: return 0.5
        dilated=cv2.dilate(mask.astype(np.uint8),np.ones((21,21),np.uint8))
        surround=(dilated-mask.astype(np.uint8)).astype(bool)
        ad_reg=mask.astype(bool)
        if surround.sum()==0: return 0.5
        lab=cv2.cvtColor(after,cv2.COLOR_BGR2LAB).astype(np.float32)
        sc_L=lab[:,:,0][surround].mean()
        ad_L=lab[:,:,0][ad_reg].mean() if ad_reg.sum()>0 else sc_L
        return float(max(0.,1.-abs(ad_L-sc_L)/50))

    def _occlusion(self, surf_depth, persons, ad_mask):
        depth_score=min(1.,surf_depth/0.65); penalty=0.
        if persons and ad_mask.sum()>0:
            h,w=ad_mask.shape[:2]; obs=np.zeros((h,w),np.uint8)
            for p in persons:
                x1,y1,x2,y2=p['bbox']
                obs[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]=1
            penalty=min(1.,(obs*ad_mask.astype(np.uint8)).sum()/(ad_mask.sum()+1e-6))
        return float(depth_score*(1.-penalty*0.5))

    def _visibility(self, ad_mask, frame_size):
        ratio=ad_mask.sum()/(frame_size[0]*frame_size[1]+1e-6)
        if ratio<0.01: return 0.
        if ratio<0.05: return float(ratio/0.05)
        if ratio<=0.30: return 1.
        return float(max(0.,1.-(ratio-0.30)/0.30))

    def _temporal(self, ad_mask, corners):
        """
        Temporal stability with EXPLICIT JITTER PENALTY.

        Formula:
          mask_stab    = 1 - mean(|curr_mask - prev_mask|)   [0..1]
          corner_stab  = 1 - mean_L2(curr_corners, prev_corners) / 50
          jitter_penalty = min(mean_L2 * 0.01, 0.30)

        temporal = clip(0.6*mask_stab + 0.4*corner_stab - jitter_penalty, 0, 1)

        Why: The agent must keep corners in the SAME PLACE every frame.
        Moving 5 pixels = -0.05 penalty; moving 30 pixels = -0.30 penalty.
        This directly rewards the temporal-consistency task.
        """
        if self.prev_mask is None:
            return 0.70   # first frame — no history yet

        # Mask stability
        if ad_mask.shape == self.prev_mask.shape:
            mask_stab = 1. - float(
                np.abs(ad_mask.astype(np.float32)-self.prev_mask.astype(np.float32)).mean())
        else:
            mask_stab = 0.5

        # Corner stability + jitter penalty
        if corners is not None and self.prev_corners is not None:
            dists         = np.linalg.norm(corners - self.prev_corners, axis=1)
            mean_movement = float(dists.mean())
            corner_stab   = max(0., 1. - mean_movement / 50.)
            # ← EXPLICIT JITTER PENALTY (requested fix)
            jitter_penalty = min(mean_movement * 0.01, 0.30)
        else:
            corner_stab   = 0.5
            jitter_penalty = 0.0

        raw = 0.6 * mask_stab + 0.4 * corner_stab
        return float(np.clip(raw - jitter_penalty, 0., 1.))

print('[v] reward.py v5')
print('  [v] Explicit jitter penalty: movement * 0.01 subtracted from temporal score')
print('  [v] Perfect static placement -> 0 penalty, max temporal reward')
print('  [v] 30px camera drift -> -0.30 penalty, strong incentive for stability')
