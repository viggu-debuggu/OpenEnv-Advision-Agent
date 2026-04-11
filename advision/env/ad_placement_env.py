from __future__ import annotations
import numpy as np, cv2, gymnasium as gym, os, sys
from gymnasium import spaces
from typing import Optional, List, Dict, Any

_root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path: sys.path.insert(0,_root)

from advision.models.vision_models import (
    ObjectDetector,DepthEstimator,SceneSegmenter,
    AdSelector,SceneAnalysis,DetectedSurface)
from advision.pipeline.placement_engine import PlacementEngine,PlacementConfig
from advision.env.reward import RewardFunction


class AdPlacementEnv(gym.Env):
    """
    7-dimensional action space (UPGRADED from 5D):
      [0] surface_idx      0..1    which detected surface to use
      [1] x_offset        -0.2..0.2  horizontal shift (fraction of frame width)
      [2] y_offset        -0.2..0.2  vertical shift
      [3] scale            0.5..1.5  ad size relative to surface
      [4] ad_idx           0..1    which uploaded ad image to use
      [5] rotation_deg   -30..30   clockwise rotation of the ad (degrees)
      [6] perspective_tilt 0..1    top-edge perspective tilt (0=frontal, 1=angled)

    Observation: 28-dim float32 Box
      [0:3]  mean BGR / 255
      [3]    brightness
      [4]    edge density
      [5]    sharpness
      [6:26] 5 surfaces × 4 dims (cx, cy, area, depth)
      [26]   rotation hint (dominant edge angle, 0..1)
      [27]   scene depth gradient (0..1)
    """
    metadata = {'render_modes':['rgb_array'],'render_fps':30}
    OBS_DIM  = 28   # upgraded from 26

    def __init__(self, video_path=None, ad_paths=None,
                 max_frames=60, render_mode='rgb_array', seed=42):
        super().__init__()
        self.video_path=video_path; self.ad_paths=ad_paths or []
        self.max_frames=max_frames; self.render_mode=render_mode; self._seed=seed

        self.observation_space=spaces.Box(0.,1.,shape=(self.OBS_DIM,),dtype=np.float32)

        # 7-DIM ACTION SPACE
        self.action_space=spaces.Box(
            low =np.float32([0., -0.2, -0.2, 0.5, 0., -30.,  0.]),
            high=np.float32([1.,  0.2,  0.2, 1.5, 1.,  30.,  1.]),
            dtype=np.float32)

        self.detector=ObjectDetector(); self.depth_est=DepthEstimator()
        self.segmenter=SceneSegmenter(); self.selector=AdSelector()
        self.engine=PlacementEngine(); self.reward_fn=RewardFunction()

        self._cap: Optional[cv2.VideoCapture]=None
        self._frame: Optional[np.ndarray]=None
        self._depth_map: Optional[np.ndarray]=None
        self._surfaces: List[DetectedSurface]=[]
        self._persons: List[Dict]=[]
        self._ad_images: List[np.ndarray]=[]
        self._frame_idx: int=0
        self._last_result: Optional[np.ndarray]=None
        self._last_mask: Optional[np.ndarray]=None
        self._last_reward_info: Dict[str,Any]={}
        self._scene: Optional[SceneAnalysis]=None
        self._using_mock: bool=False
        self._mock_frames: int=0
        self._prev_good_surfaces: List[DetectedSurface]=[]
        self._prev_fkey: bytes=b''
        self._locked_si: Optional[int]=None
        self._locked_ai: Optional[int]=None
        self._load_ads()

    def _load_ads(self):
        for p in self.ad_paths:
            if p and os.path.exists(p):
                # FIX 1: IMREAD_UNCHANGED preserves alpha channel (PNG transparency)
                img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # remove_background handles BGRA and BGR both
                    from advision.pipeline.placement_engine import remove_background
                    bgr, _alpha = remove_background(img)
                    self._ad_images.append(bgr)
        if not self._ad_images: self._ad_images=[self._default_ad()]

    def _default_ad(self):
        ad=np.zeros((100,200,3),np.uint8); ad[:,:]=(0,120,255)
        cv2.rectangle(ad,(4,4),(195,95),(255,200,0),3)
        cv2.putText(ad,'YOUR AD HERE',(15,58),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
        return ad

    def reset(self, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self._seed)
        self._frame_idx=0; self._mock_frames=0; self._using_mock=False
        self._prev_good_surfaces=[]; self._prev_fkey=b''
        self._locked_si=None; self._locked_ai=None
        self.engine.reset(); self.reward_fn=RewardFunction()
        if self.video_path and os.path.exists(self.video_path):
            if self._cap: self._cap.release()
            self._cap=cv2.VideoCapture(self.video_path)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        else:
            self._cap=None
        self._frame=self._next_frame(); self._analyze()
        return self._obs(),{'frame_idx':0,'n_surfaces':len(self._surfaces),'n_persons':len(self._persons)}

    def step(self, action):
        action=np.clip(action,self.action_space.low,self.action_space.high)
        # Unpack 7-dim action
        surf_n, x_off, y_off, scale, ad_n, rot_deg, persp_tilt = action

        n_surf=max(1,len(self._surfaces))
        if self._locked_si is None:
            self._locked_si=min(int(surf_n*n_surf),n_surf-1)
        if self._locked_ai is None:
            raw_ai=min(int(ad_n*len(self._ad_images)),len(self._ad_images)-1)
            if self._scene and len(self._ad_images)>1:
                raw_ai=self.selector.select(self._ad_images,self._scene)
            self._locked_ai=raw_ai
        si=self._locked_si; ai=self._locked_ai
        surface=self._surfaces[si]; ad_img=self._ad_images[ai]

        alpha_val = float(getattr(self, '_override_alpha', 0.97))
        cfg=PlacementConfig(
            scale=float(scale), x_offset=float(x_off), y_offset=float(y_off),
            alpha=alpha_val, shadow_strength=0.45, enable_shadow=True,
            respect_occlusion=True,
            rotation_deg=float(rot_deg),
            perspective_tilt=float(persp_tilt))

        frame_before=self._frame.copy() if self._frame is not None else np.zeros((360,640,3),np.uint8)
        try:
            result,mask,adj=self.engine.place(self._frame,ad_img,surface.corners,
                                           self._persons,self._depth_map,cfg)
            surface.corners = adj
        except Exception:
            result=frame_before.copy(); mask=np.zeros(self._frame.shape[:2],np.uint8)

        self._last_result=result; self._last_mask=mask
        h,w=self._frame.shape[:2]
        surf_mask=np.zeros((h,w),np.uint8)
        if surface.corners is not None:
            cv2.fillPoly(surf_mask,[surface.corners.astype(np.int32)],1)
        depth=float(self._depth_map[
            int(np.clip(surface.centroid[1],0,h-1)),
            int(np.clip(surface.centroid[0],0,w-1))
        ]) if self._depth_map is not None else 0.5

        rc=self.reward_fn.compute(frame_before,result,mask,surf_mask,depth,
                                   self._persons,corners=surface.corners)
        raw_reward=rc.total*(0.35 if self._using_mock else 1.0)
        # Round to 1 decimal, then stay strictly between 0 and 1 (0.1 to 0.9)
        val = round(float(raw_reward), 1)
        reward=float(np.clip(val,0.1,0.9))
        self._last_reward_info=rc.to_dict()
        self._last_reward_info['using_mock']=self._using_mock

        next_f=self._next_frame()
        if next_f is not None: self._frame=next_f; self._analyze()
        self._frame_idx+=1
        terminated=self._frame_idx>=self.max_frames
        mock_ratio=self._mock_frames/max(1,self._frame_idx)
        truncated=self._frame_idx>=10 and mock_ratio>0.80
        info={'frame_idx':self._frame_idx,'reward_components':self._last_reward_info,
              'surface_used':si,'ad_used':ai,'n_surfaces':len(self._surfaces),
              'n_persons':len(self._persons),'using_mock':self._using_mock,
              'mock_ratio':round(mock_ratio,3)}
        return self._obs(),float(reward),terminated,truncated,info

    def render(self):
        if self._last_result is None: return None
        return cv2.cvtColor(self._last_result,cv2.COLOR_BGR2RGB)

    def close(self):
        if self._cap is not None: self._cap.release(); self._cap=None

    def state(self):
        return {'frame_idx':self._frame_idx,'n_surfaces':len(self._surfaces),
                'n_ads':len(self._ad_images),'n_persons':len(self._persons),
                'last_reward':self._last_reward_info,
                'surfaces':[s.to_dict() for s in self._surfaces]}

    def _next_frame(self):
        if self._cap is not None and self._cap.isOpened():
            ret,fr=self._cap.read()
            if ret and fr is not None and fr.size>0:
                return cv2.resize(fr,(640,360))
        return self._synthetic_frame()

    def _synthetic_frame(self):
        rng=np.random.RandomState(self._seed+self._frame_idx)
        frame=np.zeros((360,640,3),np.uint8)
        for y in range(360):
            v=int(30+190*(1-y/360)); frame[y,:]=(v//3,v//2,v)
        col=int(rng.randint(185,225))
        cv2.rectangle(frame,(30,30),(300,230),(col,col-5,col-10),-1)
        for y in range(30,230,16):
            for x in range(30,300,32):
                cv2.rectangle(frame,(x,y),(x+29,y+13),(col-18,col-22,col-12),-1)
        cv2.rectangle(frame,(330,20),(620,180),(220,180,140),-1)
        cv2.rectangle(frame,(330,20),(620,180),(100,80,60),4)
        cv2.fillPoly(frame,[np.array([[0,270],[640,270],[700,360],[-60,360]])],(90,85,80))
        noise=rng.randint(-6,6,frame.shape,dtype=np.int16)
        return np.clip(frame.astype(np.int16)+noise,0,255).astype(np.uint8)

    def _analyze(self):
        if self._frame is None: return
        _fkey=self._frame.tobytes()
        if _fkey==self._prev_fkey: return
        self._prev_fkey=_fkey
        if self._frame_idx % 5 == 0 or not self._prev_good_surfaces:
            detected,self._persons=self.detector.detect(self._frame)
            if detected:
                self._surfaces=detected; self._using_mock=False
                self._prev_good_surfaces=detected
            elif self._prev_good_surfaces:
                self._surfaces=self._prev_good_surfaces; self._using_mock=False
            else:
                self._surfaces=self.detector._mock(self._frame)[0]
                self._using_mock=True; self._mock_frames+=1
            self._depth_map=self.depth_est.estimate(self._frame)
            for s in self._surfaces:
                s.depth_mean=self.depth_est.region_depth(self._depth_map,s.bbox)
            seg_mask=self.segmenter.segment(self._frame)
            scene_type=self.segmenter.classify_scene(self._frame)
        else:
            seg_mask=self._scene.segmentation_mask if self._scene else __import__('numpy').zeros((self._frame.shape[0], self._frame.shape[1]), __import__('numpy').uint8)
            scene_type=self._scene.scene_type if self._scene else "unknown"
        dom_color=self.selector.dominant_color(self._frame)
        brightness=float(self._frame.mean())/255
        self._scene=SceneAnalysis(surfaces=self._surfaces,persons=self._persons,
                                   depth_map=self._depth_map,segmentation_mask=seg_mask,
                                   dominant_color=dom_color,brightness=brightness,
                                   scene_type=scene_type)

    def _obs(self):
        obs=np.zeros(self.OBS_DIM,np.float32)
        if self._frame is None: return obs
        mean_bgr=self._frame.mean(axis=(0,1))/255
        gray=cv2.cvtColor(self._frame,cv2.COLOR_BGR2GRAY)
        edge_density=float(cv2.Canny(gray,50,150).mean())/255
        sharpness=float(np.clip(cv2.Laplacian(gray,cv2.CV_64F).var()/5000,0,1))
        obs[0:3]=np.clip(mean_bgr,0,1)
        obs[3]=float(self._frame.mean())/255; obs[4]=edge_density; obs[5]=sharpness
        h,w=self._frame.shape[:2]
        for i,s in enumerate(self._surfaces[:5]):
            base=6+i*4
            obs[base]=float(np.clip(s.centroid[0]/w,0,1))
            obs[base+1]=float(np.clip(s.centroid[1]/h,0,1))
            obs[base+2]=float(np.clip(s.area,0,1))
            obs[base+3]=float(np.clip(s.depth_mean,0,1))
        # NEW dim 26: dominant edge angle (rotation hint for the agent)
        sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
        sobely=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
        angle=float(np.arctan2(sobely.mean(),sobelx.mean()+1e-6))
        obs[26]=float(np.clip((angle+np.pi)/(2*np.pi),0,1))
        # NEW dim 27: scene depth gradient (top-to-bottom brightness falloff)
        obs[27]=float(np.clip(gray.astype(np.float32).mean(axis=1).std()/128,0,1))
        return obs

print('[v] ad_placement_env.py v5 - 7-dim action space')
print('  Actions: [surface_idx, x_offset, y_offset, scale, ad_idx, rotation_deg, perspective_tilt]')
print('  Obs: 28-dim (added rotation_hint + depth_gradient)')
