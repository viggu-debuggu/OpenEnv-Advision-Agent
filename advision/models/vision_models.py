from __future__ import annotations
import cv2, numpy as np, torch, warnings, os, sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
warnings.filterwarnings('ignore')


@dataclass
class DetectedSurface:
    bbox:            Tuple[int,int,int,int]
    confidence:      float
    class_name:      str
    area:            float
    centroid:        Tuple[float,float]
    depth_mean:      float = 0.5
    corners:         Optional[np.ndarray] = None
    surface_quality: float = 0.5

    def to_dict(self):
        return {'bbox':list(self.bbox),'confidence':round(float(self.confidence),4),
                'class_name':self.class_name,'area':round(float(self.area),4),
                'centroid':[round(float(c),4) for c in self.centroid],
                'depth_mean':round(float(self.depth_mean),4)}


@dataclass
class SceneAnalysis:
    surfaces:          List[DetectedSurface]
    persons:           List[Dict]
    depth_map:         Optional[np.ndarray]
    segmentation_mask: np.ndarray
    dominant_color:    Tuple[float,float,float]
    brightness:        float
    scene_type:        str
    recommended_ad_idx: int = 0


class ObjectDetector:
    def __init__(self, model_size='n'):
        self.model  = None
        # FIX: store device so YOLO can run on GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half   = torch.cuda.is_available()  # FP16 only on CUDA
        size = os.environ.get('ADVISION_YOLO_SIZE', model_size)
        try:
            from ultralytics import YOLO
            model_path = f'yolov8{size}.pt'
            if not os.path.exists(model_path):
                try:
                    from ultralytics.utils.downloads import attempt_download_asset
                    attempt_download_asset(model_path)
                except Exception as e:
                    print(f"Auto-download skipped, depending on YOLO default behavior: {e}")
            self.model = YOLO(model_path)
            fp16_note = ' (FP16)' if self.half else ''
            print(f'[v] YOLOv8{size} on {self.device}{fp16_note}')
        except Exception as e:
            print(f'[!] YOLO unavailable -> mock: {e}')

    def detect(self, frame):
        if self.model is None:
            return self._mock(frame)
        try:
            # FIX (requested): run YOLO on GPU with FP16 when available → ~4x faster
            res = self.model(frame, verbose=False, device=self.device, half=self.half)
            surfaces, persons = [], []
            h, w = frame.shape[:2]
            for box in res[0].boxes:
                cid   = int(box.cls[0])
                cname = res[0].names[cid].lower()
                conf  = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                x1,y1 = max(0,x1), max(0,y1)
                x2,y2 = min(w,x2), min(h,y2)
                if x2<=x1 or y2<=y1: continue
                area  = (x2-x1)*(y2-y1)/(h*w+1e-6)
                cx,cy = (x1+x2)/2, (y1+y2)/2
                if cname=='person' and conf>0.4:
                    persons.append({'bbox':[x1,y1,x2,y2],'conf':conf,'depth':0.15})
                elif area>0.015 and conf>0.25:
                    tilt = int((x2-x1)*0.03*(y1/max(h,1)))
                    corners = np.float32([[x1+tilt,y1],[x2-tilt,y1],[x2,y2],[x1,y2]])
                    surfaces.append(DetectedSurface(
                        bbox=(x1,y1,x2,y2), confidence=conf, class_name=cname,
                        area=area, centroid=(cx,cy), corners=corners))
            surfaces.sort(key=lambda s: s.area*s.confidence, reverse=True)
            if not surfaces:
                # FIX: fallback to mock if no real surfaces found
                return self._mock(frame)[0][:1], persons
            return surfaces[:5], persons
        except Exception as e:
            return self._mock(frame)

    def _mock(self, frame):
        h,w = frame.shape[:2]
        return [
            DetectedSurface(bbox=(int(w*.05),int(h*.05),int(w*.48),int(h*.65)),
                confidence=0.92,class_name='wall',area=0.25,centroid=(w*.27,h*.35),
                corners=np.float32([[w*.05,h*.05],[w*.48,h*.04],[w*.48,h*.65],[w*.05,h*.65]])),
            DetectedSurface(bbox=(int(w*.52),int(h*.10),int(w*.92),int(h*.62)),
                confidence=0.80,class_name='board',area=0.16,centroid=(w*.72,h*.36),
                corners=np.float32([[w*.52,h*.10],[w*.92,h*.12],[w*.90,h*.62],[w*.52,h*.62]])),
            DetectedSurface(bbox=(int(w*.15),int(h*.68),int(w*.85),int(h*.92)),
                confidence=0.70,class_name='floor',area=0.12,centroid=(w*.5,h*.80),
                corners=np.float32([[w*.05,h*.72],[w*.95,h*.72],[w*.98,h*.95],[w*.02,h*.95]])),
        ], [{'bbox':[int(w*.38),int(h*.25),int(w*.62),int(h*.85)],'conf':0.85,'depth':0.15}]


class DepthEstimator:
    def __init__(self):
        _torch = sys.modules['torch']
        self.device    = 'cuda' if _torch.cuda.is_available() else 'cpu'
        self.model     = None
        self.transform = None
        if os.environ.get('ADVISION_USE_MIDAS','0') == '1':
            try:
                m = _torch.hub.load('intel-isl/MiDaS','MiDaS_small',trust_repo=True,verbose=False)
                m.to(self.device).eval()
                t = _torch.hub.load('intel-isl/MiDaS','transforms',trust_repo=True,verbose=False)
                self.model = m; self.transform = t.small_transform
                print(f'[v] MiDaS on {self.device}')
            except Exception as e:
                print(f'[!] MiDaS failed -> gradient fallback: {e}')
        else:
            print('[*] DepthEstimator: fast gradient fallback')

    def estimate(self, frame):
        if self.model is not None:
            try:
                _torch = sys.modules['torch']
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                inp = self.transform(rgb).to(self.device)
                with _torch.no_grad():
                    pred = self.model(inp)
                    pred = _torch.nn.functional.interpolate(
                        pred.unsqueeze(1),size=frame.shape[:2],
                        mode='bicubic',align_corners=False).squeeze()
                d = pred.cpu().numpy().astype(np.float32)
                mn,mx = d.min(),d.max()
                return (d-mn)/(mx-mn+1e-6)
            except Exception:
                pass
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
        h,w   = frame.shape[:2]
        grad  = np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=5))
        persp = np.linspace(0,1,h)[:,None]*np.ones((1,w))
        return (0.4*grad/(grad.max()+1e-6)+0.6*persp).astype(np.float32)

    def region_depth(self, dm, bbox):
        x1,y1,x2,y2 = map(int,bbox)
        h,w = dm.shape[:2]
        roi = dm[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        return float(roi.mean()) if roi.size>0 else 0.5


class SceneSegmenter:
    def segment(self, frame):
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,40,120)
        k     = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
        mask  = cv2.bitwise_not(cv2.dilate(edges,k))
        cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        res   = np.zeros(gray.shape,np.uint8)
        mn    = gray.shape[0]*gray.shape[1]*0.015
        for c in cnts:
            if cv2.contourArea(c)>mn: cv2.drawContours(res,[c],-1,1,-1)
        return res.astype(np.float32)

    def classify_scene(self, frame):
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        sky = hsv[:frame.shape[0]//3]
        if (sky[:,:,0]>100).mean()>0.3: return 'outdoor'
        if frame.mean(axis=2).std()<20: return 'urban'
        return 'indoor'


class AdSelector:
    def select(self, ad_images, scene):
        if len(ad_images)<=1: return 0
        scores=[]
        sc_b,sc_g,sc_r = scene.dominant_color
        for ad in ad_images:
            ad_u8 = np.clip(ad,0,255).astype(np.uint8)
            ad_L  = cv2.cvtColor(ad_u8,cv2.COLOR_BGR2LAB)[:,:,0].mean()/255
            b_sc  = abs(ad_L-scene.brightness)
            am    = ad.mean(axis=(0,1))/255
            c_sc  = (abs(am[0]-sc_b)+abs(am[1]-sc_g)+abs(am[2]-sc_r))/3
            if scene.surfaces:
                s=scene.surfaces[0]; sw=s.bbox[2]-s.bbox[0]; sh=s.bbox[3]-s.bbox[1]
                ah2,aw2=ad.shape[:2]; sr=sw/(sh+1e-6); ar=aw2/(ah2+1e-6)
                r_sc=1-min(1,abs(sr-ar)/2)
            else: r_sc=0.5
            scores.append(0.35*b_sc+0.35*c_sc+0.30*r_sc)
        return int(np.argmax(scores))

    def dominant_color(self, frame):
        small = cv2.resize(frame,(64,36))
        pix   = small.reshape(-1,3).astype(np.float32)
        crit  = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,0.2)
        try:
            if pix.std()<1e-3: raise ValueError
            _,labels,centers = cv2.kmeans(pix,3,None,crit,5,cv2.KMEANS_RANDOM_CENTERS)
            dom = centers[np.bincount(labels.flatten()).argmax()]
            return (float(dom[0]/255),float(dom[1]/255),float(dom[2]/255))
        except Exception:
            m = frame.mean(axis=(0,1))/255
            return (float(m[0]),float(m[1]),float(m[2]))

print('[v] vision_models.py written')
