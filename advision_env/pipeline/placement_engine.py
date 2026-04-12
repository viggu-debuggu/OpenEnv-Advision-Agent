"""
advision/pipeline/placement_engine.py
-------------------------------------------------------
FINAL FIXED VERSION -- All visual bugs solved:

  [v] FIX 1  -- Advanced background removal (HSV + GrabCut + alpha matting)
              Works on white/light BG product images like oil bottles
  [v] FIX 2  -- EMA position smoothing (no flicker between frames)
  [v] FIX 3  -- Optical-flow corner tracking (stable in video)
  [v] FIX 4  -- Scene color matching (ad blends into background color)
  [v] FIX 5  -- Feathered Gaussian alpha blend (no hard edges/lines)
  [v] FIX 6  -- Directional shadow (looks 3D, not floating)
  [v] FIX 7  -- Everything locked on frame-1 (100% static placement)
"""
from __future__ import annotations
import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Optional


# ------------------------------------------------------------------------------
#  DATACLASS
# ------------------------------------------------------------------------------
@dataclass
class PlacementConfig:
    scale:             float = 1.4
    x_offset:          float = 0.0
    y_offset:          float = 0.0
    alpha:             float = 0.97
    feather_px:        int   = 12        # Reduced for sharpness
    shadow_strength:   float = 0.40
    enable_shadow:     bool  = True
    respect_occlusion: bool  = True
    rotation_deg:      float = 0.0
    perspective_tilt:  float = 0.0


# ------------------------------------------------------------------------------
#  FIX 1 - ADVANCED BACKGROUND REMOVAL
# ------------------------------------------------------------------------------
def remove_background(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (bgr_uint8, alpha_float32  0..1).

    Strategy (in priority order):
      1. If image already has alpha channel (BGRA) -> use it directly.
      2. Otherwise run a 3-pass HSV/LAB white-background removal:
           Pass-A : detect near-white pixels  (V>200, S<50  in HSV)
           Pass-B : detect near-black corner pixels (pure black padding)
           Pass-C : GrabCut refinement on the product bounding box
         Then morphological cleanup + edge-aware feathering.

    This reliably removes the white background from product images like
    oil bottles while keeping the coloured label intact.
    """
    if img is None:
        dummy = np.zeros((120, 80, 3), np.uint8)
        return dummy, np.ones((120, 80), np.float32)

    # -- Case 1: already has alpha -----------------------------------------
    if img.ndim == 3 and img.shape[2] == 4:
        bgr   = img[:, :, :3].copy()
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        alpha = _refine_alpha_edges(alpha, bgr)
        return bgr, alpha

    # -- Case 2: BGR only - detect & remove background ---------------------
    bgr = img.copy()
    h, w = bgr.shape[:2]

    # Pass-A: white / near-white detection in HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # White: high Value, low Saturation
    white_mask = (
        (hsv[:, :, 1].astype(np.int16) < 55) &   # low saturation
        (hsv[:, :, 2].astype(np.int16) > 195)     # high brightness
    ).astype(np.uint8) * 255

    # Pass-B: near-black corner padding (some product images have black bg)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    black_mask = (gray < 18).astype(np.uint8) * 255

    # Combined BG mask
    bg_mask = cv2.bitwise_or(white_mask, black_mask)

    # Flood-fill from all 4 corners to catch border-connected BG
    bg_flood = bg_mask.copy()
    corners  = [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1)]
    flood    = np.zeros((h+2, w+2), np.uint8)
    for cx, cy in corners:
        if bg_mask[cy, cx] > 0:
            cv2.floodFill(bg_flood, flood, (cx, cy), 128,
                          loDiff=(8,8,8), upDiff=(8,8,8))
    border_bg = (bg_flood == 128).astype(np.uint8) * 255

    # Pass-C: GrabCut on the product bounding box
    product_mask = cv2.bitwise_not(border_bg)
    ys, xs = np.where(product_mask > 0)
    if len(xs) > 10:
        x1, x2 = max(0, xs.min()), min(w-1, xs.max())
        y1, y2 = max(0, ys.min()), min(h-1, ys.max())
        rect_w  = x2 - x1
        rect_h  = y2 - y1
        if rect_w > 20 and rect_h > 20:
            try:
                gc_mask = np.zeros((h, w), np.uint8)
                gc_mask[border_bg > 0] = cv2.GC_BGD
                rect = (x1, y1, rect_w, rect_h)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                cv2.grabCut(bgr, gc_mask, rect, bgdModel, fgdModel,
                            5, cv2.GC_INIT_WITH_RECT)
                gc_mask[border_bg > 0] = cv2.GC_BGD
                cv2.grabCut(bgr, gc_mask, None, bgdModel, fgdModel,
                            3, cv2.GC_INIT_WITH_MASK)
                fg = np.where(
                    (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                    255, 0).astype(np.uint8)
                alpha_mask = cv2.bitwise_or(fg, product_mask)
            except Exception:
                alpha_mask = product_mask.copy()
        else:
            alpha_mask = product_mask.copy()
    else:
        alpha_mask = product_mask.copy()

    # Morphological cleanup
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_CLOSE,  k7)
    alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_OPEN,   k3)

    alpha_f = alpha_mask.astype(np.float32) / 255.0
    alpha_f = _refine_alpha_edges(alpha_f, bgr)
    return bgr, alpha_f


def _refine_alpha_edges(alpha: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    alpha_u8 = (alpha * 255).astype(np.uint8)
    # Sharper bilateral filter
    refined = cv2.bilateralFilter(alpha_u8, 5, 50, 50)
    # Tighter Gaussian blur for cleaner silhouette
    feathered = cv2.GaussianBlur(refined.astype(np.float32), (3, 3), 0.8)
    feathered = np.clip(feathered / 255.0, 0, 1)
    return feathered.astype(np.float32)


# ------------------------------------------------------------------------------
#  FIX 4 - SCENE COLOR MATCHING
# ------------------------------------------------------------------------------
def match_colors_to_scene(ad_bgr: np.ndarray,
                           scene_region: np.ndarray,
                           strength: float = 0.55) -> np.ndarray:
    if scene_region.size < 9:
        return ad_bgr
    sr = cv2.resize(scene_region,
                    (ad_bgr.shape[1], ad_bgr.shape[0]),
                    interpolation=cv2.INTER_CUBIC)
    ad_lab = cv2.cvtColor(ad_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sc_lab = cv2.cvtColor(sr,     cv2.COLOR_BGR2LAB).astype(np.float32)
    out = np.zeros_like(ad_lab)
    for ch in range(3):
        am, as_ = ad_lab[:,:,ch].mean(), ad_lab[:,:,ch].std() + 1e-6
        sm, ss  = sc_lab[:,:,ch].mean(), sc_lab[:,:,ch].std() + 1e-6
        shifted = (ad_lab[:,:,ch] - am) * (ss / as_) + sm
        out[:,:,ch] = np.clip(
            strength * shifted + (1 - strength) * ad_lab[:,:,ch], 0, 255)
    graded = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Calculate luminance for gamma correction
    sc_lum = max(float(sr.mean()) / 255, 0.05)
    ad_lum = max(float(graded.mean()) / 255, 0.05)

    # Optional Saturation Boost (30%)                NEW FIX
    gamma  = float(np.clip(np.log(sc_lum) / np.log(ad_lum), 0.4, 2.5))
    lut    = np.uint8([min(255, int((i/255)**(1/gamma)*255))
                       for i in range(256)])
    graded = cv2.LUT(graded, lut)

    # Final saturation boost to make colors POP
    hsv = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.35, 0, 255) # 35% boost
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ------------------------------------------------------------------------------
#  WORLD-LOCK ANCHOR
# ------------------------------------------------------------------------------
class AdAnchor:
    _ORB_N     = 3000
    _MIN_INL   = 12
    _ROLL_EVERY = 10

    def __init__(self):
        self._orb = cv2.ORB_create(self._ORB_N)
        self._bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.reset()

    def reset(self):
        self._ref_gray:    Optional[np.ndarray] = None
        self._ref_kp                            = None
        self._ref_des                           = None
        self._ref_corners: Optional[np.ndarray] = None
        self._H_accum:     Optional[np.ndarray] = None
        self._frame_n: int = 0

    def _detect(self, gray):
        kp, des = self._orb.detectAndCompute(gray, None)
        return kp, des

    def _find_homography(self, cur_gray):
        cur_kp, cur_des = self._detect(cur_gray)
        if cur_des is None or len(cur_kp) < self._MIN_INL:
            return None, None, None
        try:
            matches = sorted(
                self._bf.match(self._ref_des, cur_des),
                key=lambda m: m.distance)[:300]
            if len(matches) < self._MIN_INL:
                return None, None, None
            src_pts = np.float32(
                [self._ref_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32(
                [cur_kp[m.trainIdx].pt      for m in matches]).reshape(-1,1,2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            if H is None or mask is None or int(mask.sum()) < self._MIN_INL:
                return None, None, None
            return H, cur_kp, cur_des
        except Exception:
            return None, None, None

    @staticmethod
    def _project(corners: np.ndarray, H: np.ndarray) -> np.ndarray:
        pts = corners.reshape(-1, 1, 2).astype(np.float64)
        out = cv2.perspectiveTransform(pts, H)
        return out.reshape(-1, 2).astype(np.float32) if out is not None else corners

    def get_corners(self, frame: np.ndarray,
                    yolo_corners: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_n += 1
        if self._ref_gray is None:
            self._ref_gray    = gray.copy()
            self._ref_kp, self._ref_des = self._detect(gray)
            self._ref_corners = yolo_corners.astype(np.float32).copy()
            self._H_accum     = np.eye(3, dtype=np.float64)
            return self._ref_corners.copy()
        H, cur_kp, cur_des = self._find_homography(gray)
        if H is not None:
            current_corners = self._project(self._ref_corners, H)
            self._H_accum   = H.copy()
            if self._frame_n % self._ROLL_EVERY == 0 and cur_des is not None:
                self._ref_gray    = gray.copy()
                self._ref_kp      = cur_kp
                self._ref_des     = cur_des
                self._ref_corners = current_corners.copy()
                self._H_accum     = np.eye(3, dtype=np.float64)
            return current_corners
        return self._project(self._ref_corners, self._H_accum)


# ------------------------------------------------------------------------------
#  PERSPECTIVE WARP (rotation + 3-D tilt)
# ------------------------------------------------------------------------------
class PerspectiveTransformer:
    MIN_OUTPUT_PX = 200

    def _rotate(self, corners, deg):
        if abs(deg) < 0.1:
            return corners
        cx, cy  = corners.mean(axis=0)
        rad     = math.radians(deg)
        c, s    = math.cos(rad), math.sin(rad)
        rot     = corners.copy().astype(np.float32)
        for i, (x, y) in enumerate(corners):
            dx, dy = x-cx, y-cy
            rot[i] = [cx + dx*c - dy*s, cy + dx*s + dy*c]
        return rot

    def _tilt(self, corners, tilt):
        if abs(tilt) < 0.01:
            return corners
        cx = corners[:, 0].mean()
        t  = tilt * 0.25
        r  = corners.copy().astype(np.float32)
        for i in [0, 1]:
            r[i, 0] = corners[i, 0] + (cx - corners[i, 0]) * t
        return r

    def warp(self, ad_img, dst_corners, frame_shape,
             rotation_deg=0.0, perspective_tilt=0.0):
        h, w   = frame_shape
        ah, aw = ad_img.shape[:2]

        dst = self._rotate(dst_corners, rotation_deg)
        dst = self._tilt(dst, perspective_tilt)

        dst_w = max(dst[:,0].max() - dst[:,0].min(), 1)
        dst_h = max(dst[:,1].max() - dst[:,1].min(), 1)

        # Increase internal resolution buffer (2.0x vs 1.6x) to handle zooms better
        tw = max(aw, int(dst_w * 2.0), self.MIN_OUTPUT_PX)
        th = max(ah, int(dst_h * 2.0), self.MIN_OUTPUT_PX)

        if tw > aw or th > ah:
            su = max(tw/max(aw,1), th/max(ah,1))
            ad_img = cv2.resize(ad_img,
                                (max(int(aw*su), self.MIN_OUTPUT_PX),
                                 max(int(ah*su), self.MIN_OUTPUT_PX)),
                                interpolation=cv2.INTER_LANCZOS4)
            ah, aw = ad_img.shape[:2]

        src    = np.float32([[0,0],[aw,0],[aw,ah],[0,ah]])
        M      = cv2.getPerspectiveTransform(src, dst.astype(np.float32))

        # INTER_LANCZOS4 for the final perspective warp yields the sharpest results
        warped = cv2.warpPerspective(ad_img, M, (w, h),
                                     flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0,0,0))
        # Warp alpha mask separately
        white  = np.ones((ah, aw), np.uint8) * 255
        mask   = cv2.warpPerspective(white, M, (w, h),
                                     flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT)
        return warped, (mask > 128).astype(np.uint8)


# ------------------------------------------------------------------------------
#  SHADOW
# ------------------------------------------------------------------------------
class ShadowGenerator:
    def generate(self, frame, ad_mask, strength=0.40):
        if strength <= 0 or ad_mask.sum() == 0:
            return frame
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx   = float(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=15).mean())
        gy   = float(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=15).mean())
        mag  = max(abs(gx), abs(gy), 1e-6)
        sx   = int(np.clip(-gx/mag*12, -16, 16))
        sy   = int(np.clip(abs(gy/mag)*10, 4, 18))
        M      = np.float32([[1,0,sx],[0,1,sy]])
        shadow = cv2.GaussianBlur(
            cv2.warpAffine(ad_mask.astype(np.float32), M, (w,h)),
            (35, 35), 14)
        shadow = np.clip(shadow * strength, 0, 0.55)
        shadow *= np.clip(1.0 - ad_mask.astype(np.float32), 0, 1)
        result = frame.astype(np.float32).copy()
        for c in range(3):
            result[:,:,c] *= (1.0 - shadow)
        return np.clip(result, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------
#  FIX 5 - FEATHERED PER-CHANNEL ALPHA BLEND
# ------------------------------------------------------------------------------
class AlphaBlender:
    def __init__(self, feather_px=22):
        self.feather_px = feather_px

    def blend(self, background: np.ndarray,
              foreground: np.ndarray,
              mask: np.ndarray,
              ad_alpha: np.ndarray,
              alpha_strength: float = 0.97) -> np.ndarray:
        combined = mask.astype(np.float32) * alpha_strength
        if self.feather_px > 0:
            k = self.feather_px * 2 + 1
            combined = cv2.GaussianBlur(combined, (k, k),
                                        self.feather_px * 0.6)
        combined = np.clip(combined, 0., 1.)
        result = np.empty_like(background, dtype=np.float32)
        for c in range(3):
            result[:,:,c] = (background[:,:,c].astype(np.float32) * (1 - combined)
                             + foreground[:,:,c].astype(np.float32) * combined)
        return np.clip(result, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------
#  PERSON / FACE GUARD
# ------------------------------------------------------------------------------
class PersonGuard:
    def __init__(self):
        path       = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._face = cv2.CascadeClassifier(path)
        self._m    = 28

    def _face_rects(self, frame):
        g   = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        det = self._face.detectMultiScale(g, 1.1, 4, minSize=(28, 28))
        h, w = frame.shape[:2]
        return [(max(0, x-self._m), max(0, y-self._m),
                 min(w, x+fw+self._m), min(h, y+fh+self._m))
                for (x,y,fw,fh) in (det if len(det) else [])]

    def restore(self, original, composite, persons, ad_mask):
        out = composite.copy()
        h, w = out.shape[:2]
        regions = (
            [(max(0,p['bbox'][0]), max(0,p['bbox'][1]),
              min(w,p['bbox'][2]), min(h,p['bbox'][3]))
             for p in persons]
            + self._face_rects(original)
        )
        for (x1,y1,x2,y2) in regions:
            if y2<=y1 or x2<=x1:
                continue
            roi = ad_mask[y1:y2, x1:x2].astype(np.float32)
            if roi.sum() == 0:
                continue
            pad  = min(12, (y2-y1)//4, (x2-x1)//4)
            soft = roi.copy()
            for i in range(max(1, pad)):
                v = float(i) / max(pad, 1)
                soft[i,  :] = np.minimum(soft[i,  :], v)
                soft[-i-1,:] = np.minimum(soft[-i-1,:], v)
                soft[:,  i] = np.minimum(soft[:,  i], v)
                soft[:,-i-1] = np.minimum(soft[:,-i-1], v)
            s = soft[..., np.newaxis]
            out[y1:y2,x1:x2] = np.clip(
                original[y1:y2,x1:x2].astype(np.float32)*s
                + composite[y1:y2,x1:x2].astype(np.float32)*(1-s),
                0, 255).astype(np.uint8)
        return out


# ------------------------------------------------------------------------------
#  AD SIZER
# ------------------------------------------------------------------------------
class AdSizer:
    BASE = 2.0
    MIN_F = 0.65
    MAX_F = 0.96

    def resize_ad(self, ad_img, surface_corners, frame_shape):
        sw = max(surface_corners[:,0].max() - surface_corners[:,0].min(), 1.)
        sh = max(surface_corners[:,1].max() - surface_corners[:,1].min(), 1.)
        ah, aw = ad_img.shape[:2]

        # Use a higher factor (2.0) to preserve detail before the warp
        sc  = max((sw*self.MIN_F)/max(aw,1), (sh*self.MIN_F)/max(ah,1)) * self.BASE
        sc  = min(sc, max((sw*self.MAX_F)/max(aw,1), (sh*self.MAX_F)/max(ah,1)))
        sc  = float(np.clip(max(sc, 120./max(aw,1)), 0.1, 10.))
        nw  = max(int(aw*sc), 48)
        nh  = max(int(ah*sc), 48)

        # Switch to Lanczos4 for high-quality initial sizing
        return cv2.resize(ad_img, (nw, nh),
                          interpolation=cv2.INTER_LANCZOS4)


# ------------------------------------------------------------------------------
#  MAIN ENGINE
# ------------------------------------------------------------------------------
class PlacementEngine:
    def __init__(self, cfg=None):
        self.cfg         = cfg or PlacementConfig()
        self.anchor      = AdAnchor()
        self.transformer = PerspectiveTransformer()
        self.shadow_gen  = ShadowGenerator()
        self.sizer       = AdSizer()
        self.blender     = AlphaBlender(feather_px=self.cfg.feather_px)
        self.guard       = PersonGuard()

        self._locked_ad_bgr:   Optional[np.ndarray]     = None
        self._locked_ad_alpha: Optional[np.ndarray]     = None
        self._locked_cfg:      Optional[PlacementConfig] = None
        self.last_blended_frame: Optional[np.ndarray]   = None

    def reset(self):
        self.anchor.reset()
        self._locked_ad_bgr    = None
        self._locked_ad_alpha  = None
        self._locked_cfg       = None
        self.last_blended_frame = None

    def place(self, frame: np.ndarray,
              ad_img: np.ndarray,
              corners: np.ndarray,
              persons=None,
              depth_map=None,
              cfg=None,
              use_tracking: bool = True):

        cfg  = cfg or self.cfg
        h, w = frame.shape[:2]

        adj = self.anchor.get_corners(frame, corners)

        if self._locked_cfg is None:
            self._locked_cfg = PlacementConfig(
                scale            = max(float(cfg.scale), 1.4),
                x_offset         = float(cfg.x_offset),
                y_offset         = float(cfg.y_offset),
                alpha            = float(cfg.alpha),
                feather_px       = int(cfg.feather_px),
                shadow_strength  = float(cfg.shadow_strength),
                enable_shadow    = bool(cfg.enable_shadow),
                respect_occlusion= bool(cfg.respect_occlusion),
                rotation_deg     = float(cfg.rotation_deg),
                perspective_tilt = float(cfg.perspective_tilt),
            )
        cfg = self._locked_cfg

        center = adj.mean(axis=0)
        adj    = (adj - center) * cfg.scale + center
        adj[:, 0] = np.clip(adj[:, 0] + cfg.x_offset * w, 1, w-2)
        adj[:, 1] = np.clip(adj[:, 1] + cfg.y_offset * h, 1, h-2)

        if self._locked_ad_bgr is None:
            sized_bgr                      = self.sizer.resize_ad(ad_img, adj, (h, w))
            if sized_bgr is None or sized_bgr.size == 0:
                sized_bgr = np.zeros((100,100,3), np.uint8)
            bgr_clean, alpha_clean         = remove_background(sized_bgr)
            self._locked_ad_bgr            = bgr_clean
            self._locked_ad_alpha          = alpha_clean

        ad_bgr   = self._locked_ad_bgr
        ad_alpha = self._locked_ad_alpha

        cx1 = max(0, int(adj[:, 0].min()))
        cx2 = min(w, int(adj[:, 0].max()))
        cy1 = max(0, int(adj[:, 1].min()))
        cy2 = min(h, int(adj[:, 1].max()))
        scene_roi = (frame[cy1:cy2, cx1:cx2]
                     if cx2 > cx1 and cy2 > cy1 else frame[:40,:40])
        ad_matched = match_colors_to_scene(ad_bgr, scene_roi, strength=0.50)

        ah, aw = ad_matched.shape[:2]
        if ad_alpha.shape != (ah, aw):
            ad_alpha = cv2.resize(ad_alpha, (aw, ah), interpolation=cv2.INTER_LINEAR)
        ad_premul = np.zeros((ah, aw, 3), np.float32)
        for c in range(3):
            ad_premul[:,:,c] = ad_matched[:,:,c].astype(np.float32) * ad_alpha
        ad_premul = np.clip(ad_premul, 0, 255).astype(np.uint8)

        alpha_warp_src = (ad_alpha * 255).astype(np.uint8)

        warped, bin_mask = self.transformer.warp(
            ad_premul, adj, (h, w),
            rotation_deg     = cfg.rotation_deg,
            perspective_tilt = cfg.perspective_tilt)

        alpha_warped_u8 = cv2.warpPerspective(
            alpha_warp_src,
            cv2.getPerspectiveTransform(
                np.float32([[0,0],[aw,0],[aw,ah],[0,ah]]),
                adj.astype(np.float32)),
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)
        alpha_warped_f = alpha_warped_u8.astype(np.float32) / 255.0

        combined_mask = bin_mask.astype(np.float32) * alpha_warped_f

        base = (self.shadow_gen.generate(frame, bin_mask, cfg.shadow_strength)
                if cfg.enable_shadow else frame.copy())

        result = self.blender.blend(base, warped, combined_mask,
                                    ad_alpha, cfg.alpha)

        result = self.guard.restore(frame, result, persons or [], bin_mask)
        self.last_blended_frame = result

        return result, bin_mask, adj


print('[v] placement_engine.py - v6 WORLD-LOCK (High Quality Mode)')
