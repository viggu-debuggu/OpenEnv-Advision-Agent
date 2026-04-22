import os
import cv2
import numpy as np
import tempfile
from advision_env.models.vision_models import ObjectDetector, DepthEstimator
from advision_env.pipeline.placement_engine import PlacementEngine, PlacementConfig
from advision_env.env.reward import RewardFunction

# Core components (lazy loaded)
detector = None
depth_est = None
engine = None
reward_fn = None

def load_models():
    global detector, depth_est, engine, reward_fn
    if detector is None:
        detector = ObjectDetector()
        depth_est = DepthEstimator()
        engine = PlacementEngine()
        reward_fn = RewardFunction()
    return detector, depth_est, engine, reward_fn

def run_processing_pipeline(
    input_video_path,
    ad_image_bgr,
    config: PlacementConfig,
    progress_callback=None
):
    """
    Core logic for processing a video and placing an ad.
    Returns: (output_video_path, avg_metrics_dict, frame_count)
    """
    det, de, eng, rf = load_models()
    eng.reset()
    rf.__init__() # Reset temporal history

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_process_frames = min(total_frames, 50) # Limit for demo

    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    all_rewards = []
    frame_idx = 0
    
    try:
        while cap.isOpened() and frame_idx < max_process_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if progress_callback:
                progress_callback(frame_idx, max_process_frames)

            # Run detection and depth on first frame or periodically
            if frame_idx == 0:
                surfaces, persons = det.detect(frame)
                depth_map = de.estimate(frame)

                if not surfaces:
                    # Fallback to center if nothing found
                    h, w = frame.shape[:2]
                    mock_corners = np.array([
                        [w*0.3, h*0.3], [w*0.7, h*0.3],
                        [w*0.7, h*0.7], [w*0.3, h*0.7]
                    ], dtype=np.float32)
                    target_corners = mock_corners
                    target_surf_mask = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(target_surf_mask, [target_corners.astype(np.int32)], 1)
                    persons = []
                    depth_map = np.ones((h, w), np.float32) * 0.5
                    target_depth = 0.5
                else:
                    # Use largest surface
                    best = max(surfaces, key=lambda s: s.area)
                    target_corners = best.corners
                    h, w = frame.shape[:2]
                    target_surf_mask = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(target_surf_mask, [target_corners.astype(np.int32)], 1)
                    target_depth = de.region_depth(depth_map, best.bbox)

            # Place ad
            result, bin_mask, adj = eng.place(
                frame,
                ad_image_bgr,
                target_corners,
                persons=persons,
                depth_map=depth_map,
                cfg=config
            )

            # Update corners for next frame temporal calc
            target_corners = adj

            # Calculate reward for this frame
            rc = rf.compute(
                frame, result, bin_mask, target_surf_mask,
                target_depth, persons, corners=adj
            )
            all_rewards.append(rc.to_dict())

            out.write(result)
            frame_idx += 1

    finally:
        cap.release()
        out.release()

    # Post-process with FFmpeg for web compatibility (H.264)
    web_friendly_path = out_path.replace(".mp4", "_web.mp4")
    try:
        import subprocess
        # -vcodec libx264 -pix_fmt yuv420p is the standard for web playback
        cmd = [
            "ffmpeg", "-y", "-i", out_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            web_friendly_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        # Replace the original with the web-friendly version
        os.remove(out_path)
        out_path = web_friendly_path
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}. Falling back to raw OpenCV output.")
        if os.path.exists(web_friendly_path):
            os.remove(web_friendly_path)

    if not all_rewards:
        return out_path, {}, 0

    # Compute average scores
    avg_metrics = {k: float(np.mean([r[k] for r in all_rewards])) for k in all_rewards[0].keys()}
    
    return out_path, avg_metrics, frame_idx
