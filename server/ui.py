import os
import sys
import cv2
import numpy as np
import tempfile
import gradio as gr
from pathlib import Path

# Setup paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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

def process_video(
    input_video,
    ad_image,
    scale,
    rotation,
    tilt,
    alpha,
    feather,
    shadow_strength,
    progress=gr.Progress()
):
    if input_video is None or ad_image is None:
        return None, "Please upload both a video and an ad image."

    det, de, eng, rf = load_models()
    eng.reset()
    rf.__init__() # Reset temporal history
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        return None, "Error: Could not open video file."

    # Gradio provides RGB, but our OpenCV engine expects BGR
    ad_image_bgr = cv2.cvtColor(ad_image, cv2.COLOR_RGB2BGR)

    # Video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_process_frames = min(total_frames, 300) # Limit for demo

    # Output setup
    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    cfg = PlacementConfig(
        scale=scale,
        rotation_deg=rotation,
        perspective_tilt=tilt,
        alpha=alpha,
        feather_px=int(feather),
        shadow_strength=shadow_strength,
        enable_shadow=(shadow_strength > 0)
    )

    all_rewards = []
    frame_idx = 0
    try:
        while cap.isOpened() and frame_idx < max_process_frames:
            ret, frame = cap.read()
            if not ret:
                break

            progress(frame_idx / max_process_frames, desc=f"Processing frame {frame_idx}...")

            # Run detection and depth on first frame or periodically
            if frame_idx == 0:
                surfaces, persons = detector.detect(frame)
                depth_map = depth_est.estimate(frame)
                
                if not surfaces:
                    # Fallback to center if nothing found
                    h, w = frame.shape[:2]
                    from advision_env.models.vision_models import DetectedSurface
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
                    target_depth = depth_est.region_depth(depth_map, best.bbox)
            
            # Place ad
            result, bin_mask, adj = engine.place(
                frame, 
                ad_image_bgr, 
                target_corners, 
                persons=persons, 
                depth_map=depth_map, 
                cfg=cfg
            )
            
            # Update corners for next frame temporal calc
            target_corners = adj

            # Calculate reward for this frame
            rc = reward_fn.compute(
                frame, result, bin_mask, target_surf_mask, 
                target_depth, persons, corners=adj
            )
            all_rewards.append(rc.to_dict())

            out.write(result)
            frame_idx += 1

    finally:
        cap.release()
        out.release()

    if not all_rewards:
        return out_path, "No frames processed."

    # Compute average scores
    avg_scores = {k: np.mean([r[k] for r in all_rewards]) for k in all_rewards[0].keys()}
    summary = f"✅ Processed {frame_idx} frames.\n\n"
    summary += "🏆 REWARD BREAKDOWN (Phase 3 Evaluation):\n"
    summary += f"• Overall Quality: {avg_scores['total']:.2f}/0.90\n"
    summary += f"• Realism: {avg_scores['realism']:.2f}  • Alignment: {avg_scores['alignment']:.2f}\n"
    summary += f"• Lighting: {avg_scores['lighting']:.2f} • Occlusion: {avg_scores['occlusion']:.2f}\n"
    summary += f"• Stability: {avg_scores['temporal']:.2f}"

    return out_path, summary

# UI Construction
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown(
        """
        # 🚀 AdVision AI - In-Content Ad Placement
        ### Phase 3: Human Evaluation Demo
        This interface allows you to evaluate the quality of our ad-placement agent. 
        Upload a video and a transparent ad (PNG) to see the results.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🛠️ Inputs")
                video_input = gr.Video(label="Base Scene (Video)")
                ad_input = gr.Image(label="Ad Banner/Bottle (Image)", type="numpy")
            
            with gr.Accordion("⚙️ Placement Fine-Tuning", open=True):
                scale_slider = gr.Slider(0.5, 2.5, value=1.4, step=0.1, label="Ad Scale")
                alpha_slider = gr.Slider(0.0, 1.0, value=0.97, step=0.01, label="Opacity (Alpha)")
                rotation_slider = gr.Slider(-180, 180, value=0, step=5, label="Rotation (Degrees)")
                tilt_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Perspective Tilt")
                shadow_slider = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Shadow Strength")
                feather_slider = gr.Slider(0, 50, value=22, step=1, label="Boundary Feathering")
            
            run_btn = gr.Button("✨ Process Ad Placement", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📺 Output")
            video_output = gr.Video(label="Augmented Scene")
            status_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Group():
                gr.Markdown("### 💡 Technology Insights")
                gr.Markdown(
                    """
                    - **YOLOv8 & MiDaS**: Real-time object detection and monocular depth estimation.
                    - **ORB Homography**: Advanced World-Lock tracking for temporal stability.
                    - **LAB Color Transfer**: Real-time color grading to match scene lighting.
                    - **Gaussian Feathering**: Eliminates hard edges for seamless blending.
                    """
                )

    gr.Examples(
        examples=[
            [
                os.path.join(ROOT_DIR, "data", "samples", "living_room.mp4") if os.path.exists(os.path.join(ROOT_DIR, "data", "samples", "living_room.mp4")) else None,
                os.path.join(ROOT_DIR, "data", "ad_images", "oil_ad.png"),
                1.4, 0, 0, 0.97, 22, 0.4
            ],
            [
                os.path.join(ROOT_DIR, "data", "samples", "office.mp4") if os.path.exists(os.path.join(ROOT_DIR, "data", "samples", "office.mp4")) else None,
                os.path.join(ROOT_DIR, "data", "ad_images", "sample_ad.png"),
                1.2, 0, 0.1, 0.95, 15, 0.3
            ],
        ],
        inputs=[video_input, ad_input, scale_slider, rotation_slider, tilt_slider, alpha_slider, feather_slider, shadow_slider]
    )

    run_btn.click(
        fn=process_video,
        inputs=[
            video_input, ad_input, scale_slider, rotation_slider, tilt_slider, 
            alpha_slider, feather_slider, shadow_slider
        ],
        outputs=[video_output, status_text]
    )

if __name__ == "__main__":
    demo.launch()
