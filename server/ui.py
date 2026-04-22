import os
import sys

# Setup paths (at top to satisfy linter)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tempfile  # noqa: E402
import gradio as gr  # noqa: E402

from advision_env.models.vision_models import ObjectDetector, DepthEstimator  # noqa: E402
from advision_env.pipeline.placement_engine import PlacementEngine, PlacementConfig  # noqa: E402
from advision_env.env.reward import RewardFunction  # noqa: E402

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
        return None, "<div style='color: #ef4444; padding: 10px;'>⚠️ Please upload both a video and an ad image.</div>"

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
    max_process_frames = min(total_frames, 150) # Limit for demo

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
        return out_path, "<div style='color: #ef4444; padding: 10px;'>⚠️ No frames were processed. Please check input video.</div>"

    # Compute average scores
    avg_scores = {k: np.mean([r[k] for r in all_rewards]) for k in all_rewards[0].keys()}
    
    def get_bar(val, color="#3b82f6"):
        percent = int(val * 100)
        return f"""
        <div style='width: 100%; background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; margin-top: 4px;'>
            <div style='width: {percent}%; background: {color}; height: 100%; border-radius: 4px; box-shadow: 0 0 10px {color}88;'></div>
        </div>
        """

    summary_html = f"""
    <div style='color: #f8fafc;'>
        <h3 style='margin: 0; color: #60a5fa;'>🏆 Episode Performance</h3>
        <p style='font-size: 0.9rem; color: #94a3b8;'>Processed {frame_idx} frames successfully.</p>
        
        <div style='margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
            <div class='metric-card'>
                <span style='font-size: 0.8rem; color: #94a3b8;'>Overall Quality</span>
                <div style='font-size: 1.5rem; font-weight: 800; color: #3b82f6;'>{avg_scores['total']:.2f}</div>
                {get_bar(avg_scores['total'], "#3b82f6")}
            </div>
            <div class='metric-card'>
                <span style='font-size: 0.8rem; color: #94a3b8;'>Temporal Stability</span>
                <div style='font-size: 1.5rem; font-weight: 800; color: #10b981;'>{avg_scores['temporal']:.2f}</div>
                {get_bar(avg_scores['temporal'], "#10b981")}
            </div>
            <div class='metric-card'>
                <span style='font-size: 0.8rem; color: #94a3b8;'>Realism & Blend</span>
                <div style='font-size: 1.2rem; font-weight: 600;'>{avg_scores['realism']:.2f}</div>
                {get_bar(avg_scores['realism'], "#60a5fa")}
            </div>
            <div class='metric-card'>
                <span style='font-size: 0.8rem; color: #94a3b8;'>Spatial Alignment</span>
                <div style='font-size: 1.2rem; font-weight: 600;'>{avg_scores['alignment']:.2f}</div>
                {get_bar(avg_scores['alignment'], "#a855f7")}
            </div>
        </div>
        
        <div style='margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px; font-size: 0.85rem;'>
            <span style='color: #94a3b8;'>Detail:</span> Light: {avg_scores['lighting']:.2f} | Occ: {avg_scores['occlusion']:.2f}
        </div>
    </div>
    """

    return out_path, summary_html

# UI Construction
TITLE = "🎯 AdVision AI — Precision Ad Placement"
SUBTITLE = "Meta PyTorch OpenEnv Hackathon | Real-World Spatial AI"

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: radial-gradient(circle at 50% 0%, #1a1c2e 0%, #0d0e1a 100%) !important;
}

.glass-card {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}

.header-container {
    text-align: center;
    padding: 40px 0;
}

.header-title {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #60a5fa 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
}

.header-subtitle {
    color: #94a3b8 !important;
    font-size: 1.2rem !important;
    font-weight: 400 !important;
}

.mt-20 { margin-top: 20px !important; }
.mt-40 { margin-top: 40px !important; }

.metric-card {
    background: rgba(30, 41, 59, 0.5) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    border-left: 4px solid #3b82f6 !important;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    background: rgba(30, 41, 59, 0.8) !important;
}

footer { visibility: hidden; }

.gr-button-primary {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

.gr-button-primary:hover {
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.5) !important;
    transform: scale(1.02) !important;
}

.status-badge {
    background: rgba(16, 185, 129, 0.1) !important;
    color: #10b981 !important;
    padding: 4px 12px !important;
    border-radius: 9999px !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"), css=CUSTOM_CSS) as demo:
    with gr.Column(elem_classes="header-container"):
        gr.HTML(f"""
            <h1 class='header-title'>{TITLE}</h1>
            <p class='header-subtitle'>{SUBTITLE}</p>
            <div style='margin-top: 20px;'>
                <span class='status-badge'>Compliant with OpenEnv v1.0</span>
                <span class='status-badge' style='margin-left: 10px; border-color: rgba(96, 165, 250, 0.2); color: #60a5fa;'>Vision: YOLOv8 + MiDaS</span>
            </div>
        """)

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(elem_classes="glass-card"):
                gr.Markdown("### 📥 Source Media")
                with gr.Row():
                    video_input = gr.Video(label="Scene Video", elem_id="video-input")
                    ad_input = gr.Image(label="Ad Asset (PNG preferred)", type="numpy")
            
            with gr.Column(elem_classes=["glass-card", "mt-20"]):
                gr.Markdown("### ⚙️ Placement Precision Engine")
                with gr.Row():
                    with gr.Column():
                        scale_slider = gr.Slider(0.5, 2.5, value=1.4, step=0.1, label="🔍 Ad Scale")
                        alpha_slider = gr.Slider(0.0, 1.0, value=0.97, step=0.01, label="💧 Opacity (Alpha)")
                    with gr.Column():
                        rotation_slider = gr.Slider(-180, 180, value=0, step=5, label="🔄 Rotation")
                        tilt_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="📐 Perspective Tilt")
                
                with gr.Accordion("✨ Advanced Effects", open=False):
                    with gr.Row():
                        shadow_slider = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="🌑 Shadow Strength")
                        feather_slider = gr.Slider(0, 50, value=22, step=1, label="🌫️ Edge Feathering")

                run_btn = gr.Button("✨ Render Augmented Scene", variant="primary", size="lg")

        with gr.Column(scale=5):
            with gr.Column(elem_classes="glass-card"):
                gr.Markdown("### 📺 Production Output")
                video_output = gr.Video(label="Final Composite", interactive=False)
                
            with gr.Column(elem_classes=["glass-card", "mt-20"]):
                gr.Markdown("### 📊 Performance Analytics")
                status_text = gr.HTML(label="Agent Feedback & Rewards", elem_id="status-display")

    with gr.Row(elem_classes="mt-40"):
        with gr.Column():
            with gr.Column(elem_classes="metric-card"):
                gr.Markdown("#### 🟠 Spatial Reasoning\nPrecise 3D localization via monocular depth estimation.")
        with gr.Column():
            with gr.Column(elem_classes="metric-card"):
                gr.Markdown("#### 🟢 Temporal Stability\nORB-Homography tracking ensures zero pixel drift.")
        with gr.Column():
            with gr.Column(elem_classes="metric-card"):
                gr.Markdown("#### 🔵 Realistic Blending\nLAB color transfer matches ad to scene lighting.")

    # Fix Example Paths
    EXAMPLE_VIDEO = os.path.join(ROOT_DIR, "data", "input_videos", "test.mp4")
    if not os.path.exists(EXAMPLE_VIDEO):
        EXAMPLE_VIDEO = None
        
    gr.Examples(
        examples=[
            [
                EXAMPLE_VIDEO,
                os.path.join(ROOT_DIR, "data", "ad_images", "oil_ad.png"),
                1.4, 0, 0, 0.97, 22, 0.4
            ],
            [
                EXAMPLE_VIDEO,
                os.path.join(ROOT_DIR, "data", "ad_images", "sample_ad.png"),
                1.2, 0, 0.1, 0.95, 15, 0.3
            ],
        ],
        inputs=[video_input, ad_input, scale_slider, rotation_slider, tilt_slider, alpha_slider, feather_slider, shadow_slider],
        cache_examples=False
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
