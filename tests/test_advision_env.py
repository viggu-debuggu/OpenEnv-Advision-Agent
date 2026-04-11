import pytest
import numpy as np
import os
import sys

# Add root to sys.path to find server and models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.advision_environment import AdVisionEnvironment
from advision_env.models import AdVisionAction

def test_reset():
    env = AdVisionEnvironment()
    obs = env.reset()
    assert obs is not None
    assert hasattr(obs, 'detected_surfaces')
    assert obs.reward == 0.0

def test_step():
    env = AdVisionEnvironment()
    env.reset()
    action = AdVisionAction(
        x_position=0.0, y_position=0.0,
        scale=1.0, rotation=0.0, tilt=0.0,
        ad_selection=0.0, alpha=0.97
    )
    obs = env.step(action)
    assert obs.reward is not None
    assert isinstance(obs.reward, float)
    assert obs.done is not None

def test_state():
    env = AdVisionEnvironment()
    env.reset()
    s = env.state()
    assert s.video_path != ""
    assert s.max_frames == 30
    assert hasattr(s, "detected_surfaces_count")
