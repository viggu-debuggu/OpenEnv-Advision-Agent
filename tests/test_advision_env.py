import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server.advision_environment import AdVisionEnvironment

try:
    from advision_env.models import AdVisionAction
except ImportError:
    from models import AdVisionAction

def test_reset():
    env = AdVisionEnvironment()
    obs = env.reset()
    assert obs is not None
    assert hasattr(obs, 'reward')

def test_step():
    env = AdVisionEnvironment()
    env.reset()
    action = AdVisionAction(
        x_position=0.0, y_position=0.0, scale=1.0,
        rotation=0.0, tilt=0.0, ad_selection=0.0, alpha=0.97
    )
    obs = env.step(action)
    assert obs.reward >= 0.0

def test_state():
    env = AdVisionEnvironment()
    env.reset()
    s = env.state           # must be property, not method call
    assert hasattr(s, 'step_count')
    assert hasattr(s, 'episode_id')
