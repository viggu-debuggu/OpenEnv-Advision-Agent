from openenv.core import EnvClient
try:
    from .models import AdVisionAction, AdVisionObservation, AdVisionState
except ImportError:
    from models import AdVisionAction, AdVisionObservation, AdVisionState

class AdVisionEnv(EnvClient):
    """
    Typed Synchronous client for the AdVision environment.
    Follows the official OpenEnv pattern.
    """
    action_type = AdVisionAction
    observation_type = AdVisionObservation
    state_type = AdVisionState
