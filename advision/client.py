from openenv.core import SyncEnvClient
try:
    from .models import AdVisionAction, AdVisionObservation, AdVisionState
except ImportError:
    from models import AdVisionAction, AdVisionObservation, AdVisionState

class AdVisionEnv(SyncEnvClient):
    """
    Typed Synchronous client for the AdVision environment.
    Follows the official OpenEnv pattern.
    """
    action_type = AdVisionAction
    observation_type = AdVisionObservation
    state_type = AdVisionState
