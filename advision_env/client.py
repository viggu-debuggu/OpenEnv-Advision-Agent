try:
    from openenv.core.env_client import SyncEnvClient
except ImportError:
    from openenv.core.env_client import HTTPEnvClient as SyncEnvClient

try:
    from .models import AdVisionAction, AdVisionObservation, AdVisionState
except ImportError:
    from models import AdVisionAction, AdVisionObservation, AdVisionState

class AdVisionEnv(SyncEnvClient):
    action_type = AdVisionAction
    observation_type = AdVisionObservation
    state_type = AdVisionState
