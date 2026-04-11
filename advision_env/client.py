from openenv.core import SyncEnvClient
from .models import AdVisionAction, AdVisionObservation

class AdVisionEnv(SyncEnvClient):
    action_type = AdVisionAction
    observation_type = AdVisionObservation
    
    @classmethod
    def from_docker_image(cls, image: str, **kwargs):
        return super().from_docker_image(image, **kwargs)
