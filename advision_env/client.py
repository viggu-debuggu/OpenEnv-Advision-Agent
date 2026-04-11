from openenv.core.env_client import HTTPEnvClient
from .models import AdVisionAction, AdVisionObservation

class AdVisionEnv(HTTPEnvClient):
    action_type = AdVisionAction
    observation_type = AdVisionObservation
    
    @classmethod
    def from_docker_image(cls, image: str, **kwargs):
        return super().from_docker_image(image, **kwargs)
