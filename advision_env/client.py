from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import AdVisionAction, AdVisionObservation, AdVisionState
except ImportError:
    from models import AdVisionAction, AdVisionObservation, AdVisionState

class AdVisionEnv(EnvClient[AdVisionAction, AdVisionObservation, AdVisionState]):
    """
    Typed Synchronous client for the AdVision environment.
    Follows the official OpenEnv pattern.
    """
    action_type = AdVisionAction
    observation_type = AdVisionObservation
    state_type = AdVisionState
    
    def _step_payload(self, action: Any) -> Dict[str, Any]:
        if isinstance(action, dict):
            return action
        if hasattr(action, 'model_dump'):
            return action.model_dump()
        return dict(action)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AdVisionObservation]:
        obs_data = payload.get("observation", {})
        if isinstance(obs_data, dict):
            obs = AdVisionObservation(**obs_data)
        else:
            obs = obs_data
        
        result = StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )
        # Attach info dictionary so the grading logic works
        # Extracts info from obs.info if the OpenEnv core wrapper stripped it from the root payload
        result.info = payload.get("info") or getattr(obs, "info", {})
        return result

    def _parse_state(self, payload: Dict[str, Any]) -> AdVisionState:
        if isinstance(payload, dict):
            return AdVisionState(**payload)
        return payload
