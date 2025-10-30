# src/model/mamba_policy.py
# <<< NEW MODULE: Defines the ActorCriticPolicy that uses our Mamba extractor. >>>

from stable_baselines3.common.policies import ActorCriticPolicy
import logging

# <<< Relative import from our new package structure >>>
from .mamba_extractor import MambaFeaturesExtractor

log = logging.getLogger(__name__)

class MambaActorCriticPolicy(ActorCriticPolicy):
    """
    An Actor-Critic policy that uses the MambaFeaturesExtractor by default.
    This class acts as the "glue" to connect our custom Mamba network
    to the Stable Baselines 3 PPO/SAC algorithms.
    """
    def __init__(self, *args, **kwargs):
        # Ensure our custom extractor is used if none is specified
        if 'features_extractor_class' not in kwargs:
            kwargs['features_extractor_class'] = MambaFeaturesExtractor
        
        # Set a default MLP network architecture if not provided
        if 'net_arch' not in kwargs:
             kwargs['net_arch'] = dict(pi=[64, 64], vf=[64, 64]) 
             log.info(f"MambaActorCriticPolicy using default net_arch: {kwargs['net_arch']}")
        
        log.info("Initializing MambaActorCriticPolicy...")
        super().__init__(*args, **kwargs)
        log.info("MambaActorCriticPolicy initialized successfully.")