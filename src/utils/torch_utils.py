# src/utils/torch_utils.py
import torch
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def get_device(cfg: DictConfig) -> str:
    """
    Determines the device (CPU/CUDA) based on config and availability.
    """
    if hasattr(cfg, 'training') and hasattr(cfg.training, 'device'):
        requested_device = cfg.training.device
    else:
        requested_device = 'auto'
    
    if requested_device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = requested_device
    
    if device == 'cuda' and not torch.cuda.is_available():
        log.warning("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    return device
