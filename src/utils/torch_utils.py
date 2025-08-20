# src/torch_utils.py
# <<< NEW MODULE: For PyTorch-specific utility functions. >>>

import torch
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def get_device(cfg: DictConfig) -> str:
    """Selects the appropriate compute device based on config and availability."""
    requested_device = cfg.get("device", "auto")
    device = "cpu"

    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"'auto' device selected: {device}")
    elif requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            log.warning(f"CUDA '{requested_device}' requested, but not available. Falling back to CPU.")
            device = "cpu"
        else:
            if ":" in requested_device:
                try:
                    idx = int(requested_device.split(":")[1])
                    if idx >= torch.cuda.device_count():
                         log.warning(f"Invalid CUDA index {idx}. Defaulting to cuda:0.")
                         device = "cuda:0"
                    else:
                         device = requested_device
                except (ValueError, IndexError):
                    log.warning(f"Invalid CUDA format '{requested_device}'. Defaulting to cuda:0.")
                    device = "cuda:0"
            else:
                device = "cuda:0"
            log.info(f"CUDA device selected: {device}")
    elif requested_device == "cpu":
        device = "cpu"
        log.info(f"CPU device explicitly selected.")
    else:
        log.warning(f"Unrecognized device '{requested_device}'. Falling back to CPU.")
        device = "cpu"

    log.info(f"Final compute device determined: {device}")
    return device