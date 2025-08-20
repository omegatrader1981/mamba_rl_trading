# <<< NEW MODULE: A comprehensive toolkit for robust SageMaker execution. >>>

import os
import logging
import functools
from typing import Callable, Any
from omegaconf import DictConfig, OmegaConf
import torch
import psutil
import pandas as pd

log = logging.getLogger(__name__)

def sagemaker_safe(func: Callable) -> Callable:
    """Decorator for SageMaker-safe function execution with comprehensive error handling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            log.info(f"--- Entering SageMaker-safe function: {func.__name__} ---")
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            log.error(f"FATAL ERROR in {func.__name__}: File or directory not found. This is likely a misconfigured path or a missing SageMaker data channel.", exc_info=True)
            raise
        except PermissionError as e:
            log.error(f"FATAL ERROR in {func.__name__}: Permission denied. Check the IAM role permissions for S3 and other services.", exc_info=True)
            raise
        except OSError as e:
            if e.errno == 28:
                log.error("FATAL ERROR: No space left on device. The instance storage is full. Increase the volume size for the SageMaker job.", exc_info=True)
            else:
                log.error(f"FATAL OS ERROR in {func.__name__}: {e}", exc_info=True)
            raise
        except Exception as e:
            log.error(f"An unexpected FATAL ERROR occurred in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper

def check_resources():
    """Logs the current system and GPU resource utilization."""
    log.info("--- Checking System Resources ---")
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/opt/ml')
    log.info(f"Memory: {memory.percent:.1f}% used ({memory.available / 1024**3:.2f} GB free)")
    log.info(f"Disk (/opt/ml): {disk.percent:.1f}% used ({disk.free / 1024**3:.2f} GB free)")
    if memory.percent > 85:
        log.warning("High memory usage detected (>85%).")
    if disk.percent > 90:
        log.warning("Low disk space detected (>90%).")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_percent = (gpu_allocated / gpu_memory * 100) if gpu_memory > 0 else 0
        log.info(f"GPU Memory: {gpu_percent:.1f}% allocated ({gpu_allocated / 1024**3:.2f} GB / {gpu_memory / 1024**3:.2f} GB)")
    else:
        log.info("No GPU detected.")
    log.info("--- Resource Check Complete ---")

def initialize_sagemaker_environment(cfg: DictConfig) -> DictConfig:
    """
    A master function to validate and prepare the entire SageMaker environment.
    """
    log.info("--- Initializing and Validating SageMaker Environment ---")
    
    required_keys = [
        'saving.optuna_db_name', 'checkpointing.s3_base_path', 'saving.output_dir',
        'data.data_dir', 'instrument.symbol', 'experiment.name'
    ]
    missing_keys = []
    for key in required_keys:
        try:
            OmegaConf.select(cfg, key)
        except Exception:
            missing_keys.append(key)
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    log.info("✓ Configuration schema validated.")

    paths_to_create = [
        os.path.dirname(cfg.saving.optuna_db_name),
        cfg.checkpointing.s3_base_path,
        cfg.saving.output_dir,
        "/opt/ml/model",
        "/opt/ml/checkpoints"
    ]
    for path in paths_to_create:
        if path:
            os.makedirs(path, exist_ok=True)
            log.info(f"✓ Directory ensured: {path}")

    data_dir = cfg.data.data_dir
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        available_dirs = []
        base_dir = '/opt/ml/input/data'
        if os.path.exists(base_dir):
            available_dirs = os.listdir(base_dir)
        raise FileNotFoundError(
            f"Data directory '{data_dir}' is missing or empty.\n"
            f"Available SageMaker channels: {available_dirs}\n"
            f"Check your SageMaker training job data channel configuration."
        )
    log.info(f"✓ Data directory validated: {data_dir}")

    check_resources()
    
    log.info("--- SageMaker Environment Initialized Successfully ---")
    return cfg
