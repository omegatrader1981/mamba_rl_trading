# src/train.py
# <<< REFACTORED: This is now a pure entry point and orchestrator. >>>

import logging
import time
import hydra
from omegaconf import DictConfig
import os

# <<< Import from our new, organized package structure >>>
from src.pipeline.sagemaker_compat import adapt_to_sagemaker
from src.pipeline.data_pipeline import prepare_data
from src.pipeline.training_pipeline import train_and_evaluate_model

# Adapt to SageMaker before Hydra initializes
adapt_to_sagemaker()

log = logging.getLogger(__name__)
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conf"))

@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    The main entry point for the DRL trading pipeline.
    Orchestrates data preparation and model training stages.
    """
    start_time = time.time()
    log.info("🚀 STARTING PROJECT PHOENIX - DRL TRADING PIPELINE 🚀")

    # --- STAGE 1: DATA PREPARATION ---
    data_artifacts = prepare_data(cfg)
    
    # --- STAGE 2: MODEL TRAINING & EVALUATION ---
    train_and_evaluate_model(cfg, *data_artifacts)

    elapsed_time = time.time() - start_time
    log.info(f"✅ PIPELINE FINISHED SUCCESSFULLY in {elapsed_time:.2f} seconds. ✅")

if __name__ == "__main__":
    main()