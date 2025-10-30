import os
import sys
import logging

print("--- train.py STARTED ---")
print(f"Current CWD: {os.getcwd()}")
print(f"sys.path: {sys.path}")

CODE_PATH = '/opt/ml/code'
if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)
    print(f"Added {CODE_PATH} to sys.path")

import time
import hydra
from omegaconf import DictConfig

from src.pipeline.sagemaker_compat import adapt_to_sagemaker
from src.pipeline.data_pipeline import prepare_data
from src.pipeline.training_pipeline import train_and_evaluate_model

adapt_to_sagemaker()

log = logging.getLogger(__name__)
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conf"))

@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    log.info("🚀 STARTING DRL TRADING PIPELINE 🚀")

    log.info("\n--- [STAGE 1/2] Kicking off Data Preparation ---")
    df_train_s, df_val_s, df_test_s, feature_cols, scaler = prepare_data(cfg)
    log.info("--- ✅ [STAGE 1/2] Data Preparation COMPLETED ---")
    
    log.info("\n--- [STAGE 2/2] Kicking off Model Training & Evaluation ---")
    train_and_evaluate_model(
        cfg=cfg,
        df_train_s=df_train_s,
        df_val_s=df_val_s,
        df_test_s=df_test_s,
        feature_cols=feature_cols,
        scaler=scaler
    )
    log.info("--- ✅ [STAGE 2/2] Model Training & Evaluation COMPLETED ---")

    elapsed_time = time.time() - start_time
    log.info(f"\n🎉 PIPELINE FINISHED SUCCESSFULLY in {elapsed_time:.2f} seconds. 🎉")

if __name__ == "__main__":
    main()
