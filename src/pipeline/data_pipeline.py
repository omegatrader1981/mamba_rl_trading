# src/pipeline/data_pipeline.py
# <<< FINAL CORRECTED VERSION: Decouples scaler path from config interpolation. >>>

import pandas as pd
import logging
import os
from omegaconf import DictConfig
from typing import Tuple

from src.data import load_futures_data, build_regime_dataset
from src.features import create_feature_set, fit_and_transform_scaler
from src.pipeline.sagemaker_compat import is_running_in_sagemaker

log = logging.getLogger(__name__)

def prepare_data(cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list, object]:
    """
    Executes the full data preparation pipeline with environment-aware pathing.
    """
    log.info("--- Starting Data Preparation Pipeline ---")
    
    if is_running_in_sagemaker():
        base_data_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
        log.info(f"SageMaker environment detected. Using data path: {base_data_dir}")
    else:
        base_data_dir = cfg.data.data_dir
        log.info(f"Local environment detected. Using data path: {base_data_dir}")

    filepaths = [os.path.join(base_data_dir, fname) for fname in cfg.data.data_filenames]
    
    full_df = load_futures_data(filepaths, **cfg.data.cleaning)

    train_regimes = {k: v for k, v in cfg.data.regime_definitions.items() if k.startswith('train_')}
    df_train_raw = build_regime_dataset(full_df, train_regimes)
    
    val_dates = cfg.data.regime_definitions.validation_set[0]
    df_val_raw = full_df.loc[val_dates[0]:val_dates[1]]
    
    test_dates = cfg.data.regime_definitions.test_set[0]
    df_test_raw = full_df.loc[test_dates[0]:test_dates[1]]

    df_train_feat, feature_cols = create_feature_set(df_train_raw, cfg, df_train_raw)
    df_val_feat, _ = create_feature_set(df_val_raw, cfg, df_train_raw)
    df_test_feat, _ = create_feature_set(df_test_raw, cfg, df_train_raw)

    dfs_to_scale = {"train": df_train_feat, "validation": df_val_feat, "test": df_test_feat}
    
    # <<< THE FIX IS HERE: Hardcode the scaler filename to remove the dependency. >>>
    # Hydra ensures that os.getcwd() points to the correct, unique output directory
    # for this run, both locally and on SageMaker. We create a simple, reliable
    # path for this intermediate artifact.
    output_dir = os.getcwd()
    scaler_filename = "scaler.joblib" # Simple, robust filename
    scaler_path = os.path.join(output_dir, scaler_filename)
    log.info(f"Scaler will be saved to a simple, robust path: {scaler_path}")
    
    scaled_dfs, scaler = fit_and_transform_scaler(df_train_feat, dfs_to_scale, feature_cols, scaler_path)

    log.info("--- Data Preparation Pipeline COMPLETED ---")
    return scaled_dfs["train"], scaled_dfs["validation"], scaled_dfs["test"], feature_cols, scaler