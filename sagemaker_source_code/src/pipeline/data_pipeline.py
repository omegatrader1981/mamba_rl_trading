# <<< DEFINITIVE FINAL FIX: Ensures all datasets (train, val, test) have a 'source_regime' column. >>>

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
    log.info("--- Starting Data Preparation Pipeline (Final Corrected Version) ---")
    
    if is_running_in_sagemaker():
        base_data_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
        log.info(f"SageMaker environment detected. Using data path: {base_data_dir}")
    else:
        base_data_dir = cfg.data.data_dir
        log.info(f"Local environment detected. Using data path: {base_data_dir}")

    filepaths = [os.path.join(base_data_dir, fname) for fname in cfg.data.data_filenames]
    
    full_df = load_futures_data(filepaths, **cfg.data.cleaning)

    # --- Training Data ---
    train_regimes = {k: v for k, v in cfg.data.regime_definitions.items() if k.startswith('train_')}
    df_train_raw = build_regime_dataset(full_df, train_regimes)
    
    # --- Validation Data ---
    val_dates = cfg.data.regime_definitions.validation_set[0]
    df_val_raw = full_df.loc[val_dates[0]:val_dates[1]].copy()
    # <<< THE FIX IS HERE: Explicitly assign a source_regime >>>
    df_val_raw['source_regime'] = 'validation_set'
    
    # --- Test Data ---
    test_dates = cfg.data.regime_definitions.test_set[0]
    df_test_raw = full_df.loc[test_dates[0]:test_dates[1]].copy()
    # <<< THE FIX IS HERE: Explicitly assign a source_regime >>>
    df_test_raw['source_regime'] = 'test_set'

    log.info("All raw datasets (train, val, test) now have a 'source_regime' column.")

    # --- Feature Creation ---
    # The training data for fitting HMM and scalers remains the original spliced training set.
    df_train_for_fitting = df_train_raw.copy()

    df_train_feat, feature_cols = create_feature_set(df_train_raw, cfg, df_train_for_fitting)
    df_val_feat, _ = create_feature_set(df_val_raw, cfg, df_train_for_fitting)
    df_test_feat, _ = create_feature_set(df_test_raw, cfg, df_train_for_fitting)

    # --- Scaling ---
    dfs_to_scale = {"train": df_train_feat, "validation": df_val_feat, "test": df_test_feat}
    
    output_dir = os.getcwd()
    scaler_filename = "scaler.joblib"
    scaler_path = os.path.join(output_dir, scaler_filename)
    
    scaled_dfs, scaler = fit_and_transform_scaler(df_train_feat, dfs_to_scale, feature_cols, scaler_path)

    log.info("--- Data Preparation Pipeline COMPLETED ---")
    return scaled_dfs["train"], scaled_dfs["validation"], scaled_dfs["test"], feature_cols, scaler
