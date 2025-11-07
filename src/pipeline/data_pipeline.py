# src/pipeline/data_pipeline.py
import pandas as pd
import logging
import os
from omegaconf import DictConfig
from typing import Tuple
from src.data import load_futures_data, build_regime_dataset
from src.features import create_feature_set, fit_and_transform_scaler
from src.pipeline.sagemaker_compat import is_running_in_sagemaker
# from src.features.microstructure import aggregate_1s_to_1min  # ← Phase 1.0

log = logging.getLogger(__name__)

def prepare_data(cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list, object]:
    log.info("--- Starting Data Preparation Pipeline ---")
   
    if is_running_in_sagemaker():
        base_data_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    else:
        base_data_dir = cfg.data.data_dir
    filepaths = [os.path.join(base_data_dir, f) for f in cfg.data.data_filenames]
    full_df = load_futures_data(filepaths, **cfg.data.cleaning)

    # <<< PHASE 0.5: 5-min OHLCV ONLY >>>
    if cfg.data.get("cadence", "5min") == "1s":
        raise ValueError("1s data not supported in Phase 0.5. Use 5min OHLCV.")
    else:
        log.info("Using 5-min OHLCV data (Phase 0.5)")

    # --- Regimes ---
    train_regimes = {k: v for k, v in cfg.data.regime_definitions.items() if k.startswith('train_')}
    df_train_raw = build_regime_dataset(full_df, train_regimes)
    val_dates = cfg.data.regime_definitions.validation_set[0]
    df_val_raw = full_df.loc[val_dates[0]:val_dates[1]].copy()
    df_val_raw['source_regime'] = 'validation_set'
    test_dates = cfg.data.regime_definitions.test_set[0]
    df_test_raw = full_df.loc[test_dates[0]:test_dates[1]].copy()
    df_test_raw['source_regime'] = 'test_set'

    # --- Feature pipeline ---
    df_train_for_fitting = df_train_raw.copy()
    df_train_feat, feature_cols = create_feature_set(df_train_raw, cfg, df_train_for_fitting)
    df_val_feat, _ = create_feature_set(df_val_raw, cfg, df_train_for_fitting)
    df_test_feat, _ = create_feature_set(df_test_raw, cfg, df_train_for_fitting)

    # --- Scaling ---
    dfs_to_scale = {"train": df_train_feat, "validation": df_val_feat, "test": df_test_feat}
    scaler_path = os.path.join(os.getcwd(), "scaler.joblib")
    scaled_dfs, scaler = fit_and_transform_scaler(df_train_feat, dfs_to_scale, feature_cols, scaler_path)

    log.info("--- Data Preparation Pipeline COMPLETED ---")
    return scaled_dfs["train"], scaled_dfs["validation"], scaled_dfs["test"], feature_cols, scaler
