# src/features/feature_pipeline.py
import pandas as pd
import numpy as np
import logging
from omegaconf import DictConfig
from . import market_features, regime_features, time_features
log = logging.getLogger(__name__)

def create_feature_set(df: pd.DataFrame, cfg: DictConfig, df_train: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    log.info("--- Starting Feature Engineering Pipeline ---")
    df_feat = df.copy()
    p = cfg.features

    df_feat = market_features.calculate_vwap(df_feat)
    df_feat = market_features.calculate_basic_price_features(df_feat, p.pct_change_cap, p.hl_pct_cap, p.z_score_window)
    df_feat = regime_features.calculate_hmm_regime(df_feat, df_train, p.hmm_n_components, p.hmm_min_samples)
    df_feat = time_features.calculate_time_features(df_feat)

    # <<< PHASE 0.5: CLASSIC FEATURES ONLY >>>
    feature_cols_list = [
        'volume_zscore', 'high_low_pct', 'session_vwap_dist',
        'hmm_regime', 'time_sin', 'time_cos'
    ]

    log.info("Cleaning NaN/Inf...")
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_feat[feature_cols_list] = df_feat[feature_cols_list].ffill()
    df_feat.dropna(subset=feature_cols_list, inplace=True)

    if df_feat.empty:
        raise ValueError("No data after feature cleanup.")

    required_cols = ['open', 'high', 'low', 'close', 'volume', 'day']
    if 'source_regime' in df_feat.columns:
        required_cols.append('source_regime')

    final_cols = [c for c in set(feature_cols_list + required_cols) if c in df_feat.columns]
    return df_feat[sorted(final_cols)], sorted(feature_cols_list)
