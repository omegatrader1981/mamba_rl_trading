# src/features/feature_pipeline.py
# <<< REFACTORED: This is the main orchestrator for feature creation. >>>

import pandas as pd
import numpy as np
import logging
from omegaconf import DictConfig

# <<< Relative imports from our new, organized package >>>
from . import market_features, regime_features, time_features

log = logging.getLogger(__name__)

def create_feature_set(df: pd.DataFrame, cfg: DictConfig, df_train: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Orchestrates the creation of the full feature set by calling specialized functions.
    """
    log.info("--- Starting Feature Engineering Pipeline ---")
    df_feat = df.copy()
    p = cfg.features

    # --- Call specialized feature functions ---
    df_feat = market_features.calculate_hurst(df_feat, p.hurst_window)
    df_feat = market_features.calculate_vwap(df_feat)
    df_feat = market_features.calculate_risk_adjusted_momentum(df_feat, p.momentum_window, p.volatility_window)
    df_feat = market_features.calculate_basic_price_features(df_feat, p.pct_change_cap, p.hl_pct_cap, p.z_score_window)
    df_feat = regime_features.calculate_hmm_regime(df_feat, df_train, p.hmm_n_components, p.hmm_min_samples)
    df_feat = time_features.calculate_time_features(df_feat)
    
    # --- Define and clean final feature set ---
    feature_cols_list = [
        'hurst', 'session_vwap_dist', 'risk_adj_mom', 'price_change_pct',
        'high_low_pct', 'volume_zscore', 'hmm_regime', 'time_sin', 'time_cos'
    ]
    
    log.info("Performing final cleanup of all feature values (NaN/Inf)...")
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_feat[feature_cols_list] = df_feat[feature_cols_list].ffill()
    df_feat.dropna(subset=feature_cols_list, inplace=True)

    if df_feat.empty:
        raise ValueError("No data remaining after feature calculation and NaN drop.")

    log.info(f"Feature pipeline complete. Final shape: {df_feat.shape}.")
    
    # Return the DataFrame with all necessary columns and the list of feature names
    required_ohlcv = ['open', 'high', 'low', 'close', 'volume', 'day']
    final_cols = list(set(feature_cols_list + required_ohlcv))
    final_cols = [c for c in final_cols if c in df_feat.columns]
    
    return df_feat[sorted(final_cols)], sorted(feature_cols_list)