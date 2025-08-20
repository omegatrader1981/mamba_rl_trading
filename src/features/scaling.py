# <<< CORRECTED: Now uses the correct, nested import path for utils. >>>

import pandas as pd
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler

# <<< THE FIX IS HERE: Changed from 'src.utils' to 'src.utils.utils' >>>
from src.utils.utils import ensure_dir

log = logging.getLogger(__name__)

def fit_and_transform_scaler(
    df_train: pd.DataFrame, dfs_to_transform: dict, feature_cols: list, scaler_path: str
) -> tuple[dict, StandardScaler]:
    """
    Fits a StandardScaler on the training data and transforms a dictionary of DataFrames.
    """
    log.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    cols_to_scale = [c for c in feature_cols if c not in ['hmm_regime', 'time_sin', 'time_cos']]
    
    train_cols_to_scale = [col for col in cols_to_scale if col in df_train.columns]
    if not train_cols_to_scale:
        log.warning("No columns to scale in the training data. Skipping scaling.")
        return dfs_to_transform, scaler

    scaler.fit(df_train[train_cols_to_scale])
    
    scaled_dfs = {}
    for name, df in dfs_to_transform.items():
        if df.empty:
            scaled_dfs[name] = df
            continue
            
        df_scaled = df.copy()
        current_cols_to_scale = [col for col in train_cols_to_scale if col in df_scaled.columns]
        if current_cols_to_scale:
            df_scaled[current_cols_to_scale] = scaler.transform(df_scaled[current_cols_to_scale])
            log.info(f"'{name}' dataset scaled.")
        else:
            log.warning(f"No columns to scale were found in the '{name}' dataset.")

        scaled_dfs[name] = df_scaled
        
    if scaler_path:
        ensure_dir(os.path.dirname(scaler_path))
        joblib.dump(scaler, scaler_path)
        log.info(f"Scaler saved to {scaler_path}")
        
    return scaled_dfs, scaler
