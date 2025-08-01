# src/features/market_features.py
# <<< FINAL CORRECTED VERSION: Replaced np.polyfit with Numba-compatible manual regression. >>>

import pandas as pd
import numpy as np
import numba
import logging

log = logging.getLogger(__name__)

# --- Hurst Exponent (Numba Optimized) ---
@numba.jit(nopython=True, cache=True)
def _get_hurst_fast(series_np: np.ndarray, lags: np.ndarray) -> float:
    """
    Numba-optimized function to calculate a single Hurst Exponent value.
    Uses a Numba-compatible manual linear regression for maximum speed.
    """
    if len(series_np) < lags[-1]: return np.nan
    
    tau = np.zeros(len(lags), dtype=np.float64)
    for i, lag in enumerate(lags):
        diff = series_np[lag:] - series_np[:-lag]
        if len(diff) > 1:
            tau[i] = np.std(diff)
        else:
            tau[i] = 0.0

    valid_mask = tau > 1e-9
    if np.sum(valid_mask) < 2: return np.nan
    
    log_lags = np.log(lags[valid_mask])
    log_tau = np.log(tau[valid_mask])
    
    # <<< THE FIX IS HERE: Replaced np.polyfit with Numba-compatible manual regression >>>
    n = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_tau)
    sum_xy = np.sum(log_lags * log_tau)
    sum_xx = np.sum(log_lags * log_lags)

    denominator = (n * sum_xx - sum_x * sum_x)
    if np.abs(denominator) < 1e-9: # Avoid division by zero
        return np.nan

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope

def calculate_hurst(df: pd.DataFrame, window: int) -> pd.DataFrame:
    log.info(f"Calculating Hurst Exponent (window: {window})...")
    lags = np.arange(2, window + 1)
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    
    hurst_values = log_returns.rolling(window=window * 2, min_periods=window * 2).apply(
        lambda x: _get_hurst_fast(x, lags), raw=True
    )
    
    df['hurst'] = hurst_values.reindex(df.index).ffill()
    return df

# --- Session-Aware VWAP ---
def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Calculating Session-Aware VWAP...")
    df_vwap = df.copy()
    idx_eastern = df_vwap.index.tz_convert('US/Eastern')
    
    rth_start, rth_end = pd.Timestamp('09:30').time(), pd.Timestamp('16:00').time()
    is_rth = (idx_eastern.time >= rth_start) & (idx_eastern.time < rth_end)
    session_id = (is_rth != pd.Series(is_rth, index=df_vwap.index).shift(1)).cumsum()
    
    df_vwap['session_group'] = pd.Series(idx_eastern.date, index=df_vwap.index).astype(str) + '_' + session_id.astype(str)
    df_vwap['price_vol'] = ((df_vwap['high'] + df_vwap['low'] + df_vwap['close']) / 3) * df_vwap['volume']
    
    grouped = df_vwap.groupby('session_group')
    vwap_series = (grouped['price_vol'].cumsum() / grouped['volume'].cumsum().replace(0, np.nan)).ffill()
    
    df['session_vwap'] = vwap_series
    df['session_vwap_dist'] = (df['close'] - df['session_vwap']) / df['session_vwap']
    return df

# --- Other Market Features ---
def calculate_risk_adjusted_momentum(df: pd.DataFrame, mom_window: int, vol_window: int) -> pd.DataFrame:
    log.info(f"Calculating Risk-Adjusted Momentum (mom: {mom_window}, vol: {vol_window})...")
    momentum = df['close'].pct_change(periods=mom_window)
    volatility = df['close'].pct_change().rolling(vol_window).std()
    df['risk_adj_mom'] = momentum / volatility.replace(0, 1e-9)
    return df

def calculate_basic_price_features(df: pd.DataFrame, pct_cap: float, hl_cap: float, z_window: int) -> pd.DataFrame:
    log.info("Calculating basic price action features...")
    df['price_change_pct'] = np.clip(df['close'].pct_change(), -pct_cap, pct_cap)
    df['high_low_pct'] = np.clip((df['high'] - df['low']) / df['close'].replace(0, np.nan), 0, hl_cap)
    
    rolling_mean = df['volume'].rolling(window=z_window).mean()
    rolling_std = df['volume'].rolling(window=z_window).std().replace(0, 1e-9)
    df['volume_zscore'] = (df['volume'] - rolling_mean) / rolling_std
    return df