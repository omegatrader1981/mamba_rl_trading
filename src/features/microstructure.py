# src/features/microstructure.py
import pandas as pd
import numpy as np
import logging
log = logging.getLogger(__name__)

def calculate_ofi(df_1s: pd.DataFrame, window: int = 5) -> pd.Series:
    """Order-Flow Imbalance (Cont 2014)."""
    delta_bid = df_1s['bid_size'].diff()
    delta_ask = df_1s['ask_size'].diff()
    ofi = (delta_bid > 0).astype(int) * delta_bid - (delta_ask < 0).astype(int) * (-delta_ask)
    return ofi.rolling(window).sum()

def calculate_microprice(df_1s: pd.DataFrame) -> pd.Series:
    """Volume-weighted mid."""
    return (df_1s['bid_price'] * df_1s['ask_size'] + df_1s['ask_price'] * df_1s['bid_size']) /            (df_1s['bid_size'] + df_1s['ask_size'])

def aggregate_1s_to_1min(df_1s: pd.DataFrame) -> pd.DataFrame:
    """1-second L2 → 1-minute OHLCV + OFI + micro-price."""
    log.info("Aggregating 1s L2 → 1min OHLCV + OFI + microprice")
    df = df_1s.copy()
    df['ofi'] = calculate_ofi(df)
    df['microprice'] = calculate_microprice(df)
    ohlc = df['close'].resample('1T').ohlc()
    volume = df['volume'].resample('1T').sum()
    ofi_agg = df['ofi'].resample('1T').agg(['mean', 'std', 'skew'])
    microprice = df['microprice'].resample('1T').last()
    agg = pd.concat([ohlc, volume, ofi_agg, microprice], axis=1)
    agg.columns = [
        'open', 'high', 'low', 'close', 'volume',
        'ofi_mean', 'ofi_std', 'ofi_skew', 'microprice'
    ]
    agg['microprice_return'] = agg['microprice'].pct_change()
    agg['microprice_dev'] = (agg['close'] - agg['microprice']) / agg['microprice']
    return agg.dropna()
