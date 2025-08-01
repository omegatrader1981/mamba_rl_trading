# src/data/data_cleaning.py
# <<< NEW MODULE: DEDICATED TO DATA CLEANING >>>

import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

def clean_ohlcv_data(df: pd.DataFrame, 
                     min_allowable_price: float, 
                     outlier_rolling_window: int, 
                     outlier_median_dev_threshold: float
                    ) -> pd.DataFrame:
    """
    Applies a robust, multi-step cleaning pipeline to a raw OHLCV DataFrame.
    This function assumes the input df is already concatenated and has a timestamp index.
    """
    required_ohlc = ['open', 'high', 'low', 'close']
    
    initial_rows = len(df)
    log.info(f"--- Starting Data Cleaning Pipeline. Initial rows: {initial_rows} ---")

    log.info(f"1. Applying basic price validity (min_allowable_price: {min_allowable_price})...")
    prices_corrected_min_val = 0
    for col in required_ohlc:
        invalid_price_mask = (df[col].notna()) & (df[col] < min_allowable_price)
        if invalid_price_mask.any():
            count = invalid_price_mask.sum()
            prices_corrected_min_val += count
            log.warning(f"   Found {count} prices < {min_allowable_price} in '{col}'. Setting to NaN.")
            df.loc[invalid_price_mask, col] = np.nan
    if prices_corrected_min_val > 0:
        log.info(f"   Total {prices_corrected_min_val} prices set to NaN due to being below min_allowable_price.")

    log.info("2. Validating OHLC relationships...")
    ohlc_corrections = 0
    valid_hl_compare_mask = df['high'].notna() & df['low'].notna()
    mask_h_lt_l = valid_hl_compare_mask & (df['high'] < df['low'])
    if mask_h_lt_l.any():
        count = mask_h_lt_l.sum()
        ohlc_corrections += count
        log.warning(f"   Found {count} instances where high < low. Swapping them.")
        df.loc[mask_h_lt_l, ['high', 'low']] = df.loc[mask_h_lt_l, ['low', 'high']].values
    
    for oc_col in ['open', 'close']:
        mask_oc_gt_h = df[oc_col].notna() & df['high'].notna() & (df[oc_col] > df['high'])
        if mask_oc_gt_h.any():
            count = mask_oc_gt_h.sum()
            ohlc_corrections += count
            log.warning(f"   Found {count} instances where '{oc_col}' > 'high'. Adjusting 'high'.")
            df.loc[mask_oc_gt_h, 'high'] = df.loc[mask_oc_gt_h, oc_col]

        mask_oc_lt_l = df[oc_col].notna() & df['low'].notna() & (df[oc_col] < df['low'])
        if mask_oc_lt_l.any():
            count = mask_oc_lt_l.sum()
            ohlc_corrections += count
            log.warning(f"   Found {count} instances where '{oc_col}' < 'low'. Adjusting 'low'.")
            df.loc[mask_oc_lt_l, 'low'] = df.loc[mask_oc_lt_l, oc_col]
    if ohlc_corrections > 0:
        log.info(f"   Total {ohlc_corrections} OHLC relationship corrections applied.")

    log.info("3. Validating volume data...")
    volume_corrections = 0
    if 'volume' in df.columns:
        invalid_volume_mask = df['volume'].notna() & (df['volume'] < 0)
        if invalid_volume_mask.any():
            count = invalid_volume_mask.sum()
            volume_corrections += count
            log.warning(f"   Found {count} negative volumes. Setting to 0.")
            df.loc[invalid_volume_mask, 'volume'] = 0
        
        if df['volume'].isnull().any():
            nan_vol_count = df['volume'].isnull().sum()
            log.info(f"   Found {nan_vol_count} NaN volumes. Setting to 0.")
            df['volume'].fillna(0, inplace=True)
            volume_corrections += nan_vol_count
    if volume_corrections > 0:
        log.info(f"   Total {volume_corrections} volume corrections/fills applied.")

    log.info(f"4. Detecting statistical price outliers (window: {outlier_rolling_window}, threshold: {outlier_median_dev_threshold * 100}%)...")
    price_outliers_naned = 0
    if len(df) >= outlier_rolling_window:
        rolling_median_close = df['close'].rolling(window=outlier_rolling_window, center=True, min_periods=max(1, outlier_rolling_window//2)).median()
        upper_bound = rolling_median_close * (1 + outlier_median_dev_threshold)
        lower_bound = rolling_median_close * (1 - outlier_median_dev_threshold)
        outlier_mask_close = ((df['close'] > upper_bound) | (df['close'] < lower_bound)) & rolling_median_close.notna() & (rolling_median_close >= min_allowable_price)
        if outlier_mask_close.any():
            count = outlier_mask_close.sum()
            price_outliers_naned += count
            log.warning(f"   Found {count} statistical outliers in 'close' price. Setting these and corresponding OHL to NaN.")
            df.loc[outlier_mask_close, required_ohlc] = np.nan
    if price_outliers_naned > 0:
         log.info(f"   Total {price_outliers_naned} price outliers (and related OHL) set to NaN.")

    log.info("5. Filling missing (NaN) values using ffill then bfill for prices...")
    nan_counts_before_fill = df[required_ohlc].isnull().sum()
    df[required_ohlc] = df[required_ohlc].fillna(method='ffill').fillna(method='bfill')
    nan_counts_after_fill = df[required_ohlc].isnull().sum()
    for col in required_ohlc:
        if (filled_count := nan_counts_before_fill.get(col, 0) - nan_counts_after_fill.get(col, 0)) > 0:
            log.info(f"   Filled {filled_count} NaNs in '{col}'.")

    log.info("6. Dropping any rows with persistent NaNs in critical OHLC columns...")
    rows_before_final_drop = len(df)
    df.dropna(subset=required_ohlc, inplace=True)
    if (dropped_count := rows_before_final_drop - len(df)) > 0:
        log.warning(f"   Dropped {dropped_count} rows with persistent NaNs in OHLC columns.")

    log.info(f"--- Data Cleaning Pipeline Finished. Rows after: {len(df)} ---")
    return df