# src/data/data_loader.py
# <<< REFACTORED: This module now orchestrates data loading and delegates cleaning. >>>

import pandas as pd
import os
import logging
from typing import List, Union, Dict

# <<< Relative import from our new cleaning module >>>
from .data_cleaning import clean_ohlcv_data

log = logging.getLogger(__name__)

def load_futures_data(filepaths: Union[str, List[str]], start_date: str = None, end_date: str = None,
                      # Pass cleaning parameters down
                      min_allowable_price: float = 0.01,
                      outlier_rolling_window: int = 21,
                      outlier_median_dev_threshold: float = 0.20
                     ) -> pd.DataFrame:
    """
    Orchestrates loading futures data: reads from CSVs, concatenates, cleans,
    and filters by date. Delegates cleaning to the clean_ohlcv_data function.
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    all_dfs = []
    log.info(f"Attempting to load futures data from {len(filepaths)} file(s)...")
    for filepath_item in filepaths:
        if not os.path.exists(filepath_item):
            raise FileNotFoundError(f"Data file not found at {filepath_item}")
        try:
            df_single = pd.read_csv(filepath_item, index_col='timestamp', parse_dates=True)
            if not df_single.empty:
                all_dfs.append(df_single)
            else:
                log.warning(f"File {filepath_item} is empty. Skipping.")
        except Exception as e:
            log.exception(f"Error reading CSV file {filepath_item}: {e}")
            raise

    if not all_dfs:
        raise ValueError("No data loaded from the provided filepaths.")

    df = pd.concat(all_dfs).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    log.info(f"Concatenated all files. Total unique rows: {len(df)}")

    df.columns = df.columns.str.lower()
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # <<< DELEGATE CLEANING >>>
    df = clean_ohlcv_data(
        df,
        min_allowable_price=min_allowable_price,
        outlier_rolling_window=outlier_rolling_window,
        outlier_median_dev_threshold=outlier_median_dev_threshold
    )

    if df.empty:
        raise ValueError("DataFrame is empty after cleaning steps. Cannot proceed.")

    # --- Final Timezone Handling and Filtering ---
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    elif str(df.index.tz).upper() != 'UTC':
        df = df.tz_convert('UTC')

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date, utc=True)]

    log.info(f"Data loading complete. Shape: {df.shape}, from {df.index.min()} to {df.index.max()}")
    
    final_cols = [col for col in required_cols if col in df.columns]
    return df[final_cols]


def build_regime_dataset(full_df: pd.DataFrame, regime_definitions: Dict[str, list]) -> pd.DataFrame:
    """
    Constructs a non-contiguous training dataset by slicing and concatenating
    data from specified market regimes.
    """
    log.info(f"--- Building 'Greatest Hits' dataset from {len(regime_definitions)} regimes ---")
    if not isinstance(full_df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")

    all_regime_segments = []
    for regime_name, date_pairs in regime_definitions.items():
        for start_date, end_date in date_pairs:
            try:
                segment = full_df.loc[start_date:end_date].copy()
                if segment.empty:
                    log.warning(f"Regime '{regime_name}' slice [{start_date} to {end_date}] is empty. Skipping.")
                    continue
                segment['source_regime'] = regime_name
                all_regime_segments.append(segment)
                log.info(f"  + Added segment for '{regime_name}': {len(segment)} rows from {start_date} to {end_date}")
            except Exception as e:
                log.error(f"Failed to process segment for '{regime_name}' with dates {start_date}-{end_date}: {e}")

    if not all_regime_segments:
        raise ValueError("No data segments extracted. Check regime definitions and data availability.")
        
    final_dataset = pd.concat(all_regime_segments).sort_index()
    log.info(f"--- 'Greatest Hits' dataset built. Total rows: {len(final_dataset)} ---")
    return final_dataset