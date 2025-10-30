# src/features/time_features.py
# <<< NEW MODULE: For time-based cyclical features. >>>

import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Calculating Time-of-Day features...")
    minutes_in_day = 24 * 60
    time_in_minutes = df.index.hour * 60 + df.index.minute
    df['time_sin'] = np.sin(2 * np.pi * time_in_minutes / minutes_in_day)
    df['time_cos'] = np.cos(2 * np.pi * time_in_minutes / minutes_in_day)
    return df