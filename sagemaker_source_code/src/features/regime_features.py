# src/features/regime_features.py
# <<< NEW MODULE: For HMM-based market regime detection. >>>

import pandas as pd
import logging
from hmmlearn.hmm import GaussianHMM

log = logging.getLogger(__name__)

def calculate_hmm_regime(df: pd.DataFrame, df_train: pd.DataFrame, n_components: int, min_samples: int) -> pd.DataFrame:
    log.info(f"Calculating HMM Regimes (n_components: {n_components})...")
    try:
        def get_hmm_inputs(dframe):
            inputs = pd.DataFrame(index=dframe.index)
            inputs['log_return'] = dframe['close'].pct_change().rolling(5).mean()
            inputs['volatility'] = dframe['close'].pct_change().rolling(21).std().rolling(5).mean()
            return inputs.dropna()

        hmm_train_inputs = get_hmm_inputs(df_train)
        if len(hmm_train_inputs) < min_samples:
            raise ValueError(f"Not enough samples ({len(hmm_train_inputs)}) to fit HMM.")

        hmm_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000, random_state=42)
        hmm_model.fit(hmm_train_inputs)
        
        hmm_predict_inputs = get_hmm_inputs(df)
        predicted_regimes = hmm_model.predict(hmm_predict_inputs)
        
        df['hmm_regime'] = pd.Series(predicted_regimes, index=hmm_predict_inputs.index).ffill().bfill().fillna(0).astype(int)
    except Exception as e:
        log.error(f"HMM regime detection failed: {e}. Defaulting regime to 0.")
        df['hmm_regime'] = 0
    return df