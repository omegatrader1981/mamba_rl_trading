import hydra
import optuna
import logging
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import quantstats as qs # <<< THE FIX IS HERE (Part 1)

from src.data import load_futures_data, build_regime_dataset
from src.features import create_feature_set

log = logging.getLogger(__name__)

class SimpleRulesBaseline:
    def __init__(self, df: pd.DataFrame, vwap_threshold: float = 0.001, mom_threshold: float = 1.5):
        self.df = df
        self.vwap_threshold = vwap_threshold
        self.mom_threshold = mom_threshold
        # We need these for a simplified equity curve to get pct_change
        self.initial_balance = 100000 
        self.point_value = 0.20
        self.commission = 0.04

    def run(self) -> float:
        long_signal = (self.df['session_vwap_dist'] > self.vwap_threshold) & (self.df['risk_adj_mom'] > self.mom_threshold)
        short_signal = (self.df['session_vwap_dist'] < -self.vwap_threshold) & (self.df['risk_adj_mom'] < -self.mom_threshold)
        
        positions = pd.Series(0.0, index=self.df.index)
        positions[long_signal] = 1.0
        positions[short_signal] = -1.0
        positions = positions.ffill().fillna(0.0)
        
        # --- Use a simplified PnL calculation to get returns for QuantStats ---
        price_diff = self.df['close'].diff() * self.point_value
        gross_pnl = (positions.shift(1) * price_diff)
        trades = positions.diff().abs()
        commission_costs = trades * self.commission
        net_pnl = gross_pnl - commission_costs
        equity_curve = self.initial_balance + net_pnl.cumsum()
        returns = equity_curve.pct_change().dropna()

        # <<< THE FIX IS HERE (Part 2): Use QuantStats for correct annualization >>>
        if len(returns) < 20: # Need a minimum number of trades to be meaningful
            return -99.0 # Return a large negative number for failed trials

        sharpe = qs.stats.sharpe(returns, annualize=True)
        return sharpe if not pd.isna(sharpe) else -99.0

# ... (the rest of the file is the same) ...

def objective(trial: optuna.Trial, cfg: DictConfig, df_train_raw: pd.DataFrame, df_val_raw: pd.DataFrame) -> float:
    trial_feature_cfg = cfg.features.copy()
    OmegaConf.update(trial_feature_cfg, "momentum_window", trial.suggest_int("momentum_window", 5, 50))
    OmegaConf.update(trial_feature_cfg, "volatility_window", trial.suggest_int("volatility_window", 10, 60))
    OmegaConf.update(trial_feature_cfg, "hurst_window", trial.suggest_int("hurst_window", 50, 200))

    df_val_featured, _ = create_feature_set(df_val_raw.copy(), OmegaConf.create({'features': trial_feature_cfg}), df_train_raw)

    baseline = SimpleRulesBaseline(df_val_featured)
    sharpe_ratio = baseline.run()
    
    log.info(f"Trial #{trial.number}: mom_win={trial.params['momentum_window']}, vol_win={trial.params['volatility_window']}, hurst_win={trial.params['hurst_window']} -> Sharpe={sharpe_ratio:.4f}")
    
    return sharpe_ratio

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("--- Starting Feature Parameter Optimization (Phase 5c) ---")
    
    df_train_raw = build_regime_dataset(
        load_futures_data([f"data/{fname}" for fname in cfg.data.data_filenames], **cfg.data.cleaning),
        {k: v for k, v in cfg.data.regime_definitions.items() if k.startswith('train_')}
    )
    val_dates = cfg.data.regime_definitions.validation_set[0]
    df_val_raw = load_futures_data([f"data/{fname}" for fname in cfg.data.data_filenames], **cfg.data.cleaning).loc[val_dates[0]:val_dates[1]]

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, cfg, df_train_raw, df_val_raw), n_trials=100)

    log.info("--- Feature optimization complete ---")
    log.info(f"Best Sharpe Ratio from simple baseline: {study.best_value:.4f}")
    log.info(f"To achieve this, use these parameters in your conf/config.yaml: {study.best_params}")

if __name__ == "__main__":
    main()
