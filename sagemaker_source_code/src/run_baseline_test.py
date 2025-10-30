import hydra
import logging
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import os
import quantstats as qs
import numpy as np

from src.data import load_futures_data, build_regime_dataset
from src.features import create_feature_set
from src.evaluation.reporter import generate_report
from src.evaluation.backtest import BacktestResult

log = logging.getLogger(__name__)

class SimpleRulesBaseline:
    def __init__(self, df: pd.DataFrame, cfg: DictConfig):
        self.df = df
        self.cfg = cfg
        self.mom_threshold = cfg.baseline_test.mom_threshold
        self.trending_hmm_regime = cfg.baseline_test.trending_hmm_regime
        self.initial_balance = cfg.environment.initial_balance
        self.point_value = cfg.environment.point_value
        self.commission = cfg.environment.commission_per_contract

    def run(self) -> BacktestResult:
        log.info(f"Running HMM-FILTERED MOMENTUM baseline with mom_thresh={self.mom_threshold}, hmm_regime={self.trending_hmm_regime}")

        is_trending_regime = (self.df['hmm_regime'] == self.trending_hmm_regime)
        
        long_signal = (self.df['risk_adj_mom'] > self.mom_threshold) & is_trending_regime
        short_signal = (self.df['risk_adj_mom'] < -self.mom_threshold) & is_trending_regime
        
        positions = pd.Series(0.0, index=self.df.index)
        positions[long_signal] = 1.0
        positions[short_signal] = -1.0
        positions = positions.ffill().fillna(0.0)
        
        price_diff = self.df['close'].diff() * self.point_value
        gross_pnl = (positions.shift(1) * price_diff)
        trades = positions.diff().abs()
        commission_costs = trades * self.commission
        net_pnl = gross_pnl - commission_costs
        equity_curve = self.initial_balance + net_pnl.cumsum()
        equity_df = pd.DataFrame({'portfolio_value': equity_curve})
        positions_df = pd.DataFrame({'position': positions})
        
        return BacktestResult(
            equity_curve=equity_df,
            positions=positions_df,
            trades=[],
            initial_balance=self.initial_balance,
            final_balance=equity_curve.iloc[-1]
        )

def calculate_metrics(result: BacktestResult):
    returns = result.equity_curve['portfolio_value'].pct_change().dropna()
    if len(returns) < 20:
        return {'sharpe': -99.0, 'max_drawdown': -1.0}
    sharpe = qs.stats.sharpe(returns, annualize=True)
    max_drawdown = qs.stats.max_drawdown(returns)
    return {'sharpe': sharpe if not pd.isna(sharpe) else -99.0, 'max_drawdown': max_drawdown}

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("--- Starting Baseline Test: HMM-FILTERED MOMENTUM ---")
    
    if 'baseline_test' not in cfg:
        OmegaConf.set_struct(cfg, False)
        cfg.baseline_test = {
            'mom_threshold': 1.5,
            'trending_hmm_regime': 1
        }
        OmegaConf.set_struct(cfg, True)

    log.info("Using previously optimized feature parameters.")
    OmegaConf.update(cfg, "features.momentum_window", 48)
    OmegaConf.update(cfg, "features.volatility_window", 12)
    OmegaConf.update(cfg, "features.hurst_window", 151)

    df_train_raw = build_regime_dataset(
        load_futures_data([f"data/{fname}" for fname in cfg.data.data_filenames], **cfg.data.cleaning),
        {k: v for k, v in cfg.data.regime_definitions.items() if k.startswith('train_')}
    )
    val_dates = cfg.data.regime_definitions.validation_set[0]
    df_val_raw = load_futures_data([f"data/{fname}" for fname in cfg.data.data_filenames], **cfg.data.cleaning).loc[val_dates[0]:val_dates[1]]
    df_val_featured, _ = create_feature_set(df_val_raw.copy(), cfg, df_train_raw)

    log.info("--- Analyzing HMM Regime Characteristics ---")
    df_val_featured['returns'] = df_val_featured['close'].pct_change()
    df_val_featured['volatility'] = df_val_featured['returns'].rolling(21).std()
    regime_stats = df_val_featured.groupby('hmm_regime')[['returns', 'volatility']].mean().abs()
    log.info(f"\n{regime_stats}")
    
    trending_regime = regime_stats['volatility'].idxmax()
    log.info(f"Identified Regime #{trending_regime} as the most volatile/trending state. Using this for the test.")
    
    # <<< THE FIX IS HERE: Cast the NumPy type to a native Python int >>>
    cfg.baseline_test.trending_hmm_regime = int(trending_regime)
    
    baseline = SimpleRulesBaseline(df_val_featured, cfg)
    result = baseline.run()

    metrics = calculate_metrics(result)
    log.info(f"HMM-Filtered Momentum Test Final Sharpe Ratio: {metrics['sharpe']:.4f}")
    log.info(f"HMM-Filtered Momentum Test Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    output_dir = "hmm_filtered_momentum_report"
    os.makedirs(output_dir, exist_ok=True)
    generate_report(result, metrics, output_dir, cfg)
    log.info(f"Full analysis report saved to '{output_dir}' directory.")

if __name__ == "__main__":
    main()
