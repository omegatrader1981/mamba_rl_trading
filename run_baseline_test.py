#!/usr/bin/env python
"""
run_baseline_test.py — Phase 0.5 Smoke Test (5-min data)
"""
import hydra
import logging
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import os
import quantstats as qs
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
        self.trending_regime = cfg.baseline_test.trending_hmm_regime
        self.initial_balance = cfg.environment.initial_balance
        self.point_value = cfg.environment.point_value
        self.commission = cfg.environment.commission_per_contract

    def run(self) -> BacktestResult:
        log.info(f"HMM-Filtered Momentum | mom={self.mom_threshold} | regime={self.trending_regime}")
        trending = (self.df['hmm_regime'] == self.trending_regime)
        long = (self.df['risk_adj_mom'] > self.mom_threshold) & trending
        short = (self.df['risk_adj_mom'] < -self.mom_threshold) & trending
        pos = pd.Series(0.0, index=self.df.index)
        pos[long] = 1.0
        pos[short] = -1.0
        pos = pos.ffill().fillna(0.0)
        price_diff = self.df['close'].diff() * self.point_value
        gross = pos.shift(1) * price_diff
        trades = pos.diff().abs()
        commission = trades * self.commission
        net = gross - commission
        equity = self.initial_balance + net.cumsum()
        return BacktestResult(
            equity_curve=pd.DataFrame({'portfolio_value': equity}),
            positions=pd.DataFrame({'position': pos}),
            initial_balance=self.initial_balance,
            final_balance=equity.iloc[-1]
        )

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("PHASE 0.5 SMOKE TEST — 5-MIN DATA")
    if 'baseline_test' not in cfg:
        OmegaConf.set_struct(cfg, False)
        cfg.baseline_test = {'mom_threshold': 1.5, 'trending_hmm_regime': 1}
        OmegaConf.set_struct(cfg, True)

    df_train_raw = build_regime_dataset(
        load_futures_data([f"{cfg.data.data_dir}/{f}" for f in cfg.data.data_filenames], **cfg.data.cleaning),
        {k: v for k, v in cfg.data.regime_definitions.items() if k.startswith('train_')}
    )
    val_dates = cfg.data.regime_definitions.validation_set[0]
    df_val_raw = load_futures_data([f"{cfg.data.data_dir}/{f}" for f in cfg.data.data_filenames], **cfg.data.cleaning).loc[val_dates[0]:val_dates[1]]
    df_val_feat, _ = create_feature_set(df_val_raw.copy(), cfg, df_train_raw)
    df_val_feat['ret'] = df_val_feat['close'].pct_change()
    df_val_feat['vol'] = df_val_feat['ret'].rolling(21).std()
    trending_regime = df_val_feat.groupby('hmm_regime')['vol'].mean().abs().idxmax()
    cfg.baseline_test.trending_hmm_regime = int(trending_regime)
    log.info(f"Trending Regime #{trending_regime}")
    result = SimpleRulesBaseline(df_val_feat, cfg).run()
    returns = result.equity_curve['portfolio_value'].pct_change().dropna()
    sharpe = qs.stats.sharpe(returns, annualize=True) if len(returns) > 20 else -99
    log.info(f"Sharpe: {sharpe:.3f}")
    os.makedirs("smoke_test_report", exist_ok=True)
    generate_report(result, {'sharpe': sharpe}, "smoke_test_report", cfg)
    log.info("Report: smoke_test_report/report.html")

if __name__ == "__main__":
    main()
