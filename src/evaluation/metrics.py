# src/evaluation/metrics.py
# <<< NEW MODULE: Calculates all performance metrics from backtest results. >>>

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from omegaconf import DictConfig

from .backtest import BacktestResult

log = logging.getLogger(__name__)

QS_AVAILABLE = True
try:
    import quantstats as qs
    qs.extend_pandas()
except ImportError:
    QS_AVAILABLE = False
    log.warning("QuantStats not installed. Some metrics will not be available.")

def calculate_performance_metrics(result: BacktestResult, cfg: DictConfig) -> Dict[str, Any]:
    """
    Calculates a dictionary of performance metrics from a BacktestResult object.
    """
    metrics = {
        'initial_balance': result.initial_balance,
        'final_portfolio_value': result.final_balance,
        'total_profit': result.final_balance - result.initial_balance,
        'total_profit_pct': ((result.final_balance - result.initial_balance) / result.initial_balance * 100) if result.initial_balance != 0 else 0,
        'n_trades': len(result.trades)
    }

    # Calculate QuantStats metrics
    if QS_AVAILABLE and not result.equity_curve.empty:
        returns = result.equity_curve['portfolio_value'].pct_change().dropna()
        if len(returns) > 1:
            log.info("Calculating QuantStats metrics...")
            qs_metrics = {
                'sharpe': qs.stats.sharpe(returns),
                'sortino': qs.stats.sortino(returns),
                'max_drawdown': qs.stats.max_drawdown(returns),
                'win_rate': qs.stats.win_rate(returns),
                'profit_factor': qs.stats.profit_factor(returns),
                'cagr': qs.stats.cagr(returns),
                'calmar': qs.stats.calmar(returns)
            }
            metrics.update(qs_metrics)

    # Calculate Prop Firm Rule Checks
    log.info("Calculating simulated prop firm rule checks...")
    env_metrics = result.metrics
    rule_checks = {
        'Drawdown OK': not env_metrics.get('drawdown_violated', False),
        'Profit Target Met': metrics['total_profit'] >= cfg.environment.get('profit_target', float('inf')),
        'Min Trading Days Met': env_metrics.get('days_traded', 0) >= cfg.environment.get('min_trading_days_target', 0)
    }
    metrics['rule_checks'] = rule_checks
    metrics['overall_eval_passed'] = all(rule_checks.values())

    return metrics