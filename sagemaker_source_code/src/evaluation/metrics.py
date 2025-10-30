# <<< UPDATED: Adds the 'self_exit_rate' to measure agent agency. >>>

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

    # <<< THE NEW LOGIC IS HERE >>>
    if result.trades:
        trades_df = pd.DataFrame(result.trades)
        # An agent-initiated exit is ACTION_EXIT_POSITION (3)
        num_self_exits = (trades_df['exit_action'] == 3).sum()
        total_exits = len(trades_df)
        
        self_exit_rate = (num_self_exits / total_exits) if total_exits > 0 else 0.0
        metrics['self_exit_rate'] = self_exit_rate
        log.info(f"Agent Self-Exit Rate: {self_exit_rate:.2%}")
    else:
        metrics['self_exit_rate'] = 0.0

    # Calculate returns and validate length
    if not result.equity_curve.empty:
        returns = result.equity_curve['portfolio_value'].pct_change().dropna()
        if len(returns) < 10:
            log.warning("Insufficient return data for reliable metrics")
            # Provide minimal safe defaults
            metrics.update({
                'sharpe': 0.0,
                'sortino': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'cagr': 0.0,
                'calmar': 0.0
            })
        elif QS_AVAILABLE:
            log.info("Calculating QuantStats metrics...")
            try:
                qs_metrics = {
                    'sharpe': float(qs.stats.sharpe(returns)),
                    'sortino': float(qs.stats.sortino(returns)),
                    'max_drawdown': float(qs.stats.max_drawdown(returns)),
                    'win_rate': float(qs.stats.win_rate(returns)),
                    'profit_factor': float(qs.stats.profit_factor(returns)),
                    'cagr': float(qs.stats.cagr(returns)),
                    'calmar': float(qs.stats.calmar(returns))
                }
                metrics.update(qs_metrics)
            except Exception as e:
                log.error(f"Error computing QuantStats metrics: {e}")
                metrics.update({
                    'sharpe': 0.0,
                    'sortino': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'cagr': 0.0,
                    'calmar': 0.0
                })
        else:
            # QuantStats not available – return zeros
            metrics.update({
                'sharpe': 0.0,
                'sortino': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'cagr': 0.0,
                'calmar': 0.0
            })
    else:
        # No equity curve
        metrics.update({
            'sharpe': 0.0,
            'sortino': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'cagr': 0.0,
            'calmar': 0.0
        })

    # Prop firm rule checks
    env_metrics = result.metrics
    rule_checks = {
        'Drawdown OK': not env_metrics.get('drawdown_violated', False),
        'Profit Target Met': metrics.get('total_profit', 0) >= cfg.environment.get('profit_target', float('inf')),
        'Min Trading Days Met': env_metrics.get('days_traded', 0) >= cfg.environment.get('min_trading_days_target', 0)
    }
    metrics['rule_checks'] = rule_checks
    metrics['overall_eval_passed'] = all(rule_checks.values())

    return metrics
