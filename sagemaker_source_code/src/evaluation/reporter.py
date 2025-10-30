# src/evaluation/reporter.py
# <<< NEW MODULE: Generates all reports, plots, and saves artifacts. >>>

import os
import pandas as pd
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from typing import Dict, Any

from .backtest import BacktestResult

log = logging.getLogger(__name__)

QS_AVAILABLE = True
try:
    import quantstats as qs
except ImportError:
    QS_AVAILABLE = False

def generate_report(result: BacktestResult, metrics: Dict[str, Any], output_dir: str, cfg: DictConfig):
    """
    Saves all evaluation artifacts (plots, data, reports) to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log.info(f"Saving evaluation artifacts to {output_dir}...")

    # Save trades and equity curve
    if result.trades:
        pd.DataFrame(result.trades).to_csv(os.path.join(output_dir, cfg.saving.trades_filename), index=False)
    if not result.equity_curve.empty:
        result.equity_curve.to_csv(os.path.join(output_dir, cfg.saving.equity_filename))

    # Save metrics summary
    serializable_metrics = {k: (v if not isinstance(v, (np.generic, pd.Timestamp)) else str(v)) for k, v in metrics.items()}
    with open(os.path.join(output_dir, "evaluation_summary.json"), 'w') as f:
        json.dump(serializable_metrics, f, indent=4, default=str)

    # Generate and save plots
    _generate_plots(result, output_dir, cfg)

    # Generate QuantStats HTML report
    if QS_AVAILABLE and not result.equity_curve.empty:
        returns = result.equity_curve['portfolio_value'].pct_change().dropna()
        if len(returns) > 1:
            report_path = os.path.join(output_dir, cfg.saving.report_filename)
            qs.reports.html(returns, output=report_path, title=f"{cfg.data.symbol} Performance Report")
            log.info(f"QuantStats report saved: {report_path}")

def _generate_plots(result: BacktestResult, output_dir: str, cfg: DictConfig):
    """Helper function to create and save matplotlib plots."""
    # Equity Curve
    plt.figure(figsize=(15, 8))
    plt.plot(result.equity_curve.index, result.equity_curve['portfolio_value'])
    plt.title("Equity Curve"); plt.xlabel("Date"); plt.ylabel("Portfolio Value")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, cfg.saving.equity_curve_plot)); plt.close()

    # Positions
    plt.figure(figsize=(15, 4))
    plt.plot(result.positions.index, result.positions['position'], drawstyle='steps-post')
    plt.title("Positions"); plt.xlabel("Date"); plt.ylabel("Position")
    plt.yticks([-1, 0, 1]); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, cfg.saving.positions_plot)); plt.close()
    
    log.info("Generated and saved equity and position plots.")