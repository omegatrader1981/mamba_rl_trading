# src/evaluate.py
# <<< REFACTORED: This is now a lean orchestrator. >>>

import logging
from typing import Dict, Any, List
from omegaconf import DictConfig, OmegaConf
import pandas as pd

# <<< Relative imports from our new, organized package >>>
from .backtest import BacktestRunner
from .metrics import calculate_performance_metrics
from .reporter import generate_report

# Import the refactored environment
from src.environment import FuturesTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm

log = logging.getLogger(__name__)

def evaluate_agent(
    model: BaseAlgorithm,
    df_eval: pd.DataFrame,
    feature_cols: List[str],
    cfg: DictConfig,
    output_dir: str = None,
    log_prefix: str = "Evaluation"
) -> Dict[str, Any]:
    """
    Orchestrates the entire evaluation process:
    1. Runs the backtest to get raw results.
    2. Calculates performance metrics from the results.
    3. Generates and saves a report if an output directory is provided.
    """
    log.info(f"--- {log_prefix}: Starting Evaluation (Orchestrator) ---")
    if df_eval.empty:
        log.warning(f"{log_prefix}: Evaluation DataFrame is empty. Skipping.")
        return {"status": "skipped_empty_data"}

    try:
        # 1. Setup Environment
        eval_env = DummyVecEnv([lambda: FuturesTradingEnv(df_eval.copy(), feature_cols[:], cfg)])
        
        # 2. Run Backtest
        runner = BacktestRunner(model, eval_env, deterministic=cfg.evaluation.deterministic_actions)
        backtest_result = runner.run()
        
        # 3. Calculate Metrics
        metrics = calculate_performance_metrics(backtest_result, cfg)
        
        # 4. Generate Report (optional)
        if output_dir:
            generate_report(backtest_result, metrics, output_dir, cfg)
        
        eval_env.close()
        log.info(f"--- {log_prefix}: Evaluation Complete ---")
        metrics['status'] = 'completed_successfully'
        return metrics

    except Exception as e:
        log.exception(f"{log_prefix}: A critical error occurred during the evaluation process: {e}")
        return {"status": "error", "error_message": str(e)}