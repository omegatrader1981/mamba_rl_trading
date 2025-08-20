# <<< DEFINITIVE VERSION: Uses a resilient, S3-backed Optuna study database. >>>

import optuna
from optuna.storages import JournalStorage, JournalFileStorage # For S3
from optuna.visualization import plot_optimization_history, plot_param_importances
import pandas as pd
import logging
import os
from omegaconf import DictConfig
from typing import Tuple, Dict, Any, Optional

from src.utils import ensure_dir
from .objective import Objective

log = logging.getLogger(__name__)

def run_optimization(cfg: DictConfig, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Tuple[Optional[optuna.Study], Optional[Dict[str, Any]]]:
    """
    Sets up and executes the Optuna hyperparameter optimization study using S3 storage.
    """
    storage_path = cfg.saving.optuna_db_name
    storage = JournalStorage(JournalFileStorage(storage_path))
    
    output_dir = os.getcwd()
    history_plot_path = os.path.join(output_dir, "optuna_history.png")
    importance_plot_path = os.path.join(output_dir, "optuna_importance.png")
    
    ensure_dir(output_dir)
    log.info(f"Creating/loading Optuna study '{cfg.saving.optuna_study_name}' using S3 storage: {storage_path}")

    pruner = None
    if cfg.optimization.pruning.enabled:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=cfg.optimization.pruning.n_startup_trials,
            n_warmup_steps=cfg.optimization.pruning.n_warmup_steps,
            interval_steps=cfg.optimization.pruning.interval_steps
        )

    try:
        study = optuna.create_study(
            study_name=cfg.saving.optuna_study_name,
            storage=storage,
            load_if_exists=True,
            direction=cfg.optimization.direction,
            pruner=pruner
        )

        objective_callable = Objective(cfg, df_train, df_val)

        study.optimize(
            objective_callable,
            n_trials=cfg.optimization.n_trials,
            timeout=cfg.optimization.get('timeout_seconds'),
            n_jobs=cfg.optimization.n_jobs
        )

        log.info("Optuna Optimization Finished.")
        
        best_params = study.best_trial.params if study.best_trial else None
        if best_params:
            log.info(f"Best trial #{study.best_trial.number}: Value={study.best_trial.value:.4f}")
            _save_hpo_plots(study, history_plot_path, importance_plot_path)
        else:
            log.error("Could not determine best parameters from the HPO study.")

        return study, best_params

    except Exception as e:
        log.exception(f"Optuna study creation or main optimization loop failed: {e}")
        return None, None

def _save_hpo_plots(study: optuna.Study, history_path: str, importance_path: str):
    """Saves the standard Optuna visualization plots to the specified paths."""
    try:
        fig_history = plot_optimization_history(study)
        fig_history.write_image(history_path)
        log.info(f"Optuna history plot saved to {history_path}")

        fig_importance = plot_param_importances(study)
        fig_importance.write_image(importance_path)
        log.info(f"Optuna importance plot saved to {importance_path}")
    except (ValueError, ImportError) as e:
        log.warning(f"Could not generate or save Optuna plots: {e}")
