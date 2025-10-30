import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import pandas as pd
import logging
import os
from omegaconf import DictConfig
from typing import Tuple, Dict, Any, Optional

from src.utils.utils import ensure_dir
from .objective import Objective

log = logging.getLogger(__name__)

def run_optimization(cfg: DictConfig, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Tuple[Optional[optuna.Study], Optional[Dict[str, Any]]]:
    """Enhanced Optuna HPO with robust error handling"""
    
    # 🔻 DEFENSIVE FIX: Safely resolve optuna_db_name
    try:
        db_path = cfg.saving.optuna_db_name
        log.info(f"Successfully resolved optuna_db_name via interpolation: {db_path}")
    except Exception as e:
        log.warning(f"Could not resolve cfg.saving.optuna_db_name due to interpolation error: {e}")
        # Fallback: construct the name manually
        exp_name = "default_exp"
        if 'experiment' in cfg and 'name' in cfg.experiment:
            exp_name = cfg.experiment.name
        instrument = cfg.get("instrument", "unknown")
        db_path = f"/opt/ml/checkpoints/optuna_study_{exp_name}_{instrument}.db"
        log.info(f"Using fallback database path: {db_path}")
    
    storage_name = f"sqlite:///{db_path}"
    
    output_dir = cfg.saving.output_dir
    history_plot_path = os.path.join(output_dir, "optuna_history.png")
    importance_plot_path = os.path.join(output_dir, "optuna_importance.png")
    
    log.info(f"Creating/loading Optuna study '{cfg.saving.optuna_study_name}' using storage: {storage_name}")

    pruner = None
    if cfg.optimization.get('pruning', {}).get('enabled', False):
        pruner = optuna.pruners.MedianPruner()

    try:
        study = optuna.create_study(
            study_name=cfg.saving.optuna_study_name,
            storage=storage_name,
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
            log.error("No successful trials - could not determine best parameters.")

        return study, best_params

    except Exception as e:
        log.exception(f"Optuna study creation or optimization failed: {e}")
        return None, None

def _save_hpo_plots(study: optuna.Study, history_path: str, importance_path: str):
    """Save Optuna plots with comprehensive error handling"""
    try:
        if not study.trials:
            log.warning("No trials to plot.")
            return
            
        fig_history = plot_optimization_history(study)
        fig_history.write_image(history_path)
        log.info(f"Optuna history plot saved to {history_path}")

        if len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])) > 1:
            fig_importance = plot_param_importances(study)
            fig_importance.write_image(importance_path)
            log.info(f"Optuna importance plot saved: {importance_path}")
        else:
            log.info("Skipping importance plot (requires more than one completed trial).")
            
    except Exception as e:
        log.warning(f"Could not generate or save Optuna plots: {e}")
