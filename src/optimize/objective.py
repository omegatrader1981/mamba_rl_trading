# src/optimize/objective.py
# <<< NEW MODULE: Defines the Objective class for a single Optuna trial. >>>

import pandas as pd
import logging
import optuna
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv

# Import from our refactored project structure
from src.environment import FuturesTradingEnv
from src.features import create_feature_set
from src.evaluation import evaluate_agent
from .model_builder import create_ppo_model

log = logging.getLogger(__name__)

class Objective:
    """
    An Optuna objective class that handles a single trial, including
    feature engineering, model creation, training, and evaluation.
    """
    def __init__(self, cfg: DictConfig, df_train_raw: pd.DataFrame, df_val_raw: pd.DataFrame):
        self.cfg = cfg
        self.df_train_raw = df_train_raw
        self.df_val_raw = df_val_raw
        self.default_bad_value = -float('inf') if cfg.optimization.direction == 'maximize' else float('inf')

    def __call__(self, trial: optuna.Trial) -> float:
        log.info(f"\n--- Starting Optuna Trial #{trial.number} ---")
        try:
            # 1. Sample feature parameters and create features for this trial
            # Note: This is computationally intensive but required for joint optimization.
            feature_params = {
                'hurst_window': trial.suggest_categorical('hurst_window', self.cfg.optimization.hurst_window_choices),
                'momentum_window': trial.suggest_categorical('momentum_window', self.cfg.optimization.momentum_window_choices),
                'volatility_window': trial.suggest_categorical('volatility_window', self.cfg.optimization.volatility_window_choices)
            }
            trial_feature_cfg = self.cfg.features.copy()
            OmegaConf.update(trial_feature_cfg, "hurst_window", feature_params['hurst_window'])
            OmegaConf.update(trial_feature_cfg, "momentum_window", feature_params['momentum_window'])
            
            combined_raw_df = pd.concat([self.df_train_raw, self.df_val_raw]).sort_index().drop_duplicates()
            df_featured, feature_cols = create_feature_set(combined_raw_df, OmegaConf.create({'features': trial_feature_cfg}), self.df_train_raw)

            df_train_featured = df_featured.loc[df_featured.index.isin(self.df_train_raw.index)]
            df_val_featured = df_featured.loc[df_featured.index.isin(self.df_val_raw.index)]

            # 2. Setup environment with trial-specific lookback window
            lookback_window = trial.suggest_int('lookback_window', self.cfg.environment.min_lookback_hpo, self.cfg.environment.max_lookback_hpo)
            trial_env_cfg = self.cfg.environment.copy()
            OmegaConf.update(trial_env_cfg, "lookback_window", lookback_window)
            trial_full_cfg = self.cfg.copy()
            OmegaConf.update(trial_full_cfg, "environment", trial_env_cfg)
            
            env_train = DummyVecEnv([lambda: FuturesTradingEnv(df_train_featured, feature_cols, trial_full_cfg)])

            # 3. Create the PPO model using the dedicated builder
            model = create_ppo_model(trial, env_train, self.cfg)

            # 4. Train the model
            model.learn(total_timesteps=self.cfg.optimization.trial_timesteps)

            # 5. Evaluate the model using our refactored, robust evaluation function
            # This is a major improvement, eliminating the redundant _evaluate_hpo function.
            eval_metrics = evaluate_agent(model, df_val_featured, feature_cols, trial_full_cfg, output_dir=None)
            metric_value = eval_metrics.get(self.cfg.optimization.metric, self.default_bad_value)

            log.info(f"Trial #{trial.number}: Evaluation complete. Metric '{self.cfg.optimization.metric}' = {metric_value:.4f}")
            return float(metric_value)

        except optuna.exceptions.TrialPruned as e:
            log.info(f"Trial #{trial.number}: Pruned. Reason: {e}")
            raise
        except Exception as e:
            log.error(f"Trial #{trial.number}: Objective function failed: {e}", exc_info=True)
            return self.default_bad_value