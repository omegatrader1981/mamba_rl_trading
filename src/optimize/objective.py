# <<< DEFINITIVE FINAL FIX: Removes lookback_window from HPO to ensure stable checkpointing. >>>

import pandas as pd
import logging
import optuna
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import glob

from src.environment import FuturesTradingEnv
from src.features import create_feature_set
from src.evaluation import evaluate_agent
from .model_builder import create_ppo_model, create_sac_model

log = logging.getLogger(__name__)

class Objective:
    def __init__(self, cfg: DictConfig, df_train_raw: pd.DataFrame, df_val_raw: pd.DataFrame):
        self.cfg = cfg
        self.df_train_raw = df_train_raw
        self.df_val_raw = df_val_raw
        self.default_bad_value = -float('inf') if cfg.optimization.direction == 'maximize' else float('inf')
        self.checkpoint_dir = "/opt/ml/checkpoints"

    def __call__(self, trial: optuna.Trial) -> float:
        log.info(f"\n--- Starting/Resuming Optuna Trial #{trial.number} for agent type: {self.cfg.experiment.get('agent_type', 'ppo')} ---")
        try:
            trial_cfg = self.cfg.copy()
            
            trial_feature_cfg = trial_cfg.features.copy()
            OmegaConf.update(trial_feature_cfg, "hurst_window", trial.suggest_categorical('hurst_window', self.cfg.optimization.hurst_window_choices))
            OmegaConf.update(trial_feature_cfg, "momentum_window", trial.suggest_categorical('momentum_window', self.cfg.optimization.momentum_window_choices))
            
            combined_raw_df = pd.concat([self.df_train_raw, self.df_val_raw]).sort_index().drop_duplicates()
            df_featured, feature_cols = create_feature_set(combined_raw_df, OmegaConf.create({'features': trial_feature_cfg}), self.df_train_raw)

            df_train_featured = df_featured.loc[df_featured.index.isin(self.df_train_raw.index)]
            df_val_featured = df_featured.loc[df_featured.index.isin(self.df_val_raw.index)].copy()
            df_val_featured['source_regime'] = 'validation_set'

            trial_env_cfg = trial_cfg.environment.copy()
            
            agent_type = trial_cfg.experiment.get('agent_type', 'ppo')
            if agent_type == 'sac':
                activity_bonus = trial.suggest_float("activity_bonus_scale", self.cfg.optimization.activity_bonus_scale_min, self.cfg.optimization.activity_bonus_scale_max)
                hold_penalty = trial.suggest_float("hold_penalty_scale", self.cfg.optimization.hold_penalty_scale_min, self.cfg.optimization.hold_penalty_scale_max)
                win_bonus = trial.suggest_float("win_bonus_scale", self.cfg.optimization.win_bonus_scale_min, self.cfg.optimization.win_bonus_scale_max)
                loss_penalty = trial.suggest_float("loss_penalty_scale", self.cfg.optimization.loss_penalty_scale_min, self.cfg.optimization.loss_penalty_scale_max)
                unrealized_loss_penalty = trial.suggest_float("unrealized_loss_penalty_scale", self.cfg.optimization.unrealized_loss_penalty_scale_min, self.cfg.optimization.unrealized_loss_penalty_scale_max)
                OmegaConf.update(trial_env_cfg, "activity_bonus_scale", activity_bonus)
                OmegaConf.update(trial_env_cfg, "hold_penalty_scale", hold_penalty)
                OmegaConf.update(trial_env_cfg, "win_bonus_scale", win_bonus)
                OmegaConf.update(trial_env_cfg, "loss_penalty_scale", loss_penalty)
                OmegaConf.update(trial_env_cfg, "unrealized_loss_penalty_scale", unrealized_loss_penalty)
            
            OmegaConf.update(trial_cfg, "environment", trial_env_cfg)
            
            env_train = DummyVecEnv([lambda: FuturesTradingEnv(df_train_featured, feature_cols, trial_cfg)])

            latest_checkpoint = None
            if self.cfg.checkpointing.enabled and os.path.exists(self.checkpoint_dir):
                checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, f"trial_{trial.number}_*.zip"))
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                    log.info(f"Resuming Trial #{trial.number} from checkpoint: {latest_checkpoint}")

            if latest_checkpoint:
                if agent_type == 'sac':
                    model = SAC.load(latest_checkpoint, env=env_train)
                else:
                    model = PPO.load(latest_checkpoint, env=env_train)
            else:
                log.info(f"No checkpoint found for Trial #{trial.number}. Starting fresh.")
                if agent_type == 'sac':
                    model = create_sac_model(trial, env_train, trial_cfg)
                else:
                    model = create_ppo_model(trial, env_train, trial_cfg)

            callbacks = []
            if self.cfg.checkpointing.enabled:
                checkpoint_callback = CheckpointCallback(
                    save_freq=self.cfg.checkpointing.save_freq,
                    save_path=self.checkpoint_dir,
                    name_prefix=f"trial_{trial.number}"
                )
                callbacks.append(checkpoint_callback)

            remaining_timesteps = trial_cfg.optimization.trial_timesteps - model.num_timesteps
            if remaining_timesteps > 0:
                model.learn(
                    total_timesteps=remaining_timesteps,
                    callback=callbacks if callbacks else None,
                    reset_num_timesteps=False
                )

            eval_metrics = evaluate_agent(model, df_val_featured, feature_cols, trial_cfg, output_dir=None)
            metric_value = eval_metrics.get(trial_cfg.optimization.metric, self.default_bad_value)
            log.info(f"Trial #{trial.number}: Evaluation complete. Metric '{trial_cfg.optimization.metric}' = {metric_value:.4f}")
            return float(metric_value)

        except Exception as e:
            log.error(f"Trial #{trial.number}: Objective function failed: {e}", exc_info=True)
            return self.default_bad_value
