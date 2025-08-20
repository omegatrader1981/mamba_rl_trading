# <<< DEFINITIVE FINAL VERSION: Fully instrumented for robust SageMaker execution. >>>

import pandas as pd
import logging
import joblib
import os
import glob
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import List

from src.optimize import run_optimization
from src.evaluation import evaluate_agent
from src.optimize.model_builder import create_ppo_model, create_sac_model
from src.environment import FuturesTradingEnv
from src.utils.sagemaker_utils import initialize_sagemaker_environment, sagemaker_safe

log = logging.getLogger(__name__)

@sagemaker_safe
def train_and_evaluate_model(
    cfg: DictConfig,
    df_train_s: pd.DataFrame,
    df_val_s: pd.DataFrame,
    df_test_s: pd.DataFrame,
    feature_cols: List[str],
    scaler: object
):
    cfg = initialize_sagemaker_environment(cfg)
    
    log.info("--- Starting Model Training & Evaluation Pipeline (Resilient & Validated) ---")
    
    checkpoint_dir = cfg.checkpointing.s3_base_path
    sagemaker_model_dir = "/opt/ml/model"
    sagemaker_output_dir = cfg.saving.output_dir

    study, best_params = run_optimization(cfg, df_train_s, df_val_s)
    if not best_params:
        raise RuntimeError("Hyperparameter optimization failed to produce best parameters.")
    
    best_params_path = os.path.join(sagemaker_output_dir, cfg.saving.best_params_filename)
    joblib.dump(best_params, best_params_path)
    log.info(f"Best HPO parameters saved to {best_params_path}")

    log.info("Preparing for final model training...")
    df_train_val_s = pd.concat([df_train_s, df_val_s]).sort_index()
    
    final_full_cfg = cfg.copy()
    OmegaConf.set_struct(final_full_cfg, False)
    final_full_cfg.environment.lookback_window = best_params.get('lookback_window') or cfg.environment.lookback_window
    agent_type = final_full_cfg.experiment.get('agent_type', 'ppo')
    if agent_type == 'sac':
        final_full_cfg.environment.activity_bonus_scale = best_params.get('activity_bonus_scale') or cfg.environment.activity_bonus_scale
        final_full_cfg.environment.hold_penalty_scale = best_params.get('hold_penalty_scale') or cfg.environment.hold_penalty_scale
        final_full_cfg.environment.win_bonus_scale = best_params.get('win_bonus_scale') or cfg.environment.win_bonus_scale
        final_full_cfg.environment.loss_penalty_scale = best_params.get('loss_penalty_scale') or cfg.environment.loss_penalty_scale
        final_full_cfg.environment.unrealized_loss_penalty_scale = best_params.get('unrealized_loss_penalty_scale') or cfg.environment.unrealized_loss_penalty_scale
    OmegaConf.set_struct(final_full_cfg, True)

    final_env = DummyVecEnv([lambda: FuturesTradingEnv(df_train_val_s, feature_cols, final_full_cfg)])

    latest_checkpoint = None
    if cfg.checkpointing.enabled and os.path.exists(checkpoint_dir):
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "final_model_*.zip"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            log.info(f"Resuming FINAL training from checkpoint: {latest_checkpoint}")

    if latest_checkpoint:
        if agent_type == 'sac':
            final_model = SAC.load(latest_checkpoint, env=final_env)
        else:
            final_model = PPO.load(latest_checkpoint, env=final_env)
    else:
        log.info("No final model checkpoint found. Constructing new model from best HPO params.")
        class DummyTrial:
            def __init__(self, params): self.params = params
            def suggest_float(self, name, low, high, log=False): return self.params.get(name)
            def suggest_categorical(self, name, choices): return self.params.get(name)
            def suggest_int(self, name, low, high): return self.params.get(name)
        dummy_trial = DummyTrial(best_params)
        if agent_type == 'sac':
            final_model = create_sac_model(dummy_trial, final_env, final_full_cfg)
        else:
            final_model = create_ppo_model(dummy_trial, final_env, final_full_cfg)
        
    callbacks = []
    if cfg.checkpointing.enabled:
        checkpoint_callback = CheckpointCallback(
            save_freq=cfg.checkpointing.save_freq,
            save_path=checkpoint_dir,
            name_prefix="final_model"
        )
        callbacks.append(checkpoint_callback)

    log.info(f"Starting/resuming final training for {final_full_cfg.training.total_timesteps} timesteps...")
    remaining_timesteps = final_full_cfg.training.total_timesteps - final_model.num_timesteps
    if remaining_timesteps > 0:
        final_model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks if callbacks else None,
            reset_num_timesteps=False
        )
    
    final_model_sagemaker_path = os.path.join(sagemaker_model_dir, "final_model.zip")
    final_model.save(final_model_sagemaker_path)
    log.info(f"Final model saved to SageMaker model directory: {final_model_sagemaker_path}")
    
    final_model_output_path = os.path.join(sagemaker_output_dir, cfg.saving.final_model_filename)
    final_model.save(final_model_output_path)
    log.info(f"Final model also saved to output directory: {final_model_output_path}")

    log.info("Evaluating final model on unseen test data...")
    evaluate_agent(final_model, df_test_s, feature_cols, final_full_cfg, output_dir=sagemaker_output_dir)
    
    log.info("--- Model Training & Evaluation Pipeline COMPLETED ---")
