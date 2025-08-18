# <<< DEFINITIVE FINAL FIX: Corrects the naming inconsistency for reward shaping parameters. >>>

import pandas as pd
import logging
import joblib
import os
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from typing import List

from src.optimize import run_optimization
from src.evaluation import evaluate_agent
from src.model import MambaActorCriticPolicy, MambaFeaturesExtractor
from src.environment import FuturesTradingEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from src.torch_utils import get_device

log = logging.getLogger(__name__)

def train_and_evaluate_model(
    cfg: DictConfig,
    df_train_s: pd.DataFrame,
    df_val_s: pd.DataFrame,
    df_test_s: pd.DataFrame,
    feature_cols: List[str],
    scaler: object
):
    log.info("--- Starting Model Training & Evaluation Pipeline ---")

    study, best_params = run_optimization(cfg, df_train_s, df_val_s)
    if not best_params:
        raise RuntimeError("Hyperparameter optimization failed to produce best parameters.")
    
    sagemaker_model_dir = "/opt/ml/model"
    sagemaker_output_dir = "/opt/ml/output/data" 
    os.makedirs(sagemaker_model_dir, exist_ok=True)
    os.makedirs(sagemaker_output_dir, exist_ok=True)
    
    best_params_path = os.path.join(sagemaker_output_dir, cfg.saving.best_params_filename)
    joblib.dump(best_params, best_params_path)
    log.info(f"Best HPO parameters saved to {best_params_path}")

    log.info("Training final model with best parameters on combined train+validation data...")
    df_train_val_s = pd.concat([df_train_s, df_val_s]).sort_index()
    
    final_full_cfg = cfg.copy()
    
    # Temporarily disable struct mode to allow updates
    OmegaConf.set_struct(final_full_cfg, False)
    
    # Update the final config with the best parameters found during HPO
    final_full_cfg.environment.lookback_window = best_params.get('lookback_window') or cfg.environment.lookback_window
    
    agent_type = final_full_cfg.experiment.get('agent_type', 'ppo')
    if agent_type == 'sac':
        # <<< THE FIX IS HERE: Use the correct, consistent parameter names >>>
        final_full_cfg.environment.activity_bonus_scale = best_params.get('activity_bonus_scale') or cfg.environment.activity_bonus_scale
        final_full_cfg.environment.hold_penalty_scale = best_params.get('hold_penalty_scale') or cfg.environment.hold_penalty_scale
        final_full_cfg.environment.win_bonus_scale = best_params.get('win_bonus_scale') or cfg.environment.win_bonus_scale
        final_full_cfg.environment.loss_penalty_scale = best_params.get('loss_penalty_scale') or cfg.environment.loss_penalty_scale
        final_full_cfg.environment.unrealized_loss_penalty_scale = best_params.get('unrealized_loss_penalty_scale') or cfg.environment.unrealized_loss_penalty_scale

    # Re-enable struct mode for safety
    OmegaConf.set_struct(final_full_cfg, True)

    final_env = DummyVecEnv([lambda: FuturesTradingEnv(df_train_val_s, feature_cols, final_full_cfg)])

    log.info("Constructing final model architecture from best HPO parameters...")
    
    # Create a dummy trial object to reconstruct the model from best_params
    class DummyTrial:
        def __init__(self, params): self.params = params
        def suggest_float(self, name, low, high, log=False): return self.params.get(name)
        def suggest_categorical(self, name, choices): return self.params.get(name)
        def suggest_int(self, name, low, high): return self.params.get(name)

    dummy_trial = DummyTrial(best_params)

    if agent_type == 'sac':
        from src.optimize.model_builder import create_sac_model
        final_model = create_sac_model(dummy_trial, final_env, final_full_cfg)
    else: # Default to PPO
        from src.optimize.model_builder import create_ppo_model
        final_model = create_ppo_model(dummy_trial, final_env, final_full_cfg)

    log.info(f"Starting final training for {final_full_cfg.training.total_timesteps} timesteps...")
    final_model.learn(total_timesteps=final_full_cfg.training.total_timesteps)
    
    final_model_sagemaker_path = os.path.join(sagemaker_model_dir, "final_model.zip")
    final_model.save(final_model_sagemaker_path)
    log.info(f"Final model saved to SageMaker model directory: {final_model_sagemaker_path}")
    
    final_model_output_path = os.path.join(sagemaker_output_dir, cfg.saving.final_model_filename)
    final_model.save(final_model_output_path)
    log.info(f"Final model also saved to output directory: {final_model_output_path}")

    log.info("Evaluating final model on unseen test data...")
    evaluate_agent(final_model, df_test_s, feature_cols, final_full_cfg, output_dir=sagemaker_output_dir)
    
    log.info("--- Model Training & Evaluation Pipeline COMPLETED ---")
