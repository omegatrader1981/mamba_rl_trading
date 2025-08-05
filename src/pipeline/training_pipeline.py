# src/pipeline/training_pipeline.py
# <<< CORRECTED: Added SageMaker model directory support >>>

import pandas as pd
import logging
import joblib
import os
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from typing import List

from src.optimize import run_optimization
from src.evaluation import evaluate_agent
from src.model import MambaActorCriticPolicy
from src.environment import FuturesTradingEnv
from stable_baselines3 import PPO
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
    
    # <<< SAGEMAKER FIX: Use SageMaker model directory for final model >>>
    sagemaker_model_dir = "/opt/ml/model"
    output_dir = os.getcwd()  # Keep this for other outputs
    
    # Ensure SageMaker model directory exists
    os.makedirs(sagemaker_model_dir, exist_ok=True)
    
    best_params_path = os.path.join(output_dir, cfg.saving.best_params_filename)
    joblib.dump(best_params, best_params_path)
    log.info(f"Best HPO parameters saved to {best_params_path}")

    log.info("Training final model with best parameters on combined train+validation data...")
    df_train_val_s = pd.concat([df_train_s, df_val_s]).sort_index()
    
    final_env_cfg = cfg.environment.copy()
    OmegaConf.update(final_env_cfg, "lookback_window", best_params.get('lookback_window'))
    OmegaConf.update(final_env_cfg, "activity_reward_scale", best_params.get('activity_reward_scale'))
    final_full_cfg = cfg.copy()
    OmegaConf.update(final_full_cfg, "environment", final_env_cfg)

    final_env = DummyVecEnv([lambda: FuturesTradingEnv(df_train_val_s, feature_cols, final_full_cfg)])

    log.info("Constructing final model architecture from best HPO parameters...")
    act_fn_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'silu': nn.SiLU}
    mlp_activation = act_fn_map.get(cfg.model.mlp_activation_fn_name, nn.ReLU)
    mamba_activation = act_fn_map.get(cfg.model.mamba_activation_fn_name, nn.SiLU)

    policy_kwargs = dict(
        features_extractor_class=MambaActorCriticPolicy.features_extractor_class,
        features_extractor_kwargs=dict(
            features_dim=best_params['features_dim'], mamba_d_model=best_params['mamba_d_model'],
            num_mamba_layers=best_params['num_mamba_layers'], mamba_d_state=best_params['d_state'],
            dropout_rate=best_params['dropout_rate'], activation_fn_class=mamba_activation,
            mamba_d_conv=cfg.model.mamba_d_conv_default, mamba_expand=cfg.model.mamba_expand_default
        ),
        activation_fn=mlp_activation,
        optimizer_kwargs=dict(weight_decay=best_params['weight_decay'])
    )
    
    device = get_device(cfg)

    ppo_params = {
        'learning_rate': best_params['learning_rate'], 'n_steps': best_params['n_steps'],
        'batch_size': best_params['batch_size'], 'n_epochs': cfg.training.ppo_n_epochs,
        'gamma': best_params['gamma'], 'gae_lambda': best_params['gae_lambda'],
        'clip_range': best_params['clip_range'], 'ent_coef': best_params['ent_coef'],
        'vf_coef': best_params['vf_coef'], 'max_grad_norm': best_params['max_grad_norm'],
        'verbose': 1, 'device': device
    }
    
    final_model = PPO(MambaActorCriticPolicy, final_env, policy_kwargs=policy_kwargs, **ppo_params)
    
    # <<< ENHANCED LOGGING: Add episode progress tracking >>>
    log.info(f"Starting training for {cfg.training.total_timesteps} timesteps...")
    
    # Add callback for progress monitoring
    class ProgressCallback:
        def __init__(self, log_freq=1000):
            self.log_freq = log_freq
            self.n_calls = 0
            
        def __call__(self, locals_, globals_):
            self.n_calls += 1
            if self.n_calls % self.log_freq == 0:
                log.info(f"Training progress: {self.n_calls} steps completed")
            return True
    
    progress_callback = ProgressCallback()
    final_model.learn(total_timesteps=cfg.training.total_timesteps, callback=progress_callback)
    
    # <<< CRITICAL FIX: Save model to SageMaker directory >>>
    # Save to SageMaker model directory (creates model.tar.gz)
    final_model_sagemaker_path = os.path.join(sagemaker_model_dir, "final_model")
    final_model.save(final_model_sagemaker_path)
    log.info(f"Final model saved to SageMaker model directory: {final_model_sagemaker_path}")
    
    # Also save to output directory for local access
    final_model_local_path = os.path.join(output_dir, cfg.saving.final_model_filename)
    final_model.save(final_model_local_path)
    log.info(f"Final model also saved locally to: {final_model_local_path}")
    
    # <<< ADDITIONAL: Save training metadata to SageMaker model directory >>>
    metadata = {
        'best_params': best_params,
        'feature_cols': feature_cols,
        'model_config': final_full_cfg,
        'training_timesteps': cfg.training.total_timesteps
    }
    metadata_path = os.path.join(sagemaker_model_dir, "training_metadata.pkl")
    joblib.dump(metadata, metadata_path)
    log.info(f"Training metadata saved to: {metadata_path}")

    log.info("Evaluating final model on unseen test data...")
    evaluate_agent(final_model, df_test_s, feature_cols, final_full_cfg, output_dir=output_dir)
    
    log.info("--- Model Training & Evaluation Pipeline COMPLETED ---")
    log.info(f"Model artifacts saved to SageMaker model directory: {sagemaker_model_dir}")
    log.info("SageMaker will automatically create model.tar.gz from this directory")
