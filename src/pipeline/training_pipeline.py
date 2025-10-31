# <<< DEFINITIVE FINAL VERSION: With correct HPO guard clause. >>>
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
from src.evaluation import evaluate_agent
from src.optimize.model_builder import create_ppo_model, create_sac_model
from src.environment import FuturesTradingEnv
from src.utils.sagemaker_utils import initialize_sagemaker_environment, sagemaker_safe

log = logging.getLogger(__name__)

@sagemaker_safe
def train_and_evaluate_model(cfg: DictConfig, df_train_s: pd.DataFrame, df_val_s: pd.DataFrame, df_test_s: pd.DataFrame, feature_cols: List[str], scaler: object):
    cfg = initialize_sagemaker_environment(cfg)
    log.info("--- Starting Model Training & Evaluation Pipeline ---")
    
    checkpoint_dir = cfg.get("checkpointing", {}).get("s3_base_path", "/opt/ml/checkpoints")
    sagemaker_model_dir = "/opt/ml/model"
    sagemaker_output_dir = cfg.saving.output_dir
    
    if cfg.optimization.get("enabled", False):
        log.info("🚀 Optimization is ENABLED. Starting HPO...")
        from src.optimize.hpo_runner import run_optimization
        study, best_params = run_optimization(cfg, df_train_s, df_val_s)
        if not best_params: 
            raise RuntimeError("Hyperparameter optimization failed.")
        joblib.dump(best_params, os.path.join(sagemaker_output_dir, cfg.saving.best_params_filename))
        log.info(f"Best HPO parameters saved.")
    else:
        best_params = None
        log.info("⏩ Optimization is DISABLED. Using default hyperparameters.")
    
    log.info("Preparing for final model training...")
    df_train_val_s = pd.concat([df_train_s, df_val_s]).sort_index()
    final_full_cfg = cfg.copy()
    OmegaConf.set_struct(final_full_cfg, False)
    final_full_cfg.env.lookback_window = best_params.get('lookback_window') if best_params else cfg.env.lookback_window
    
    agent_type = final_full_cfg.experiment.get('agent_type', 'ppo')
    final_env = DummyVecEnv([lambda: FuturesTradingEnv(df_train_val_s, feature_cols, final_full_cfg)])
    
    if best_params:
        class DummyTrial:
            def __init__(self, params): 
                self.params = params
            def suggest_float(self, name, low, high, log=False): 
                return self.params.get(name, (low + high) / 2)
            def suggest_categorical(self, name, choices): 
                return self.params.get(name, choices[0])
            def suggest_int(self, name, low, high): 
                return self.params.get(name, (low + high) // 2)
        
        dummy_trial = DummyTrial(best_params)
        if agent_type == 'sac': 
            final_model = create_sac_model(dummy_trial, final_env, final_full_cfg)
        else: 
            final_model = create_ppo_model(dummy_trial, final_env, final_full_cfg)
    else:
        log.info("Building model with default config (no HPO)")
        if agent_type == 'sac': 
            final_model = SAC("MlpPolicy", final_env, verbose=1, device="auto")
        else: 
            final_model = PPO("MlpPolicy", final_env, verbose=1, device="auto")
    
    log.info(f"Starting final training for {final_full_cfg.training.total_timesteps} timesteps...")
    final_model.learn(total_timesteps=final_full_cfg.training.total_timesteps)
    
    final_model.save(os.path.join(sagemaker_model_dir, "final_model.zip"))
    log.info(f"Final model saved to SageMaker model directory.")
    
    final_model.save(os.path.join(sagemaker_output_dir, cfg.saving.final_model_filename))
    log.info(f"Final model also saved to output directory.")
    
    log.info("Evaluating final model on unseen test data...")
    evaluate_agent(final_model, df_test_s, feature_cols, final_full_cfg, output_dir=sagemaker_output_dir)
    
    log.info("--- Model Training & Evaluation Pipeline COMPLETED ---")
