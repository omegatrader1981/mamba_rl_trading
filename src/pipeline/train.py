# src/pipeline/train.py
import logging
import time
import hydra
from omegaconf import DictConfig
import os

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from src.pipeline.sagemaker_compat import adapt_to_sagemaker
from src.pipeline.data_pipeline import prepare_data
from src.environment import FuturesTradingEnv
from src.model import MambaFeaturesExtractor
from src.utils.torch_utils import get_device
from src.evaluation.evaluate import evaluate_agent

# Adapt to SageMaker before Hydra initializes
adapt_to_sagemaker()

log = logging.getLogger(__name__)
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conf"))

@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    log.info("STARTING PROJECT PHOENIX - DRL TRADING PIPELINE")
    log.info(f"Training timesteps configured: {cfg.training.total_timesteps}")
   
    # --- STAGE 1: DATA PREPARATION ---
    df_train, df_val, df_test, feature_cols, scaler = prepare_data(cfg)
    log.info(f"Features: {feature_cols}")
    log.info(f"Train: {len(df_train)} rows, Val: {len(df_val)} rows, Test: {len(df_test)} rows")
   
    # --- STAGE 2: CREATE ENVIRONMENT ---
    log.info("Creating training environment...")
    train_env = DummyVecEnv([lambda: FuturesTradingEnv(df_train, feature_cols, cfg)])
   
    # --- STAGE 3: CREATE SAC MODEL ---
    log.info("Initializing SAC with Mamba backbone...")
   
    policy_kwargs = dict(
        features_extractor_class=MambaFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            mamba_d_model=64,
            num_mamba_layers=2,
            mamba_d_state=16,
            dropout_rate=0.1,
        )
    )
   
    device = get_device(cfg)
    log.info(f"Using device: {device}")
   
    model = SAC(
        SACPolicy,
        train_env,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=256,
        tau=0.01,
        gamma=0.99,
        train_freq=16,
        gradient_steps=16,
        ent_coef='auto',
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs
    )
   
    # --- STAGE 4: TRAIN ---
    log.info(f"Starting training for {cfg.training.total_timesteps} timesteps...")
    model.learn(total_timesteps=cfg.training.total_timesteps)
   
    # --- STAGE 5: SAVE MODEL ---
    model_path = f"{cfg.saving.output_dir}/model_final.zip"
    os.makedirs(cfg.saving.output_dir, exist_ok=True)
    model.save(model_path)
    log.info(f"Model saved to {model_path}")
   
    train_env.close()
   
    # --- STAGE 6: EVALUATE ON TEST SET ---
    log.info("Starting evaluation on test set...")
    test_metrics = evaluate_agent(
        model=model,
        df_eval=df_test,
        feature_cols=feature_cols,
        cfg=cfg,
        output_dir=cfg.saving.output_dir,
        log_prefix="Test"
    )
   
    log.info(f"Test Metrics: {test_metrics}")
    if 'sharpe' in test_metrics:
        log.info(f"Test Sharpe: {test_metrics['sharpe']:.3f}")
    if 'sortino' in test_metrics:
        log.info(f"Test Sortino: {test_metrics['sortino']:.3f}")
    if 'max_drawdown' in test_metrics:
        log.info(f"Max Drawdown: {test_metrics['max_drawdown']:.2%}")
   
    elapsed_time = time.time() - start_time
    log.info(f"PIPELINE FINISHED SUCCESSFULLY in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
