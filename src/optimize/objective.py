import optuna
import pandas as pd
import logging
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment import FuturesTradingEnv
from src.optimize.model_builder import create_ppo_model, create_sac_model
from src.evaluation import evaluate_agent

log = logging.getLogger(__name__)

class Objective:
    def __init__(self, cfg: DictConfig, df_train: pd.DataFrame, df_val: pd.DataFrame):
        self.cfg = cfg
        self.df_train = df_train
        self.df_val = df_val
        self.feature_cols = [col for col in df_train.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]

    def __call__(self, trial: optuna.Trial) -> float:
        try:
            trial_cfg = self.cfg.copy()
            OmegaConf.set_struct(trial_cfg, False)

            # --- 🔻 THE FIX: Use the correct config key 'env.lookback_window' --- 🔻
            # Also, suggest the parameter from Optuna
            lookback_window = trial.suggest_int('lookback_window', 
                                                self.cfg.optimization.get('lookback_window_min', 20), 
                                                self.cfg.optimization.get('lookback_window_max', 300))
            trial_cfg.env.lookback_window = lookback_window
            # --- 🔺 END OF FIX ---

            agent_type = trial_cfg.experiment.get('agent_type', 'ppo')
            
            train_env = DummyVecEnv([lambda: FuturesTradingEnv(self.df_train, self.feature_cols, trial_cfg)])
            
            if agent_type == 'sac':
                model = create_sac_model(trial, train_env, trial_cfg)
            else:
                model = create_ppo_model(trial, train_env, trial_cfg)

            model.learn(total_timesteps=trial_cfg.optimization.trial_timesteps)

            eval_results = evaluate_agent(model, self.df_val, self.feature_cols, trial_cfg, is_hpo_trial=True)
            
            metric = eval_results.get(self.cfg.optimization.metric, -1.0)
            log.info(f"Trial {trial.number} finished. {self.cfg.optimization.metric}: {metric:.4f}")
            
            return metric
        except Exception as e:
            log.error(f"Trial #{trial.number}: Objective function failed: {e}", exc_info=True)
            return -1.0 # Return a poor score to penalize failing trials
