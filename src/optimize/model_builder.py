# <<< UPDATED FOR PHASE 6: Now includes a builder for the SAC agent. >>>
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecEnv
import optuna
from omegaconf import DictConfig
from src.model import MambaFeaturesExtractor, MambaActorCriticPolicy
from src.torch_utils import get_device
def create_ppo_model(trial: optuna.Trial, env: VecEnv, cfg: DictConfig) -> PPO:
    lr = trial.suggest_float("learning_rate", cfg.optimization.lr_min, cfg.optimization.lr_max, log=True)
    n_steps = trial.suggest_categorical("n_steps", cfg.optimization.ppo_n_steps_choices_hpo)
    batch_size = trial.suggest_categorical("batch_size", cfg.optimization.batch_size_choices)
    gamma = trial.suggest_float("gamma", cfg.optimization.gamma_min, cfg.optimization.gamma_max, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", cfg.optimization.gae_lambda_min, cfg.optimization.gae_lambda_max)
    ent_coef = trial.suggest_float("ent_coef", cfg.optimization.ent_coef_min, cfg.optimization.ent_coef_max)
    vf_coef = trial.suggest_float("vf_coef", cfg.optimization.vf_coef_min, cfg.optimization.vf_coef_max)
    max_grad_norm = trial.suggest_float("max_grad_norm", cfg.optimization.max_grad_norm_min, cfg.optimization.max_grad_norm_max)
    clip_range = trial.suggest_float("clip_range", cfg.optimization.clip_range_min, cfg.optimization.clip_range_max)
    weight_decay = trial.suggest_float("weight_decay", cfg.optimization.weight_decay_min, cfg.optimization.weight_decay_max)
    num_mamba_layers = trial.suggest_int("num_mamba_layers", cfg.optimization.mamba_layers_min, cfg.optimization.mamba_layers_max)
    mamba_d_model = trial.suggest_categorical("mamba_d_model", cfg.optimization.mamba_d_model_choices)
    mamba_d_state = trial.suggest_categorical("mamba_d_state", cfg.optimization.mamba_d_state_choices)
    features_dim = trial.suggest_categorical("features_dim", cfg.optimization.features_dim_choices)
    dropout_rate = trial.suggest_float("dropout_rate", cfg.optimization.dropout_min, cfg.optimization.dropout_max)
    act_fn_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'silu': nn.SiLU}
    mlp_activation = act_fn_map.get(cfg.model.mlp_activation_fn_name, nn.ReLU)
    mamba_activation = act_fn_map.get(cfg.model.mamba_activation_fn_name, nn.SiLU)
    policy_kwargs = dict(features_extractor_class=MambaFeaturesExtractor, features_extractor_kwargs=dict(features_dim=features_dim, mamba_d_model=mamba_d_model, num_mamba_layers=num_mamba_layers, mamba_d_state=mamba_d_state, dropout_rate=dropout_rate, activation_fn_class=mamba_activation, mamba_d_conv=cfg.model.mamba_d_conv_default, mamba_expand=cfg.model.mamba_expand_default), activation_fn=mlp_activation, optimizer_kwargs=dict(weight_decay=weight_decay))
    device = get_device(cfg)
    ppo_params = {'learning_rate': lr, 'n_steps': n_steps, 'batch_size': batch_size, 'n_epochs': cfg.training.ppo_n_epochs, 'gamma': gamma, 'gae_lambda': gae_lambda, 'clip_range': clip_range, 'ent_coef': ent_coef, 'vf_coef': vf_coef, 'max_grad_norm': max_grad_norm, 'verbose': 0, 'device': device}
    model = PPO(MambaActorCriticPolicy, env, policy_kwargs=policy_kwargs, **ppo_params)
    return model
def create_sac_model(trial: optuna.Trial, env: VecEnv, cfg: DictConfig) -> SAC:
    lr = trial.suggest_float("learning_rate", cfg.optimization.lr_min, cfg.optimization.lr_max, log=True)
    batch_size = trial.suggest_categorical("batch_size", cfg.optimization.batch_size_choices)
    buffer_size = trial.suggest_categorical("buffer_size", cfg.optimization.buffer_size_choices)
    gamma = trial.suggest_float("gamma", cfg.optimization.gamma_min, cfg.optimization.gamma_max, log=True)
    tau = trial.suggest_float("tau", cfg.optimization.tau_min, cfg.optimization.tau_max)
    train_freq = trial.suggest_categorical("train_freq", cfg.optimization.train_freq_choices)
    gradient_steps = train_freq
    weight_decay = trial.suggest_float("weight_decay", cfg.optimization.weight_decay_min, cfg.optimization.weight_decay_max)
    num_mamba_layers = trial.suggest_int("num_mamba_layers", cfg.optimization.mamba_layers_min, cfg.optimization.mamba_layers_max)
    mamba_d_model = trial.suggest_categorical("mamba_d_model", cfg.optimization.mamba_d_model_choices)
    mamba_d_state = trial.suggest_categorical("mamba_d_state", cfg.optimization.mamba_d_state_choices)
    features_dim = trial.suggest_categorical("features_dim", cfg.optimization.features_dim_choices)
    dropout_rate = trial.suggest_float("dropout_rate", cfg.optimization.dropout_min, cfg.optimization.dropout_max)
    act_fn_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'silu': nn.SiLU}
    mlp_activation = act_fn_map.get(cfg.model.mlp_activation_fn_name, nn.ReLU)
    mamba_activation = act_fn_map.get(cfg.model.mamba_activation_fn_name, nn.SiLU)
    policy_kwargs = dict(features_extractor_class=MambaFeaturesExtractor, features_extractor_kwargs=dict(features_dim=features_dim, mamba_d_model=mamba_d_model, num_mamba_layers=num_mamba_layers, mamba_d_state=mamba_d_state, dropout_rate=dropout_rate, activation_fn_class=mamba_activation, mamba_d_conv=cfg.model.mamba_d_conv_default, mamba_expand=cfg.model.mamba_expand_default), activation_fn=mlp_activation, optimizer_kwargs=dict(weight_decay=weight_decay))
    device = get_device(cfg)
    sac_params = {'learning_rate': lr, 'buffer_size': buffer_size, 'learning_starts': cfg.optimization.learning_starts, 'batch_size': batch_size, 'tau': tau, 'gamma': gamma, 'train_freq': (train_freq, "step"), 'gradient_steps': gradient_steps, 'ent_coef': 'auto', 'verbose': 0, 'device': device}
    model = SAC(MambaActorCriticPolicy, env, policy_kwargs=policy_kwargs, **sac_params)
    return model
