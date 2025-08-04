# src/model/mamba_extractor.py
# <<< FINAL CORRECTED VERSION: Moves input tensor to the correct GPU device. >>>

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging
from typing import Optional, Type

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    raise ImportError("mamba-ssm is required. Please run: pip install mamba-ssm 'causal-conv1d>=1.1.0'")

log = logging.getLogger(__name__)

class MambaFeaturesExtractor(BaseFeaturesExtractor):
    """
    A features extractor for DRL that uses Mamba (SSM) layers to process
    time-series observations. Includes inter-layer normalization for stability.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        mamba_d_model: int = 128,
        num_mamba_layers: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        dropout_rate: float = 0.1,
        activation_fn_class: Type[nn.Module] = nn.ReLU,
        input_nan_inf_replacement: float = 0.0,
        input_clipping_value: Optional[float] = None,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        feature_obs_space = observation_space.spaces['features']
        self.num_raw_features = feature_obs_space.shape[1]
        
        self.input_proj = nn.Linear(self.num_raw_features, mamba_d_model)
        self.dropout = nn.Dropout(dropout_rate)

        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=mamba_d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
            for _ in range(num_mamba_layers)
        ])
        
        self.inter_layer_norms = nn.ModuleList([nn.LayerNorm(mamba_d_model) for _ in range(num_mamba_layers)])
        
        self.fc_out = nn.Linear(mamba_d_model, features_dim)
        self.activation_out = activation_fn_class()

        self.input_nan_inf_replacement = input_nan_inf_replacement
        self.input_clipping_value = input_clipping_value

    def _clean_input(self, x: torch.Tensor) -> torch.Tensor:
        """Replaces NaNs/Infs and applies clipping."""
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=self.input_nan_inf_replacement, posinf=1e6, neginf=-1e6)
        if self.input_clipping_value is not None:
            x = torch.clamp(x, -self.input_clipping_value, self.input_clipping_value)
        return x

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        x = observations['features']

        # <<< THE FIX IS HERE: Move the input tensor to the same device as the model. >>>
        # This is the critical step that solves the "Expected x.is_cuda()" error.
        device = next(self.parameters()).device
        x = x.to(device).float()
        
        x = self._clean_input(x)
        
        x = self.input_proj(x)
        x = self.dropout(x)

        for i, mamba_block in enumerate(self.mamba_layers):
            x = mamba_block(x)
            x = self.inter_layer_norms[i](x)
            if torch.isnan(x).any() or torch.isinf(x).any():
                log.error(f"NaN/Inf detected after Mamba layer {i+1}. Returning zeros.")
                return torch.zeros(x.shape[0], self._features_dim, device=x.device)

        last_time_step_features = x[:, -1, :]
        
        extracted_features = self.fc_out(last_time_step_features)
        extracted_features = self.activation_out(extracted_features)
        
        return extracted_features
