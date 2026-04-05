"""Transformer-style feature extractor for SB3 policies.

AttentionFeaturesExtractor replaces the flat MLP with a Multi-Head
Self-Attention block operating over the time dimension of the observation.

Architecture
------------
  Input obs  : (batch, seq_len * n_features)     — flat from the env
  Reshape    : (batch, seq_len, n_features)
  InputProj  : Linear(n_features → d_model)
  Self-Attn  : MultiheadAttention(d_model, num_heads) + residual + LayerNorm
  FFN        : Linear→GELU→Dropout→Linear + residual + LayerNorm
  Mean-pool  : (batch, d_model)   — attend globally over all past candles
  OutHead    : Linear→ReLU → (batch, features_dim)

This allows the policy to directly "attend" to specific past candles
(e.g., a FII/DII spike 10 days ago) rather than relying on the LSTM's
fading hidden state.

The Critic (value function) uses a separate, deeper network via the
``net_arch = {"pi": [...], "vf": [...]}`` dict in policy_kwargs,
preventing the Critic's convergence difficulties from pulling the Actor
in the wrong direction on volatile data.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """Multi-head self-attention feature extractor for flat sequential observations."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        seq_len: int = 15,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(observation_space, features_dim)

        obs_dim = int(np.prod(observation_space.shape))
        n_features = obs_dim // seq_len

        self.seq_len = seq_len
        self.n_features = n_features

        # d_model must be divisible by num_heads; round up to nearest multiple
        d_model = max(n_features, num_heads * 8)
        d_model = ((d_model + num_heads - 1) // num_heads) * num_heads

        # Project raw features to d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # Transformer encoder block
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Output head: mean-pooled representation → features_dim
        self.out_head = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]

        # Reshape flat obs → (batch, seq_len, n_features)
        x = observations.view(batch, self.seq_len, self.n_features)

        # Input projection
        x = self.input_proj(x)                          # (batch, seq_len, d_model)

        # Self-attention with residual connection
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)                    # (batch, seq_len, d_model)

        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)                      # (batch, seq_len, d_model)

        # Global mean-pool over the time dimension
        x = x.mean(dim=1)                               # (batch, d_model)

        return self.out_head(x)                         # (batch, features_dim)
